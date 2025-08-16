import datetime
import json
import os
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    JSON,
    func,
    Text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, selectinload

# Slack + LLM
from slack_sdk.web.async_client import AsyncWebClient
from litellm import acompletion

# LangChain (modern package split)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

import faiss  # pip install faiss-cpu

load_dotenv()

# =========================
# Config
# =========================
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_ALERT_CHANNEL_ID = os.environ.get("SLACK_ALERT_CHANNEL_ID")

# AI keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")       # for email drafting (LiteLLM -> Groq Llama3)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")   # for agent brain + embeddings

DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Constants
DEFAULT_HEALTH_SCORE = 50
STATUS_AT_RISK_THRESHOLD = 35
STATUS_HEALTHY_THRESHOLD_UPPER = 70
OUR_CSM_PRODUCT_NAME = "EvoMind AI CSM"

# =========================
# DB Models
# =========================
Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    users = relationship("User", back_populates="company")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    company = relationship("Company", back_populates="users")
    events = relationship("Event", back_populates="user", order_by="desc(Event.timestamp)")
    health_scores = relationship("HealthScore", back_populates="user", order_by="desc(HealthScore.calculated_at)")
    interventions = relationship("Intervention", back_populates="user", order_by="desc(Intervention.sent_at)")

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    event_name = Column(String, index=True)
    properties = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user = relationship("User", back_populates="events")

class HealthScore(Base):
    __tablename__ = "health_scores"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    score = Column(Integer, default=DEFAULT_HEALTH_SCORE)
    status = Column(String, default="Neutral")
    calculated_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user = relationship("User", back_populates="health_scores")

class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    intervention_type = Column(String, index=True)
    channel = Column(String)
    content_template = Column(String)
    content_sent = Column(Text)
    status = Column(String, default="sent")  # "sent", "draft", "failed"
    sent_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user_response = Column(String, nullable=True)
    user = relationship("User", back_populates="interventions")

# =========================
# Pydantic Schemas (Pydantic v2 style)
# =========================
class CompanyBase(BaseModel):
    name: str

class CompanyCreate(CompanyBase):
    pass

class CompanyResponse(CompanyBase):
    id: int
    created_at: datetime.datetime
    class Config:
        from_attributes = True

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    company_id: int

class UserResponse(UserBase):
    id: int
    company_id: int
    created_at: datetime.datetime
    company: Optional[CompanyResponse] = None
    class Config:
        from_attributes = True

class EventBase(BaseModel):
    event_name: str
    properties: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime.datetime] = None

class EventCreate(EventBase):
    user_email: EmailStr

class EventResponse(EventBase):
    id: int
    user_id: int
    timestamp: datetime.datetime
    class Config:
        from_attributes = True

class HealthScoreBase(BaseModel):
    score: int
    status: str

class HealthScoreCreate(HealthScoreBase):
    user_id: int

class HealthScoreResponse(HealthScoreBase):
    id: int
    user_id: int
    calculated_at: datetime.datetime
    class Config:
        from_attributes = True

class InterventionBase(BaseModel):
    intervention_type: str
    channel: str
    content_template: str
    content_sent: str
    status: Optional[str] = "sent"
    user_response: Optional[str] = None

class InterventionCreate(InterventionBase):
    user_id: int

class InterventionResponse(InterventionBase):
    id: int
    user_id: int
    sent_at: datetime.datetime
    status: str
    class Config:
        from_attributes = True

class ManualNudgePayload(BaseModel):
    message: str = Field(default="", min_length=0, max_length=10000)
    ai_assist_topic: Optional[str] = Field(None, min_length=1, max_length=200)
    draft_only: Optional[bool] = Field(False)

# =========================
# DB setup
# =========================
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Agentic AI CSM System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helpers (Email / Slack / AI draft)
# =========================
async def send_email_notification(to_email: str, subject: str, body: str) -> bool:
    print(
        f"--- SENDING EMAIL ---\nTo: {to_email}\nSubject: {subject}\nBody:\n{body}\n--- EMAIL SENT (console) ---"
    )
    return True

async def send_slack_alert(message: str, channel_id: Optional[str] = SLACK_ALERT_CHANNEL_ID) -> bool:
    if not SLACK_BOT_TOKEN or not channel_id:
        print(f"[Slack console] {message}")
        return False
    try:
        client = AsyncWebClient(token=SLACK_BOT_TOKEN)
        await client.chat_postMessage(channel=channel_id, text=message)
        return True
    except Exception as e:
        print(f"Error sending Slack alert: {e}")
        return False

async def generate_ai_message(prompt: str, system_message: str) -> Optional[str]:
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set. Drafting falls back to default text.")
        return "This is a default message because AI generation is unavailable."
    try:
        response = await acompletion(
            model="groq/llama3-8b-8192",
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700,
            api_key=GROQ_API_KEY,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI draft error: {e}")
        return "We wanted to touch base regarding your recent usage…"

def parse_subject_body(ai_content: str, default_subject: str) -> Tuple[str, str]:
    subject, body = default_subject, ai_content
    try:
        if "Subject:" in ai_content and "\nBody:\n" in ai_content:
            parts = ai_content.split("\nBody:\n", 1)
            subject_line = parts[0].replace("Subject:", "").strip()
            if subject_line:
                subject = subject_line
            body = parts[1].strip()
        elif "Subject:" in ai_content:
            subject_line = ai_content.split("\n", 1)[0].replace("Subject:", "").strip()
            if subject_line:
                subject = subject_line
            body = ai_content.split("\n", 1)[1].strip() if "\n" in ai_content else ""
    except Exception:
        pass
    return subject, body

# =========================
# Memory / Vector store
# =========================
# Use OpenAI embeddings for the vector store (needs OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Start with an empty FAISS index
# We'll add texts with metadata: {"user_id": <id>, "timestamp": <iso>}
vector_store = FAISS(
    embedding_function=embeddings.embed_query,
    index=faiss.IndexFlatL2(1536),  # 1536 = text-embedding-3-small size
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={},
)

async def add_to_memory(user_id: int, content: str):
    # store small snippets with user_id metadata for filtering
    vector_store.add_texts([content], metadatas=[{"user_id": user_id, "timestamp": datetime.datetime.utcnow().isoformat()}])

async def query_memory(user_id: int, query: str, k: int = 3) -> List[str]:
    # FAISS metadata filtering in LangChain supports exact-match dict filters
    docs = vector_store.similarity_search(query, k=k, filter={"user_id": user_id})
    return [d.page_content for d in docs]

# =========================
# Agent Tools (async)
# =========================
async def tool_send_email(user_id: int, subject: str, body: str, db: AsyncSession) -> str:
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        return "User not found"
    success = await send_email_notification(user.email, subject, body)
    status = "sent" if success else "failed"
    intervention = Intervention(
        user_id=user_id,
        intervention_type="agent_email",
        channel="email",
        content_template="agent_generated",
        content_sent=f"Subject: {subject}\n\n{body}",
        status=status,
    )
    db.add(intervention)
    await db.commit()
    await add_to_memory(user_id, f"Email to {user.email}: {subject}")
    return "Email sent" if success else "Email failed"

async def tool_send_slack_alert(user_id: int, message: str, db: AsyncSession) -> str:
    user = (
        await db.execute(
            select(User).options(selectinload(User.company)).where(User.id == user_id)
        )
    ).scalars().first()
    if not user:
        return "User not found"
    full = f"User {user.email} ({user.company.name if user.company else 'no company'}): {message}"
    success = await send_slack_alert(full)
    status = "sent" if success else "failed"
    intervention = Intervention(
        user_id=user_id,
        intervention_type="agent_slack_alert",
        channel="slack_internal",
        content_template="agent_generated",
        content_sent=full,
        status=status,
    )
    db.add(intervention)
    await db.commit()
    await add_to_memory(user_id, f"Slack alert about user {user_id}: {message}")
    return "Slack sent" if success else "Slack failed"

async def tool_calculate_health_score(user_id: int, db: AsyncSession) -> Tuple[Optional[HealthScore], str]:
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        return None, "User not found"

    now = datetime.datetime.utcnow()
    events_res = await db.execute(
        select(Event).where(
            Event.user_id == user_id,
            Event.timestamp >= now - datetime.timedelta(days=30),
        )
    )
    recent_events = events_res.scalars().all()

    score = DEFAULT_HEALTH_SCORE

    # + recent login
    if any(e.event_name == "logged_in" and e.timestamp >= now - datetime.timedelta(days=7) for e in recent_events):
        score += 10

    # + key feature usage
    key_feature_events = ["viewed_dashboard", "created_report"]
    key_counts: Dict[str, int] = {}
    for e in recent_events:
        if e.event_name in key_feature_events:
            key_counts[e.event_name] = key_counts.get(e.event_name, 0) + 1
    for count in key_counts.values():
        if count >= 3:
            score += 15

    # - billing issues
    negative_billing_events = ["billing_failed", "subscription_cancelled", "payment_declined"]
    if any(e.event_name in negative_billing_events for e in recent_events):
        score -= 30

    # - inactivity
    latest_event_ts_res = await db.execute(select(func.max(Event.timestamp)).where(Event.user_id == user_id))
    latest_event_ts = latest_event_ts_res.scalars().first()
    if not latest_event_ts or latest_event_ts < now - datetime.timedelta(days=14):
        score -= 10

    score = max(0, min(100, score))

    status = "Neutral"
    if score <= STATUS_AT_RISK_THRESHOLD:
        status = "At Risk"
    elif score > STATUS_HEALTHY_THRESHOLD_UPPER:
        status = "Power User"
    elif score > STATUS_AT_RISK_THRESHOLD:
        status = "Healthy"

    new_hs = HealthScore(user_id=user_id, score=score, status=status, calculated_at=now)
    db.add(new_hs)
    await db.commit()
    await db.refresh(new_hs)
    await add_to_memory(user_id, f"Health score {score} ({status})")
    return new_hs, f"Health score calculated: {score} ({status})"

async def tool_get_user_context(user_id: int, db: AsyncSession) -> str:
    user = (
        await db.execute(
            select(User).options(
                selectinload(User.company),
                selectinload(User.events),
                selectinload(User.health_scores),
                selectinload(User.interventions),
            ).where(User.id == user_id)
        )
    ).scalars().first()

    if not user:
        return "User not found"

    context = {
        "user_id": user.id,
        "email": user.email,
        "company": user.company.name if user.company else "None",
        "created_at": user.created_at.isoformat(),
        "health_score": user.health_scores[0].score if user.health_scores else DEFAULT_HEALTH_SCORE,
        "health_status": user.health_scores[0].status if user.health_scores else "Neutral",
        "last_event": user.events[0].event_name if user.events else "None",
        "last_event_time": user.events[0].timestamp.isoformat() if user.events else "None",
        "intervention_count": len(user.interventions),
        "last_intervention": user.interventions[0].intervention_type if user.interventions else "None",
    }

    memory_context = await query_memory(user_id, f"Recent interactions with user {user.id}")
    context["memory_context"] = memory_context

    return json.dumps(context, indent=2)

# =========================
# Agent Initialization
# =========================
def initialize_csm_agent(db: AsyncSession) -> AgentExecutor:
    """
    Build an OpenAI Functions Agent with async tools.
    """
    # Define tools with async coroutines; func is a sync fallback (unused)
    tools = [
        Tool(
            name="SendEmail",
            description="Send an email to a user. Args: user_id:int, subject:str, body:str",
            func=lambda user_id, subject, body: "Use async mode",  # fallback (unused)
            coroutine=lambda user_id, subject, body: tool_send_email(int(user_id), str(subject), str(body), db),
        ),
        Tool(
            name="SendSlackAlert",
            description="Alert CSM team via Slack about a user. Args: user_id:int, message:str",
            func=lambda user_id, message: "Use async mode",
            coroutine=lambda user_id, message: tool_send_slack_alert(int(user_id), str(message), db),
        ),
        Tool(
            name="CalculateHealthScore",
            description="Calculate/recalculate a user's health score. Args: user_id:int",
            func=lambda user_id: "Use async mode",
            coroutine=lambda user_id: tool_calculate_health_score(int(user_id), db),
        ),
        Tool(
            name="GetUserContext",
            description="Get detailed JSON context for a user. Args: user_id:int",
            func=lambda user_id: "Use async mode",
            coroutine=lambda user_id: tool_get_user_context(int(user_id), db),
        ),
    ]

    system_message = SystemMessage(
        content=(
            f"You are an AI Customer Success Manager (CSM) for {OUR_CSM_PRODUCT_NAME}. "
            "Goals:\n"
            "1) Retain users by preventing churn\n"
            "2) Identify upsell opportunities\n"
            "3) Ensure customer satisfaction and engagement\n\n"
            "You have tools to email users, alert CSMs in Slack, calculate health scores, and fetch user context. "
            "Be professional, consider history, document actions, and focus on long-term relationships."
        )
    )

    # LLM for reasoning (OpenAI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

    # Build the agent
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    # Wrap with executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor

# =========================
# Agentic Processing
# =========================
async def process_user_with_agent(user_id: int, db: AsyncSession) -> bool:
    agent = initialize_csm_agent(db)

    # Pull user with context
    user = (
        await db.execute(
            select(User).options(
                selectinload(User.company),
                selectinload(User.events),
                selectinload(User.health_scores),
                selectinload(User.interventions),
            ).where(User.id == user_id)
        )
    ).scalars().first()

    if not user:
        return False

    # Ensure there is at least one health score
    if not user.health_scores:
        await tool_calculate_health_score(user_id, db)

    latest_hs = user.health_scores[0] if user.health_scores else None
    prompt_text = (
        f"User {user.email} from {user.company.name if user.company else 'Unknown company'} needs attention.\n"
        f"Current health score: {latest_hs.score if latest_hs else DEFAULT_HEALTH_SCORE} "
        f"({latest_hs.status if latest_hs else 'Neutral'})\n"
        f"Last activity: {user.events[0].event_name if user.events else 'No recent activity'}\n"
        f"Interventions in last 30 days: "
        f"{len([i for i in user.interventions if i.sent_at >= datetime.datetime.utcnow() - datetime.timedelta(days=30)])}\n\n"
        "Analyze this user and take appropriate actions to:\n"
        "1) Improve engagement if at risk\n"
        "2) Identify upsell opportunities if a power user\n"
        "3) Address immediate issues like billing problems\n\n"
        "Use the available tools to complete the tasks."
    )

    try:
        # Pass the prompt via the expected key "input"
        result = await agent.arun({"input": prompt_text})
        await add_to_memory(user_id, f"Agent run result summary: {str(result)[:300]}")
        return True
    except Exception as e:
        print(f"Agent error for user {user_id}: {e}")
        await send_slack_alert(f"⚠️ Agent error for user {user.email}: {str(e)}")
        return False

# =========================
# Startup seed
# =========================
@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()
    async with AsyncSessionLocal() as s:
        async with s.begin():
            c = (await s.execute(select(Company).where(Company.name == "TestCompany Inc."))).scalars().first()
            if not c:
                c = Company(name="TestCompany Inc.")
                s.add(c)
                await s.flush()
            u = (
                await s.execute(
                    select(User)
                    .options(selectinload(User.company))
                    .where(User.email == "testuser@example.com")
                )
            ).scalars().unique().first()
            if not u and c:
                s.add(User(email="testuser@example.com", company_id=c.id))
        await s.commit()

# =========================
# API Endpoints
# =========================
@app.post("/companies/", response_model=CompanyResponse, status_code=201)
async def create_company(c: CompanyCreate, db: AsyncSession = Depends(get_db)):
    dbc = Company(name=c.name)
    db.add(dbc)
    await db.commit()
    await db.refresh(dbc)
    return dbc

@app.get("/companies/", response_model=List[CompanyResponse])
async def read_companies(s: int = 0, l: int = 100, db: AsyncSession = Depends(get_db)):
    return (await db.execute(select(Company).offset(s).limit(l))).scalars().all()

@app.get("/companies/{cid}", response_model=CompanyResponse)
async def read_company(cid: int, db: AsyncSession = Depends(get_db)):
    c = (await db.execute(select(Company).where(Company.id == cid))).scalars().first()
    if not c:
        raise HTTPException(404, "Company not found")
    return c

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(u: UserCreate, db: AsyncSession = Depends(get_db)):
    c = (await db.execute(select(Company).where(Company.id == u.company_id))).scalars().first()
    if not c:
        raise HTTPException(404, f"Company {u.company_id} not found")
    dbu = User(email=u.email, company_id=u.company_id)
    db.add(dbu)
    await db.commit()
    await db.refresh(dbu)
    r = (
        await db.execute(
            select(User).options(selectinload(User.company)).where(User.id == dbu.id)
        )
    ).scalars().unique().first()
    return r

@app.get("/users/", response_model=List[UserResponse])
async def read_users(s: int = 0, l: int = 100, db: AsyncSession = Depends(get_db)):
    st = select(User).options(selectinload(User.company)).order_by(User.id).offset(s).limit(l)
    return (await db.execute(st)).scalars().unique().all()

@app.get("/users/{uid}", response_model=UserResponse)
async def read_user(uid: int, db: AsyncSession = Depends(get_db)):
    st = select(User).options(selectinload(User.company)).where(User.id == uid)
    u = (await db.execute(st)).scalars().unique().first()
    if not u:
        raise HTTPException(404, "User not found")
    return u

@app.post("/ingest_event/", response_model=EventResponse, status_code=201)
async def ingest_event(ed: EventCreate, db: AsyncSession = Depends(get_db)):
    u = (await db.execute(select(User).where(User.email == ed.user_email))).scalars().first()
    if not u:
        raise HTTPException(404, f"User {ed.user_email} not found")
    dbe = Event(
        user_id=u.id,
        event_name=ed.event_name,
        properties=ed.properties,
        timestamp=ed.timestamp or datetime.datetime.utcnow(),
    )
    db.add(dbe)
    await db.commit()
    await db.refresh(dbe)
    return dbe

@app.get("/users/{uid}/events/", response_model=List[EventResponse])
async def read_user_events(uid: int, s: int = 0, l: int = 20, db: AsyncSession = Depends(get_db)):
    exists = (await db.execute(select(User.id).where(User.id == uid))).scalars().first()
    if not exists:
        raise HTTPException(404, "User not found")
    return (
        await db.execute(
            select(Event)
            .where(Event.user_id == uid)
            .order_by(Event.timestamp.desc())
            .offset(s)
            .limit(l)
        )
    ).scalars().all()

@app.get("/events/", response_model=List[EventResponse])
async def read_all_events(s: int = 0, l: int = 100, db: AsyncSession = Depends(get_db)):
    return (await db.execute(select(Event).order_by(Event.timestamp.desc()).offset(s).limit(l))).scalars().all()

@app.post("/users/{uid}/calculate_health_score", response_model=HealthScoreResponse)
async def trigger_user_health_score_calc(uid: int, db: AsyncSession = Depends(get_db)):
    exists = (await db.execute(select(User.id).where(User.id == uid))).scalars().first()
    if not exists:
        raise HTTPException(404, "User not found")
    hs, _ = await tool_calculate_health_score(uid, db)
    if not hs:
        raise HTTPException(500, "Failed to calc health score")
    return hs

@app.get("/users/{uid}/health_scores", response_model=List[HealthScoreResponse])
async def get_user_health_scores(uid: int, l: int = 10, db: AsyncSession = Depends(get_db)):
    sc = (
        await db.execute(
            select(HealthScore)
            .where(HealthScore.user_id == uid)
            .order_by(HealthScore.calculated_at.desc())
            .limit(l)
        )
    ).scalars().all()
    if not sc:
        raise HTTPException(404, "No health scores for user.")
    return sc

@app.get("/users/{uid}/latest_health_score", response_model=Optional[HealthScoreResponse])
async def get_user_latest_health_score(uid: int, db: AsyncSession = Depends(get_db)):
    return (
        await db.execute(
            select(HealthScore)
            .where(HealthScore.user_id == uid)
            .order_by(HealthScore.calculated_at.desc())
            .limit(1)
        )
    ).scalars().first()

@app.post("/admin/run_agentic_processing", status_code=202)
async def run_agentic_processing(bg: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    uids = (await db.execute(select(User.id))).scalars().all()

    async def _process():
        async with AsyncSessionLocal() as tdb:
            processed = 0
            for uid in uids:
                try:
                    if await process_user_with_agent(uid, tdb):
                        processed += 1
                except Exception as e:
                    print(f"Error processing user {uid} with agent: {e}")
                    await send_slack_alert(f"⚠️ Error processing user {uid} with agent: {str(e)}")
            print(f"Agent processing done for {processed}/{len(uids)} users.")

    bg.add_task(_process)
    return {"message": f"Agentic processing started for {len(uids)} users."}

@app.post("/users/{user_id}/send_manual_nudge", status_code=200)
async def send_manual_nudge_to_user(
    user_id: int,
    payload: ManualNudgePayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(User).options(
        selectinload(User.company),
        selectinload(User.events.and_(Event.timestamp >= datetime.datetime.utcnow() - datetime.timedelta(days=30))),
    ).where(User.id == user_id)
    user_result = await db.execute(stmt)
    user = user_result.scalars().unique().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    client_saas_product_name = user.company.name if user.company else "their current platform"
    user_name = user.email.split("@")[0]

    final_message_body = payload.message or ""
    prompt_template_name = "manual_direct_v2"
    ai_generated_draft_for_frontend = None

    if not final_message_body.strip() and payload.ai_assist_topic:
        print(f"🤖 AI drafting manual nudge for topic: '{payload.ai_assist_topic}' for user {user.email}")
        recent_event_names = (
            list(dict.fromkeys(e.event_name for e in user.events[:5]))
            if user.events else
            [f"their general usage of {client_saas_product_name}"]
        )
        prompt = (
            f"User Context:\n- User Name: {user_name}\n- User Email: {user.email}\n"
            f"- User's Company: {client_saas_product_name}\n"
            f"- Recent activities: {', '.join(recent_event_names)}\n\n"
            f"Task: You are an AI assistant for a CSM of '{OUR_CSM_PRODUCT_NAME}'. "
            f"Draft a concise, friendly email body (NO subject) about: '{payload.ai_assist_topic}'. "
            f"Start with 'Hi {user_name},' and keep it helpful for their use of '{client_saas_product_name}'."
        )
        prompt_template_name = f"manual_ai_assist_{payload.ai_assist_topic.replace(' ', '_').lower()[:25]}"
        generated_body = await generate_ai_message(
            prompt,
            f"You help a CSM of '{OUR_CSM_PRODUCT_NAME}' draft outreach to users of '{client_saas_product_name}'."
        )
        final_message_body = (generated_body or "").strip() or (
            f"Hi {user_name},\n\nWe wanted to touch base regarding: {payload.ai_assist_topic} for your use of {client_saas_product_name}."
            "\nPlease let us know if you have any questions.\n\nBest regards,\nThe Team"
        )
        ai_generated_draft_for_frontend = final_message_body

        if payload.draft_only:
            return {
                "message": "AI draft generated successfully.",
                "ai_generated_draft": ai_generated_draft_for_frontend,
                "is_draft": True,
            }

    elif not final_message_body.strip() and not payload.ai_assist_topic:
        raise HTTPException(status_code=400, detail="Either a message or an AI assist topic must be provided.")

    subject = f"A message from {client_saas_product_name} (via {OUR_CSM_PRODUCT_NAME})"

    async def _send_and_record():
        async with AsyncSessionLocal() as task_db:
            email_sent = await send_email_notification(user.email, subject, final_message_body)
            intervention_status = "sent" if email_sent else "failed"
            intervention_type_val = (
                "manual_nudge_email_ai_assisted"
                if (not payload.message.strip() and payload.ai_assist_topic)
                else "manual_nudge_email_direct"
            )
            intervention = Intervention(
                user_id=user.id,
                intervention_type=intervention_type_val,
                channel="email",
                content_template=prompt_template_name,
                content_sent=f"Subject: {subject}\n\n{final_message_body}",
                status=intervention_status,
            )
            task_db.add(intervention)
            await task_db.commit()
            if email_sent:
                await send_slack_alert(
                    f"Manual nudge sent to {user.email} (of {client_saas_product_name}). Type: {intervention_type_val}. Subject: {subject}"
                )

    background_tasks.add_task(_send_and_record)

    return {
        "message": f"Manual nudge to {user.email} has been queued for sending.",
        "ai_generated_draft": None,
        "is_draft": False,
    }

@app.get("/users/{uid}/interventions", response_model=List[InterventionResponse])
async def get_user_interventions(uid: int, l: int = 10, db: AsyncSession = Depends(get_db)):
    r = (
        await db.execute(
            select(Intervention)
            .where(Intervention.user_id == uid)
            .order_by(Intervention.sent_at.desc())
            .limit(l)
        )
    ).scalars().all()
    if not r:
        raise HTTPException(404, "No interventions for user.")
    return r

@app.get("/interventions", response_model=List[InterventionResponse])
async def get_all_interventions(s: int = 0, l: int = 100, db: AsyncSession = Depends(get_db)):
    return (
        await db.execute(
            select(Intervention).order_by(Intervention.sent_at.desc()).offset(s).limit(l)
        )
    ).scalars().all()

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    import uvicorn

    print("--- Agentic AI CSM Service Starting ---")
    print(f"GROQ_API_KEY: {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print(f"OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
    print(f"SLACK_BOT_TOKEN: {'SET' if SLACK_BOT_TOKEN else 'NOT SET'}")
    print(f"SLACK_ALERT_CHANNEL_ID: {SLACK_ALERT_CHANNEL_ID or 'NOT SET'}")
    print("---------------------------------------")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
