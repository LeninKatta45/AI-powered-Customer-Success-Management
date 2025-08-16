# main.py

import datetime
import json
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import selectinload
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, JSON, func, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base # Removed Session from here as it's not directly used
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
import os
from dotenv import load_dotenv

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from litellm import acompletion

load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_ALERT_CHANNEL_ID = os.environ.get("SLACK_ALERT_CHANNEL_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

DATABASE_URL = "sqlite+aiosqlite:///./test.db"

DEFAULT_HEALTH_SCORE = 50
SCORE_LOGIN_LAST_7_DAYS = 10
SCORE_KEY_FEATURE_USED_MULTIPLE_TIMES = 15
SCORE_NEGATIVE_BILLING_EVENT = -30
SCORE_GENERAL_INACTIVITY_PERIOD = -10
KEY_FEATURE_EVENT_NAMES = ["viewed_dashboard", "created_report"]
KEY_FEATURE_MIN_USAGE_COUNT = 3
NEGATIVE_BILLING_EVENT_NAMES = ["billing_failed", "subscription_cancelled", "payment_declined"]
INACTIVITY_DAYS_THRESHOLD = 14
STATUS_AT_RISK_THRESHOLD = 35
STATUS_HEALTHY_THRESHOLD_UPPER = 70

REACTIVATION_INACTIVITY_DAYS = 7
UPSELL_POWER_USER_SCORE_THRESHOLD = 80

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
    status = Column(String, default="sent") # "sent", "draft", "failed"
    sent_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user_response = Column(String, nullable=True)
    user = relationship("User", back_populates="interventions")

class CompanyBase(BaseModel):
    name: str
class CompanyCreate(CompanyBase): pass
class CompanyResponse(CompanyBase):
    id: int
    name: str
    created_at: datetime.datetime
    class Config: from_attributes = True

class ManualNudgePayload(BaseModel):
    message: str = Field(min_length=0, max_length=10000)
    ai_assist_topic: Optional[str] = Field(None, min_length=1, max_length=200)
    # --- NEW: Tweak 1 - Draft Only Flag ---
    draft_only: Optional[bool] = Field(False, description="If true and using AI assist, only generates draft, doesn't send.")

class UserBase(BaseModel):
    email: EmailStr
class UserCreate(UserBase):
    company_id: int
class UserResponse(UserBase):
    id: int
    email: EmailStr
    company_id: int
    created_at: datetime.datetime
    company: Optional[CompanyResponse] = None
    class Config: from_attributes = True

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
    properties: Optional[Dict[str, Any]] = None
    class Config: from_attributes = True

class HealthScoreBase(BaseModel):
    score: int
    status: str
class HealthScoreCreate(HealthScoreBase):
    user_id: int
class HealthScoreResponse(HealthScoreBase):
    id: int
    user_id: int
    calculated_at: datetime.datetime
    class Config: from_attributes = True

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
    status: str # Ensure status is part of response
    class Config: from_attributes = True

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

app = FastAPI(title="AI CSM MVP - Week 1-5 (Tweaked)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

async def send_email_notification(to_email: str, subject: str, body: str):
    print(f"--- SENDING EMAIL ---\nTo: {to_email}\nSubject: {subject}\nBody:\n{body}\n--- EMAIL SENT (to console) ---")
    return True

async def send_slack_alert(message: str, channel_id: Optional[str] = SLACK_ALERT_CHANNEL_ID):
    if not SLACK_BOT_TOKEN or not channel_id:
        print(f"Slack (to console): {message}")
        return False # Indicate not sent via actual Slack
    client = AsyncWebClient(token=SLACK_BOT_TOKEN)
    try:
        await client.chat_postMessage(channel=channel_id, text=message)
        return True
    except Exception as e:
        print(f"Error sending Slack alert: {e}")
        return False

async def generate_ai_message(prompt: str, system_message: str) -> Optional[str]:
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set. AI message generation skipped.")
        return "This is a default message because AI generation is currently unavailable. Please check your settings."
    try:
        response = await acompletion(
            model="groq/llama3-8b-8192", messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=700, api_key=GROQ_API_KEY,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error during AI message generation: {e}")
        return f"AI Error Fallback: We wanted to discuss: {prompt[:100]}..." # Generic fallback

def parse_subject_body(ai_content: str, default_subject: str) -> (str, str):
    subject, body = default_subject, ai_content
    try:
        if "Subject:" in ai_content and "\nBody:\n" in ai_content:
            parts = ai_content.split("\nBody:\n", 1)
            subject_line = parts[0].replace("Subject:", "").strip()
            if subject_line: subject = subject_line
            body = parts[1].strip()
        elif "Subject:" in ai_content:
            subject_line = ai_content.split("\n",1)[0].replace("Subject:", "").strip()
            if subject_line: subject = subject_line
            body = ai_content.split("\n",1)[1].strip() if "\n" in ai_content else ""
    except Exception: # Keep defaults if parsing fails
        pass
    return subject, body

async def calculate_user_health_score(user_id: int, db: AsyncSession) -> Optional[HealthScore]:
    current_score = DEFAULT_HEALTH_SCORE; now = datetime.datetime.utcnow()
    user_res = await db.execute(select(User).where(User.id == user_id))
    user = user_res.scalars().first()
    if not user: return None

    events_res = await db.execute(select(Event).where(Event.user_id == user_id, Event.timestamp >= now - datetime.timedelta(days=30)))
    recent_events = events_res.scalars().all()

    if any(e.event_name == "logged_in" and e.timestamp >= now - datetime.timedelta(days=7) for e in recent_events):
        current_score += SCORE_LOGIN_LAST_7_DAYS
    
    key_counts = {}
    for e in recent_events:
        if e.event_name in KEY_FEATURE_EVENT_NAMES: key_counts[e.event_name] = key_counts.get(e.event_name, 0) + 1
    for count in key_counts.values():
        if count >= KEY_FEATURE_MIN_USAGE_COUNT: current_score += SCORE_KEY_FEATURE_USED_MULTIPLE_TIMES

    if any(e.event_name in NEGATIVE_BILLING_EVENT_NAMES for e in recent_events):
        current_score += SCORE_NEGATIVE_BILLING_EVENT

    latest_event_ts_res = await db.execute(select(func.max(Event.timestamp)).where(Event.user_id == user_id))
    latest_event_ts = latest_event_ts_res.scalars().first()
    if not latest_event_ts or latest_event_ts < now - datetime.timedelta(days=INACTIVITY_DAYS_THRESHOLD):
        current_score += SCORE_GENERAL_INACTIVITY_PERIOD

    current_score = max(0, min(100, current_score))
    status = "Neutral"
    if current_score <= STATUS_AT_RISK_THRESHOLD: status = "At Risk"
    elif current_score > STATUS_HEALTHY_THRESHOLD_UPPER: status = "Power User"
    elif current_score > STATUS_AT_RISK_THRESHOLD: status = "Healthy"
    
    new_hs = HealthScore(user_id=user_id, score=current_score, status=status, calculated_at=now)
    db.add(new_hs); await db.commit(); await db.refresh(new_hs)
    return new_hs

async def process_user_for_triggers(user: User, db: AsyncSession):
    now = datetime.datetime.utcnow(); interventions_made = 0; user_name = user.email.split('@')[0]
    # --- TWEAK 2: Use actual company name in prompts ---
    # Ensure user.company is loaded. It is eager loaded by run_daily_triggers.
    # This 'product_name' will be used by the AI to refer to the client's SaaS product.
    # 'Our product/platform' is the AI CSM tool itself (EvoMind in this context).
    client_saas_product_name = user.company.name if user.company else "their current SaaS platform"
    our_csm_product_name = "EvoMind AI CSM" # Your product's name

    latest_hs_val = user.health_scores[0].score if user.health_scores else (await calculate_user_health_score(user.id, db) or HealthScore(score=DEFAULT_HEALTH_SCORE)).score
    recent_events_ctx = list(dict.fromkeys(e.event_name for e in user.events[:5])) if user.events else ["general usage"]

    # Trigger 1: Reactivation
    latest_event_ts_res = await db.execute(select(func.max(Event.timestamp)).where(Event.user_id == user.id))
    latest_event_ts = latest_event_ts_res.scalars().first()
    if not latest_event_ts or latest_event_ts < now - datetime.timedelta(days=REACTIVATION_INACTIVITY_DAYS):
        q_react = select(Intervention).where(Intervention.user_id == user.id, Intervention.intervention_type == "ai_reactivation_email", Intervention.sent_at >= now - datetime.timedelta(days=7))
        if not (await db.execute(q_react)).scalars().first():
            prompt_tpl = "reactivation_v2_company_specific"
            prompt = f"User Context:\n- Name: {user_name}\n- Their Company using our client's product: {client_saas_product_name}\n- Last interaction with {client_saas_product_name}: {(latest_event_ts or user.created_at).strftime('%Y-%m-%d')}\n- Example activities before inactivity: {', '.join(recent_events_ctx)}\n\nTask: You are a friendly Customer Success AI representing '{client_saas_product_name}'. Write a short, engaging reactivation email (Subject and Body) to {user_name}. Encourage them to return to '{client_saas_product_name}' or offer assistance. Start body with 'Hi {user_name},'.\nOutput Format:\nSubject: [Your Subject]\nBody:\n[Your Email Body]"
            ai_content = await generate_ai_message(prompt, f"You are an AI assistant writing reactivation emails on behalf of '{client_saas_product_name}'.")
            if ai_content:
                subject, body = parse_subject_body(ai_content, f"Checking In From {client_saas_product_name}, {user_name}!")
                if await send_email_notification(user.email, subject, body):
                    db.add(Intervention(user_id=user.id, intervention_type="ai_reactivation_email", channel="email", content_template=prompt_tpl, content_sent=f"Subject: {subject}\n\n{body}"))
                    interventions_made += 1; await send_slack_alert(f"AI sent reactivation for '{client_saas_product_name}' to {user.email}")

    # Trigger 2: Upsell
    if latest_hs_val >= UPSELL_POWER_USER_SCORE_THRESHOLD:
        q_upsell = select(Intervention).where(Intervention.user_id == user.id, Intervention.intervention_type == "ai_upsell_csm_alert", Intervention.sent_at >= now - datetime.timedelta(days=30))
        if not (await db.execute(q_upsell)).scalars().first():
            prompt_tpl = "upsell_points_v2_company_specific"
            prompt = f"User Context:\n- Name: {user_name}\n- Their Company using our client's product: {client_saas_product_name}\n- Health Score for {client_saas_product_name}: {latest_hs_val}\n- Example activities on {client_saas_product_name}: {', '.join(recent_events_ctx)}\n\nTask: You are an AI assistant for CSMs of '{our_csm_product_name}'. This CSM is helping the company '{client_saas_product_name}'. Generate 2-3 concise talking points for the CSM of '{our_csm_product_name}' to use when contacting their client '{client_saas_product_name}' about this power user ({user_name}). The goal is to help '{client_saas_product_name}' explore upsell opportunities for {user_name} (e.g., higher tier plan for '{client_saas_product_name}', new add-ons for '{client_saas_product_name}') or gather a testimonial for '{client_saas_product_name}'. Output as a bulleted list."
            talking_points = await generate_ai_message(prompt, f"You are an AI assistant helping CSMs of '{our_csm_product_name}' craft strategies for their client '{client_saas_product_name}'.")
            if talking_points:
                msg_csm = f"🚀 AI Upsell Alert for your client '{client_saas_product_name}': User {user.email} (ID: {user.id}, Score: {latest_hs_val})\nSuggested Talking Points for '{client_saas_product_name}' to consider:\n{talking_points}"
                if await send_slack_alert(msg_csm): # This alert goes to YOUR CSM team
                    db.add(Intervention(user_id=user.id, intervention_type="ai_upsell_csm_alert", channel="slack_internal_for_our_csm", content_template=prompt_tpl, content_sent=msg_csm))
                    interventions_made += 1

    # Trigger 3: Billing
    q_bill_event_res = await db.execute(select(Event).where(Event.user_id == user.id, Event.event_name.in_(NEGATIVE_BILLING_EVENT_NAMES), Event.timestamp >= now - datetime.timedelta(days=3)).order_by(Event.timestamp.desc()).limit(1))
    failed_event = q_bill_event_res.scalars().first()
    if failed_event:
        q_bill_int = select(Intervention).where(Intervention.user_id == user.id, Intervention.intervention_type == "ai_billing_followup_email", Intervention.sent_at >= now - datetime.timedelta(days=3))
        if not (await db.execute(q_bill_int)).scalars().first():
            prompt_tpl = "billing_failed_v2_company_specific"
            failed_event_details = f"Event: {failed_event.event_name}, Properties: {failed_event.properties or 'N/A'}"
            prompt = f"User Context:\n- Name: {user_name}\n- Their Company using our client's product: {client_saas_product_name}\n- Failed payment event for {client_saas_product_name}: {failed_event_details}\n\nTask: You are an AI assistant representing '{client_saas_product_name}'. Write a polite, clear email (Subject and Body) to {user_name} about a recent failed payment for their '{client_saas_product_name}' account. Urge them to update billing. Include '[Link to Update Billing for {client_saas_product_name}]'. Start body 'Hi {user_name},'.\nOutput Format:\nSubject: [Subject]\nBody:\n[Body]"
            ai_content = await generate_ai_message(prompt, f"You are an AI writing billing issue emails on behalf of '{client_saas_product_name}'.")
            if ai_content:
                subject, body = parse_subject_body(ai_content, f"Important: Payment Issue for Your {client_saas_product_name} Account")
                if await send_email_notification(user.email, subject, body):
                    db.add(Intervention(user_id=user.id, intervention_type="ai_billing_followup_email", channel="email", content_template=prompt_tpl, content_sent=f"Subject: {subject}\n\n{body}"))
                    interventions_made +=1; await send_slack_alert(f"AI sent billing follow-up for '{client_saas_product_name}' to {user.email}")
    
    if interventions_made > 0: await db.commit()
    return interventions_made

@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()
    async with AsyncSessionLocal() as s:
        async with s.begin():
            c = (await s.execute(select(Company).where(Company.name=="TestCompany Inc."))).scalars().first()
            if not c: c = Company(name="TestCompany Inc."); s.add(c); await s.flush()
            u = (await s.execute(select(User).options(selectinload(User.company)).where(User.email=="testuser@example.com"))).scalars().unique().first()
            if not u and c: u = User(email="testuser@example.com",company_id=c.id); s.add(u)
        await s.commit()

@app.post("/companies/", response_model=CompanyResponse, status_code=201)
async def create_company(c: CompanyCreate, db: AsyncSession=Depends(get_db)):
    dbc=Company(name=c.name); db.add(dbc); await db.commit(); await db.refresh(dbc); return dbc

@app.get("/companies/", response_model=List[CompanyResponse])
async def read_companies(s:int=0, l:int=100, db:AsyncSession=Depends(get_db)):
    return (await db.execute(select(Company).offset(s).limit(l))).scalars().all()

@app.get("/companies/{cid}", response_model=CompanyResponse)
async def read_company(cid:int, db:AsyncSession=Depends(get_db)):
    c=(await db.execute(select(Company).where(Company.id==cid))).scalars().first()
    if not c: raise HTTPException(404,"Company not found")
    return c

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(u:UserCreate, db:AsyncSession=Depends(get_db)):
    c=(await db.execute(select(Company).where(Company.id==u.company_id))).scalars().first()
    if not c: raise HTTPException(404,f"Company {u.company_id} not found")
    dbu=User(email=u.email,company_id=u.company_id); db.add(dbu); await db.commit(); await db.refresh(dbu)
    r=(await db.execute(select(User).options(selectinload(User.company)).where(User.id==dbu.id))).scalars().unique().first()
    return r

@app.get("/users/", response_model=List[UserResponse])
async def read_users(s:int=0, l:int=100, db:AsyncSession=Depends(get_db)):
    st=select(User).options(selectinload(User.company)).order_by(User.id).offset(s).limit(l)
    return (await db.execute(st)).scalars().unique().all()

@app.get("/users/{uid}", response_model=UserResponse)
async def read_user(uid:int, db:AsyncSession=Depends(get_db)):
    st=select(User).options(selectinload(User.company)).where(User.id==uid)
    u=(await db.execute(st)).scalars().unique().first()
    if not u: raise HTTPException(404, "User not found")
    return u

@app.post("/ingest_event/", response_model=EventResponse, status_code=201)
async def ingest_event(ed:EventCreate, db:AsyncSession=Depends(get_db)):
    u=(await db.execute(select(User).where(User.email==ed.user_email))).scalars().first()
    if not u: raise HTTPException(404,f"User {ed.user_email} not found")
    dbe=Event(user_id=u.id,event_name=ed.event_name,properties=ed.properties,timestamp=ed.timestamp or datetime.datetime.utcnow())
    db.add(dbe); await db.commit(); await db.refresh(dbe); return dbe

@app.get("/users/{uid}/events/", response_model=List[EventResponse])
async def read_user_events(uid:int,s:int=0,l:int=20,db:AsyncSession=Depends(get_db)):
    if not (await db.execute(select(User).where(User.id==uid))).scalars().first(): raise HTTPException(404,"User not found")
    return (await db.execute(select(Event).where(Event.user_id==uid).order_by(Event.timestamp.desc()).offset(s).limit(l))).scalars().all()

@app.get("/events/", response_model=List[EventResponse])
async def read_all_events(s:int=0,l:int=100,db:AsyncSession=Depends(get_db)):
    return (await db.execute(select(Event).order_by(Event.timestamp.desc()).offset(s).limit(l))).scalars().all()

@app.post("/users/{uid}/calculate_health_score", response_model=HealthScoreResponse)
async def trigger_user_health_score_calc(uid:int, db:AsyncSession=Depends(get_db)):
    if not (await db.execute(select(User).where(User.id==uid))).scalars().first(): raise HTTPException(404,"User not found")
    hs=await calculate_user_health_score(uid,db)
    if not hs: raise HTTPException(500,"Failed to calc health score")
    return hs

@app.post("/admin/run_daily_health_score_calculations", status_code=202)
async def run_all_health_scores(bg:BackgroundTasks, db:AsyncSession=Depends(get_db)):
    uids=(await db.execute(select(User.id))).scalars().all()
    async def _calc():
        async with AsyncSessionLocal() as tdb:
            p=0
            for i in uids:
                try: 
                    if await calculate_user_health_score(i,tdb): p+=1
                except Exception as e: print(f"Err scoring user {i}: {e}")
            print(f"Health score calc task done for {p}/{len(uids)} users.")
    bg.add_task(_calc)
    return {"message": f"Health score calc started for {len(uids)} users."}

@app.get("/users/{uid}/health_scores", response_model=List[HealthScoreResponse])
async def get_user_health_scores(uid:int,l:int=10,db:AsyncSession=Depends(get_db)):
    sc=(await db.execute(select(HealthScore).where(HealthScore.user_id==uid).order_by(HealthScore.calculated_at.desc()).limit(l))).scalars().all()
    if not sc: raise HTTPException(404,"No health scores for user.")
    return sc

@app.get("/users/{uid}/latest_health_score", response_model=Optional[HealthScoreResponse])
async def get_user_latest_health_score(uid:int,db:AsyncSession=Depends(get_db)):
    return (await db.execute(select(HealthScore).where(HealthScore.user_id==uid).order_by(HealthScore.calculated_at.desc()).limit(1))).scalars().first()

@app.post("/admin/run_daily_trigger_processing", status_code=202)
async def run_daily_triggers(bg:BackgroundTasks, db:AsyncSession=Depends(get_db)):
    st=select(User).options(selectinload(User.company),selectinload(User.health_scores),selectinload(User.events.and_(Event.timestamp>=datetime.datetime.utcnow()-datetime.timedelta(days=30))))
    us=(await db.execute(st)).scalars().unique().all()
    async def _proc():
        async with AsyncSessionLocal() as tdb:
            ti=0;pc=0
            for uo in us:
                try:
                    ifr=await process_user_for_triggers(uo,tdb); ti+=ifr; pc+=1
                except Exception as e: print(f"Err proc triggers for user {uo.id} ({uo.email}): {e}")
            print(f"Daily trigger proc task done. {ti} interventions for {pc}/{len(us)} users.")
    bg.add_task(_proc)
    return {"message": f"Daily trigger proc started for {len(us)} users."}

# --- MODIFIED: Manual Nudge Endpoint (Week 5 - Tweaks 1 & 2) ---
@app.post("/users/{user_id}/send_manual_nudge", status_code=200) # 200 if draft, 202 if queued
async def send_manual_nudge_to_user(
    user_id: int,
    payload: ManualNudgePayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(User).options(
        selectinload(User.company),
        selectinload(User.events.and_(Event.timestamp >= datetime.datetime.utcnow() - datetime.timedelta(days=30)))
    ).where(User.id == user_id)
    user_result = await db.execute(stmt)
    user = user_result.scalars().unique().first()

    if not user: raise HTTPException(status_code=404, detail="User not found")

    # --- TWEAK 2: Use actual company name in prompts ---
    client_saas_product_name = user.company.name if user.company else "their current platform"
    our_csm_product_name = "EvoMind AI CSM" # Your product's name

    final_message_body = payload.message
    prompt_template_name = "manual_direct_v2"
    ai_generated_draft_for_frontend = None

    if not payload.message.strip() and payload.ai_assist_topic:
        print(f"🤖 AI drafting manual nudge for topic: '{payload.ai_assist_topic}' for user {user.email} of {client_saas_product_name}")
        user_name = user.email.split('@')[0]
        recent_event_names = list(dict.fromkeys(e.event_name for e in user.events[:5])) if user.events else ["their general usage of " + client_saas_product_name]

        prompt = f"User Context:\n- User Name: {user_name}\n- User Email: {user.email}\n- User's Company (using our client's product): {client_saas_product_name}\n- Example recent activities on {client_saas_product_name}: {', '.join(recent_event_names)}\n\nTask: You are an AI assistant for a CSM of '{our_csm_product_name}'. This CSM needs to send an email to {user_name} who uses '{client_saas_product_name}'. Draft a concise, friendly, and helpful email body (NO subject line) based on the CSM's goal/topic: '{payload.ai_assist_topic}'. Start message with 'Hi {user_name},'. Focus on being helpful and relevant to the topic concerning their use of '{client_saas_product_name}'."
        
        prompt_template_name = f"manual_ai_assist_{payload.ai_assist_topic.replace(' ', '_').lower()[:25]}"
        generated_body = await generate_ai_message(prompt, f"You are an AI assistant helping a CSM of '{our_csm_product_name}' draft outreach to users of '{client_saas_product_name}'.")
        
        if generated_body and generated_body.strip() != "":
            final_message_body = generated_body
        else: 
            final_message_body = f"Hi {user_name},\n\nWe wanted to touch base regarding: {payload.ai_assist_topic} for your use of {client_saas_product_name}.\nPlease let us know if you have any questions.\n\nBest regards,\nThe {our_csm_product_name} Team (on behalf of {client_saas_product_name})"
            print(f"⚠️ AI generation failed for manual nudge (topic: {payload.ai_assist_topic}), using fallback.")
        
        ai_generated_draft_for_frontend = final_message_body # This is the draft
        
        # --- TWEAK 1: Handle draft_only ---
        if payload.draft_only:
            print(f"🤖 AI draft generated for manual nudge (user: {user.email}, topic: {payload.ai_assist_topic}). Not sending.")
            return {"message": "AI draft generated successfully.", "ai_generated_draft": ai_generated_draft_for_frontend, "is_draft": True}

    elif not payload.message.strip() and not payload.ai_assist_topic:
        raise HTTPException(status_code=400, detail="Either a message or an AI assist topic must be provided.")
    
    # If it's not draft_only or if a direct message was provided, proceed to send.
    subject = f"A message from {client_saas_product_name} (via {our_csm_product_name})"

    async def _send_and_record_nudge_task_final():
        async with AsyncSessionLocal() as task_db:
            # Re-fetch user in new session for safety, though user.email is likely fine.
            # user_in_task = (await task_db.execute(select(User).where(User.id == user_id))).scalars().first()
            # if not user_in_task: return # Should not happen
            
            email_sent = await send_email_notification(to_email=user.email, subject=subject, body=final_message_body)
            intervention_status = "sent" if email_sent else "failed"
            
            intervention_type_val = "manual_nudge_email_ai_assisted" if (not payload.message.strip() and payload.ai_assist_topic) else "manual_nudge_email_direct"
            
            intervention = Intervention(
                user_id=user.id, intervention_type=intervention_type_val, channel="email",
                content_template=prompt_template_name, 
                content_sent=f"Subject: {subject}\n\n{final_message_body}", 
                status=intervention_status
            )
            task_db.add(intervention)
            await task_db.commit()
            
            log_msg_detail = f"AI assisted ({payload.ai_assist_topic})" if (not payload.message.strip() and payload.ai_assist_topic) else "direct message"
            print(f"Manual nudge ({log_msg_detail}) {intervention_status} for {user.email} of {client_saas_product_name}.")
            if email_sent:
                 await send_slack_alert(f"Manual nudge sent by '{our_csm_product_name}' to user: {user.email} (of {client_saas_product_name}). Type: {log_msg_detail}. Subject: {subject}")

    background_tasks.add_task(_send_and_record_nudge_task_final)
    
    return_status_code = 200 if (payload.draft_only and ai_generated_draft_for_frontend) else 202 # Should be caught by earlier return
    # If we reach here and it was a draft_only AI assist, it means it's an error or direct message.
    # If it's a direct message, draft_only is ignored.
    if payload.draft_only and (not payload.message.strip() and payload.ai_assist_topic): # This case should have returned above
         return {"message": "Error: Draft requested but sending logic proceeded.", "ai_generated_draft": ai_generated_draft_for_frontend, "is_draft": True}


    return {"message": f"Manual nudge to {user.email} has been queued for sending.", "ai_generated_draft": None, "is_draft": False}


@app.get("/users/{uid}/interventions", response_model=List[InterventionResponse])
async def get_user_interventions(uid:int,l:int=10,db:AsyncSession=Depends(get_db)):
    r=(await db.execute(select(Intervention).where(Intervention.user_id==uid).order_by(Intervention.sent_at.desc()).limit(l))).scalars().all()
    if not r: raise HTTPException(404,"No interventions for user.")
    return r

@app.get("/interventions", response_model=List[InterventionResponse])
async def get_all_interventions(s:int=0,l:int=100,db:AsyncSession=Depends(get_db)):
    return (await db.execute(select(Intervention).order_by(Intervention.sent_at.desc()).offset(s).limit(l))).scalars().all()

if __name__ == "__main__":
    import uvicorn
    print(f"--- AI CSM Service Starting (Tweaked for Week 5) ---")
    print(f"GROQ_API_KEY: {'SET' if GROQ_API_KEY else 'NOT SET - AI features might use fallback text.'}")
    print(f"SLACK_BOT_TOKEN: {'SET' if SLACK_BOT_TOKEN else 'NOT SET - Slack alerts will print to console.'}")
    print(f"SLACK_ALERT_CHANNEL_ID: {SLACK_ALERT_CHANNEL_ID if SLACK_ALERT_CHANNEL_ID else 'NOT SET - Slack alerts will print to console.'}")
    print(f"-----------------------------")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)