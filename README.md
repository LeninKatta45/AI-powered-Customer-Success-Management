# 🚀 Agentic AI CSM

**Next-gen Customer Success Platform powered by Agentic AI + Memory + Autonomous Actions**

EvoMind AI CSM is an AI-first Customer Success Manager (CSM) system that does what a human CSM would — monitor users, calculate health scores, prevent churn, upsell, and nudge customers — but with autonomous AI agents.

Unlike rule-based dashboards, EvoMind agents can:

- **Reason** over user context (usage, events, billing, interventions)
- **Decide** when to send an email, Slack alert, or recalc health score  
- **Act** autonomously with tools (Email, Slack, HealthScore, Context)
- **Remember** past actions using a FAISS vector memory
- **Collaborate** with human CSMs via AI-drafted nudges

## ✨ Key Features

### 🤖 Agentic AI Brain
Built with LangChain OpenAI Functions Agent, reasoning over user history and choosing actions.

### 🧠 Memory Layer
Uses FAISS vector store + OpenAI embeddings to store & recall interactions per user.

### 📊 Health Scoring Engine
Dynamic health score calculation (0–100) based on logins, feature usage, billing events, inactivity.

### 📬 Automated Interventions
- AI-generated personalized emails via GROQ Llama-3
- Slack alerts to CSM team for high-risk users
- Supports both manual nudges and AI-drafted nudges

### 🛠️ FastAPI + SQLAlchemy Async
Async architecture, modern Pydantic v2 models, SQLite (pluggable DB).

### ⚡ Background Agentic Runs
`/admin/run_agentic_processing` → runs the agent loop for all users in background tasks.

## 🏗️ System Architecture

```
┌────────────────────────────┐
│      Frontend (TSX)        │
│  Dashboard / Users / Admin │
└─────────────┬──────────────┘
              │ REST API
┌─────────────▼──────────────────────────────────────┐
│              FastAPI Backend (main.py)             │
│                                                    │
│  ┌─────────────┐           ┌───────────────┐       │
│  │  Database   │           │ Vector Memory │       │
│  │  (SQLite)   │           │ (FAISS+OpenAI)│       │
│  └──────┬──────┘           └──────┬────────┘       │
│         │                         │                │
│  ┌─────▼─────┐           ┌───────▼────────┐        │
│  │  Events   │           │ Health Scoring │        │
│  │ Intervent.│           │     Engine     │        │
│  └─────┬─────┘           └───────┬────────┘        │
│        │                         │                 │
│  ┌─────▼─────────────────▼────────┐                │
│  │    🤖Agentic AI Executor      │                │
│  │  (LangChain + OpenAI Functions)│                │
│  └────────────────────────────────┘                │
│          │ Tools: Email / Slack / Context          │
└──────────┴─────────────────────────────────────────┘
```

## ⚡ Quickstart

### 1️⃣ Install
```bash
git clone https://github.com/LeninKatta45/AI-powered-Customer-Success-Management.git
cd agentic-csm
pip install -r requirements.txt
```

### 2️⃣ Environment
Create `.env`:
```env
OPENAI_API_KEY=sk-xxxx
GROQ_API_KEY=groq-xxxx # optional (better email drafting)
SLACK_BOT_TOKEN=xoxb-xxxx # optional
SLACK_ALERT_CHANNEL_ID=C123456789 # optional
```

### 3️⃣ Run
```bash
uvicorn main:app --reload
```
**Backend runs at** → http://localhost:8000

### 4️⃣ Explore
**Swagger Docs** → http://localhost:8000/docs

Example agent run:
```bash
curl -X POST http://localhost:8000/admin/run_agentic_processing
```

## 📡 API Highlights

- `POST /companies/` → Add company
- `POST /users/` → Add user  
- `POST /ingest_event/` → Track event
- `POST /users/{id}/calculate_health_score` → Calculate health score
- `POST /admin/run_agentic_processing` → Run agent loop for all users
- `POST /users/{id}/send_manual_nudge` → Human/AI-assisted manual outreach

## 🤖 Agent Tools

- `SendEmail(user_id, subject, body)`
- `SendSlackAlert(user_id, message)`
- `CalculateHealthScore(user_id)`
- `GetUserContext(user_id)`

## 🔮 Roadmap

- [ ] Webhook integrations (Hubspot, Intercom, Stripe)
- [ ] Multi-tenant mode (per CSM org)
- [ ] RAG-based long-term memory (Pinecone/Weaviate)
- [ ] UI for AI-draft review + approve workflows
- [ ] Reinforcement loop (feedback from interventions → score updates)

## 🧑‍💻 Tech Stack

- **Backend**: FastAPI + SQLAlchemy Async + SQLite
- **AI Reasoning**: LangChain OpenAIFunctionsAgent
- **Memory**: FAISS + OpenAI embeddings
- **Drafting**: GROQ (Llama-3 via LiteLLM)
- **Messaging**: Slack SDK, Email console (extendable to SMTP/SendGrid)
- **Frontend**: React + TypeScript (Company/User/Dashboard Pages)

## 🏆 Why This Matters

**Traditional CSM platforms only observe.**

**EvoMind acts.**

With Agentic AI + memory, it:
- Proactively prevents churn
- Autonomously nudges users  
- Scales CSM team efficiency 10×

**This isn't just a dashboard.**

**It's a CSM agent that never sleeps.**

## 📜 License

MIT © 2025 — built with ❤️ for the future of autonomous SaaS.