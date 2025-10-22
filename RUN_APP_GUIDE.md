# 🚀 Compliance Chatbot - Complete Run Guide

> **Last Updated**: October 21, 2025  
> **Purpose**: Step-by-step guide to run the app correctly every time without conflicts

---

## 📋 Table of Contents
1. [Quick Start (TL;DR)](#quick-start-tldr)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Running the Application](#running-the-application)
6. [Common Tasks](#common-tasks)
7. [Troubleshooting](#troubleshooting)
8. [Clean Shutdown](#clean-shutdown)

---

## 🎯 Quick Start (TL;DR)

```bash
# 1. Start PostgreSQL database
cd /home/mubeen/compliance-chatbot
docker-compose up -d

# 2. Start Backend (in new terminal)
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# 3. Start Frontend (in new terminal)
cd /home/mubeen/compliance-chatbot/frontend
npm run dev

# Access:
# Frontend: http://localhost:5173
# Backend: http://localhost:8001
# API Docs: http://localhost:8001/docs
```

---

## 📁 Project Structure

```
/home/mubeen/compliance-chatbot/
├── backend/                    # Python FastAPI backend
│   ├── .venv/                 # ✅ Python virtual environment (USED)
│   ├── main.py                # FastAPI application entry point
│   ├── requirements.txt       # Python dependencies
│   ├── activate.sh            # Helper script to activate venv
│   ├── core/                  # Core business logic
│   ├── api/                   # API routes
│   ├── models/                # Database models
│   └── scripts/               # Utility scripts (ingest, etc.)
├── frontend/                   # React + Vite frontend
│   ├── node_modules/          # NPM packages
│   ├── src/                   # React source code
│   └── package.json           # NPM dependencies
├── content_store/             # Regulatory content files
├── .env                       # Environment variables (DO NOT COMMIT)
├── .env.example               # Template for environment variables
└── docker-compose.yml         # PostgreSQL database configuration
```

**⚠️ IMPORTANT**: 
- **ONLY** use `backend/.venv/` for Python virtual environment
- The root `/venv` directory has been removed (was unused/conflicting)
- Never create venv in the project root directory

---

## ✅ Prerequisites

### Required Software

| Software | Version | Check Command | Install |
|----------|---------|---------------|---------|
| Python | 3.12+ | `python3 --version` | [python.org](https://python.org) |
| Node.js | 18+ | `node --version` | [nodejs.org](https://nodejs.org) |
| npm | 9+ | `npm --version` | Comes with Node.js |
| Docker | Latest | `docker --version` | [docker.com](https://docker.com) |
| Docker Compose | 2.0+ | `docker-compose --version` | Comes with Docker Desktop |

### Verify Prerequisites

```bash
# Check all at once
python3 --version && node --version && npm --version && docker --version && docker-compose --version
```

---

## 🔧 Environment Setup

### Step 1: Clone and Navigate

```bash
cd /home/mubeen/compliance-chatbot
```

### Step 2: Configure Environment Variables

```bash
# Copy the template (if .env doesn't exist)
cp .env.example .env

# Edit with your actual credentials
nano .env  # or use your preferred editor
```

**Required variables in `.env`:**
```bash
# Database Configuration
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=compliance_db
DATABASE_URL=postgresql://user:password@localhost:5434/compliance_db

# API Keys (REQUIRED - replace with your actual keys)
OPENAI_API_KEY=sk-proj-your-actual-openai-key-here
PINECONE_API_KEY=your-actual-pinecone-key-here
PINECONE_ENVIRONMENT=your-pinecone-environment
```

⚠️ **NEVER commit `.env` file to git** (it's already in `.gitignore`)

### Step 3: Setup Backend Virtual Environment

```bash
cd backend

# If .venv doesn't exist, create it:
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# You should see (.venv) in your prompt:
# (.venv) mubeen@MubeenAccess:~/compliance-chatbot/backend$

# Install/update dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Verification:**
```bash
which python  # Should show: /home/mubeen/compliance-chatbot/backend/.venv/bin/python
python --version  # Should show: Python 3.12.x
pip list | grep fastapi  # Should show fastapi and version
```

### Step 4: Setup Frontend Dependencies

```bash
cd /home/mubeen/compliance-chatbot/frontend

# Install dependencies (if node_modules doesn't exist)
npm install

# Verify
npm list --depth=0
```

---

## 🏃 Running the Application

### Complete Startup Sequence

**Terminal 1: Start Database**
```bash
cd /home/mubeen/compliance-chatbot
docker-compose up -d

# Verify it's running
docker ps | grep compliance_db
```

**Terminal 2: Start Backend**
```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.
```

**Terminal 3: Start Frontend**
```bash
cd /home/mubeen/compliance-chatbot/frontend
npm run dev

# You should see:
# VITE v5.x.x  ready in xxx ms
# ➜  Local:   http://localhost:5173/
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Frontend UI | http://localhost:5173 | Main chat interface |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Interactive Swagger docs |
| Database | localhost:5434 | PostgreSQL (Docker) |

---

## 🛠️ Common Tasks

### Run Database Migrations

```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
alembic upgrade head
```

### Ingest Content (Load regulatory documents)

```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
python scripts/ingest.py
```

### Run Tests

```bash
# All tests
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
pytest

# Specific test file
python test_synthesis_node.py

# With coverage
pytest --cov=core --cov=api tests/
```

### Check Backend Logs

```bash
# If running with uvicorn, logs appear in terminal
# To save logs to file:
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee backend.log
```

### Check Database

```bash
# Connect to database
docker exec -it compliance_db psql -U user -d compliance_db

# Inside psql:
\dt              # List tables
\d+ table_name   # Describe table
SELECT * FROM your_table LIMIT 5;
\q               # Quit
```

### Install New Python Package

```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
pip install package-name
pip freeze > requirements.txt  # Update requirements
git add requirements.txt
git commit -m "Add package-name dependency"
```

### Install New NPM Package

```bash
cd /home/mubeen/compliance-chatbot/frontend
npm install package-name
# package.json and package-lock.json will be updated automatically
git add package.json package-lock.json
git commit -m "Add package-name dependency"
```

---

## 🐛 Troubleshooting

### Problem: Import errors / Module not found

**Solution:**
```bash
# 1. Ensure you're in the virtual environment
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
which python  # Verify

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. In VS Code, select correct interpreter
# Ctrl+Shift+P → "Python: Select Interpreter"
# Choose: /home/mubeen/compliance-chatbot/backend/.venv/bin/python
```

### Problem: Port already in use

**Solution:**
```bash
# Find process using port 8000 (backend)
lsof -i :8000
kill -9 <PID>

# Find process using port 5173 (frontend)
lsof -i :5173
kill -9 <PID>

# Find process using port 5434 (database)
docker-compose down
docker-compose up -d
```

### Problem: Database connection refused

**Solution:**
```bash
# 1. Check if database is running
docker ps | grep compliance_db

# 2. If not running, start it
docker-compose up -d

# 3. Check logs
docker logs compliance_db

# 4. Verify connection settings in .env
cat .env | grep DATABASE_URL
# Should be: postgresql://user:password@localhost:5434/compliance_db
```

### Problem: "No module named 'backend'"

**Solution:**
```bash
# You're probably running from the wrong directory
# Use absolute imports or run from project root

# Option 1: Set PYTHONPATH
export PYTHONPATH=/home/mubeen/compliance-chatbot:$PYTHONPATH

# Option 2: Run from correct directory
cd /home/mubeen/compliance-chatbot
python -m backend.scripts.ingest
```

### Problem: Frontend not connecting to backend

**Solution:**
```bash
# 1. Check backend is running
curl http://localhost:8000/

# 2. Check frontend proxy configuration
# Look in frontend/vite.config.js for proxy settings

# 3. Check CORS settings in backend/main.py
# Ensure frontend origin is allowed
```

### Problem: Virtual environment activation fails

**Solution:**
```bash
# Recreate the virtual environment
cd /home/mubeen/compliance-chatbot/backend
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: Docker compose fails

**Solution:**
```bash
# Stop all containers
docker-compose down

# Remove volumes (CAUTION: This deletes data)
docker-compose down -v

# Restart
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## 🛑 Clean Shutdown

Always shut down gracefully to avoid port conflicts and data corruption:

```bash
# 1. Stop Frontend (Ctrl+C in terminal)
# Press Ctrl+C in the terminal running npm

# 2. Stop Backend (Ctrl+C in terminal)
# Press Ctrl+C in the terminal running uvicorn

# 3. Deactivate virtual environment
deactivate

# 4. Stop Database
cd /home/mubeen/compliance-chatbot
docker-compose down
```

**Complete cleanup (if needed):**
```bash
cd /home/mubeen/compliance-chatbot

# Stop and remove database (keeps data)
docker-compose down

# Stop and remove database + volumes (DELETES ALL DATA)
docker-compose down -v

# Kill any lingering processes
pkill -f "uvicorn main:app"
pkill -f "npm run dev"
```

---

## 📝 Development Workflow Checklist

Use this checklist every time you start working:

- [ ] Navigate to project: `cd /home/mubeen/compliance-chatbot`
- [ ] Start database: `docker-compose up -d`
- [ ] Verify database: `docker ps | grep compliance_db`
- [ ] Activate backend venv: `cd backend && source .venv/bin/activate`
- [ ] Start backend: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- [ ] Start frontend: `cd frontend && npm run dev`
- [ ] Open browser: http://localhost:5173
- [ ] Verify API docs: http://localhost:8000/docs

When finished:
- [ ] Stop frontend (Ctrl+C)
- [ ] Stop backend (Ctrl+C)
- [ ] Deactivate venv: `deactivate`
- [ ] Stop database: `docker-compose down`

---

## 🔑 Key Reminders

1. **Always activate the virtual environment** before running Python commands
2. **Use `backend/.venv/`** - never create venv in project root
3. **Check .env file** is properly configured with real API keys
4. **Start database first** before backend
5. **Use separate terminals** for each service (db, backend, frontend)
6. **Gracefully shutdown** with Ctrl+C, not by closing terminals
7. **Check ports** (8000, 5173, 5434) are free before starting

---

## 📚 Additional Resources

- [Backend VENV Guide](backend/VENV_GUIDE.md) - Detailed Python environment info
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React + Vite Documentation](https://vitejs.dev/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

**Questions or issues?** Check the troubleshooting section or review logs for specific error messages.
