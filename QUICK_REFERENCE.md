# 🎯 Quick Reference Card - Compliance Chatbot

> **Save this for quick access** - Essential commands for running the app

---

## 🚀 Start Application (3 Commands)

### Terminal 1: Database
```bash
cd /home/mubeen/compliance-chatbot
docker-compose up -d
```

### Terminal 2: Backend
```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Terminal 3: Frontend
```bash
cd /home/mubeen/compliance-chatbot/frontend
npm run dev
```

**Access**: http://localhost:5173

---

## 🛑 Stop Application

```bash
# Stop frontend & backend: Ctrl+C in their terminals
# Then deactivate backend venv:
deactivate

# Stop database:
cd /home/mubeen/compliance-chatbot
docker-compose down
```

---

## 📍 Key Locations

| What | Path |
|------|------|
| Project root | `/home/mubeen/compliance-chatbot` |
| Virtual env | `backend/.venv/` (ONLY USE THIS ONE) |
| Python | `backend/.venv/bin/python` |
| Activate venv | `source backend/.venv/bin/activate` |
| Backend code | `backend/` |
| Frontend code | `frontend/` |
| Environment vars | `.env` (root) |

---

## ⚙️ Common Commands

### Database
```bash
# Check status
docker ps | grep compliance_db

# View logs
docker logs compliance_db

# Connect to DB
docker exec -it compliance_db psql -U user -d compliance_db
```

### Backend
```bash
# Run tests
cd backend && source .venv/bin/activate && pytest

# Run specific test
python test_synthesis_node.py

# Database migration
alembic upgrade head

# Ingest documents
python scripts/ingest.py
```

### Frontend
```bash
# Install package
npm install package-name

# Build for production
npm run build
```

---

## 🐛 Emergency Fixes

### Port Already in Use
```bash
# Kill backend (port 8000)
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Kill frontend (port 5173)
lsof -i :5173 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Restart database
docker-compose down && docker-compose up -d
```

### Module Not Found
```bash
cd /home/mubeen/compliance-chatbot/backend
source .venv/bin/activate
pip install -r requirements.txt
```

### Database Connection Failed
```bash
docker-compose down
docker-compose up -d
# Wait 5 seconds, then restart backend
```

---

## ✅ Pre-flight Checklist

Before starting work:
- [ ] `.env` file exists with real API keys
- [ ] Docker is running
- [ ] Ports 8000, 5173, 5434 are free

---

## 📚 Full Documentation

For complete details, see: **[RUN_APP_GUIDE.md](./RUN_APP_GUIDE.md)**

---

**Remember**: 
- Always activate venv: `source backend/.venv/bin/activate`
- Only use `backend/.venv/` (never create venv in project root)
- Start database → backend → frontend (in that order)
