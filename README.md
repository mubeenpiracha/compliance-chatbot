# AI Compliance Officer Chatbot

This is the repository for the AI Compliance Officer Chatbot, designed to provide preliminary due diligence assistance for DIFC & ADGM regulatory frameworks.

## 🚀 Quick Start

**For complete setup and run instructions, see: [RUN_APP_GUIDE.md](./RUN_APP_GUIDE.md)**

### TL;DR

```bash
# 1. Start database
docker-compose up -d

# 2. Start backend (new terminal)
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Start frontend (new terminal)
cd frontend
npm run dev
```

Access the app at http://localhost:5173

## 📚 Documentation

- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - ⚡ Quick command reference card
- **[RUN_APP_GUIDE.md](./RUN_APP_GUIDE.md)** - 📖 Complete guide to run the app (START HERE)
- **[VENV_CLEANUP_SUMMARY.md](./VENV_CLEANUP_SUMMARY.md)** - 🧹 Virtual environment cleanup notes
- **[backend/VENV_GUIDE.md](./backend/VENV_GUIDE.md)** - 🐍 Python virtual environment details
- **[ENHANCED_SYSTEM_README.md](./ENHANCED_SYSTEM_README.md)** - 🏗️ System architecture
- **[SECURE_SETUP.md](./SECURE_SETUP.md)** - 🔒 Security configuration

## Project Structure

```
compliance-chatbot/
├── backend/          # Python FastAPI backend (use backend/.venv/)
├── frontend/         # React + Vite frontend
├── content_store/    # Regulatory documents
├── .env             # Environment variables (create from .env.example)
└── docker-compose.yml # PostgreSQL database
```

## Requirements

- Python 3.12+
- Node.js 18+
- Docker & Docker Compose
- OpenAI API key
- Pinecone API key

See [RUN_APP_GUIDE.md](./RUN_APP_GUIDE.md) for detailed setup instructions.