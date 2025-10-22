# Backend Virtual Environment Guide

## Overview

The backend now uses a Python virtual environment (venv) located at `backend/.venv/` for dependency isolation and management.

## Quick Start

### Activate the Virtual Environment

**Option 1: Direct activation**
```bash
cd backend
source .venv/bin/activate
```

**Option 2: Using the activation script**
```bash
cd backend
source activate.sh
```

**Option 3: One-liner from project root**
```bash
source backend/.venv/bin/activate
```

### Verify Activation

When activated, your terminal prompt should show `(.venv)` prefix:
```bash
(.venv) user@machine:~/compliance-chatbot/backend$
```

Check Python location:
```bash
which python  # Should show: /home/mubeen/compliance-chatbot/backend/.venv/bin/python
```

### Deactivate

To exit the virtual environment:
```bash
deactivate
```

## Running Backend Services

### Development Server

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
cd backend
source .venv/bin/activate
pytest
```

### Run Specific Test
```bash
cd backend
source .venv/bin/activate
python test_synthesis_node.py
```

### Database Migrations (Alembic)

```bash
cd backend
source .venv/bin/activate
alembic upgrade head
```

### Ingestion Scripts

```bash
cd backend
source .venv/bin/activate
python scripts/ingest.py
```

## Managing Dependencies

### Install New Package

```bash
cd backend
source .venv/bin/activate
pip install package-name
pip freeze > requirements.txt  # Update requirements
```

### Update All Packages

```bash
cd backend
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### List Installed Packages

```bash
cd backend
source .venv/bin/activate
pip list
```

## Recreating the Virtual Environment

If you need to recreate the venv from scratch:

```bash
cd backend

# Remove old venv
rm -rf .venv

# Create new venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## IDE Configuration

### VS Code

The workspace should automatically detect the virtual environment. If not:

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose `/home/mubeen/compliance-chatbot/backend/.venv/bin/python`

### PyCharm

1. Go to **File → Settings → Project → Python Interpreter**
2. Click the gear icon → **Add**
3. Select **Existing environment**
4. Browse to `backend/.venv/bin/python`

## Current Environment Details

- **Python Version**: 3.12.3
- **Location**: `/home/mubeen/compliance-chatbot/backend/.venv/`
- **Type**: venv (Python's built-in virtual environment)

### Key Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| openai | 1.98.0 | OpenAI API client |
| pydantic | 2.11.7 | Data validation |
| fastapi | 0.116.1 | Web framework |
| langchain | 0.3.27 | LLM framework |
| langgraph | 0.6.3 | Agent state management |
| pinecone | 7.3.0 | Vector database |
| pytest | latest | Testing framework |

Full list: See `requirements.txt`

## Troubleshooting

### Import errors in VS Code

If you see "Import X could not be resolved" errors:

1. Make sure VS Code is using the correct Python interpreter (see IDE Configuration above)
2. Reload VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Verify the package is installed: `pip list | grep package-name`

### "No module named 'backend'" error

Make sure you're running Python from the correct directory:
```bash
# Run from project root, not backend folder
cd /home/mubeen/compliance-chatbot
python -m backend.scripts.ingest
```

Or add the parent directory to PYTHONPATH:
```bash
export PYTHONPATH=/home/mubeen/compliance-chatbot:$PYTHONPATH
```

### Permission denied

If you get permission errors:
```bash
chmod +x backend/activate.sh
```

## Environment Variables

Don't forget to set required environment variables before running:

```bash
export OPENAI_API_KEY="your-key-here"
export PINECONE_API_KEY="your-key-here"
export PINECONE_ENVIRONMENT="your-env-here"
```

Or use a `.env` file (already configured with python-dotenv).

## Testing the Setup

Run the synthesis node test to verify everything is working:

```bash
cd backend
source .venv/bin/activate
python test_synthesis_node.py
```

Expected output:
```
🎉 All tests passed! Synthesis node is ready.
```

## Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip documentation](https://pip.pypa.io/en/stable/)
- [requirements.txt format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
