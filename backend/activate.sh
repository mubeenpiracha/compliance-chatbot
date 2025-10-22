#!/bin/bash
# Activation script for the backend virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Activating Python virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"

echo "✓ Virtual environment activated"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""
echo "To deactivate, run: deactivate"
