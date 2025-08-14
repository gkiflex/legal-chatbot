#!/bin/bash

echo "Starting LegalEaseBot..."
echo

# Create virtual environment if it doesn't exist
if [ ! -d "legalease_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv legalease_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source legalease_env/bin/activate

# Install requirements if needed
if [ ! -f "legalease_env/.installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch legalease_env/.installed
fi

echo
echo "Starting the application..."
python app.py