@echo off
echo Starting LegalEaseBot...
echo.

REM Activate virtual environment if it exists
if exist legalease_env\Scripts\activate.bat (
    echo Activating virtual environment...
    call legalease_env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv legalease_env
    call legalease_env\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Starting the application...
python app.py

pause