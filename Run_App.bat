@echo off
set REPO_NAME= QA_System
set REPO_URL=https://github.com/SSivaTejaReddy/QA_System.git

echo Checking repository...
IF EXIST "%REPO_NAME%" (
    echo Repository exists. Updating...
    cd %REPO_NAME%
    git pull
) ELSE (
    echo Cloning repository...
    git clone %REPO_URL%
    cd %REPO_NAME%
)

echo Checking virtual environment...
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Installing Playwright browsers...
playwright install

echo Launching the Streamlit app...
streamlit run app/streamlit_main.py

pause
