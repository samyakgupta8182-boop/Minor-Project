@echo off
title Helix Bank - Churn Predictor Launcher
cd /d "%~dp0"
echo ===================================================
echo   Helix Bank Customer Churn Predictor Launcher
echo ===================================================
echo.

echo [INFO] Verifying Python virtual environment (venv)...
if not exist venv (
    echo [INFO] Creating virtual environment...
    py -m venv venv
    if errorlevel 1 (
        echo [INFO] Retrying with 'python' executable...
        python -m venv venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment. Please install Python!
            pause
            exit /b 1
        )
    )
)

echo [INFO] Verifying and installing required python libraries...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements!
    pause
    exit /b 1
)

if not exist churn_model.joblib (
    echo [INFO] Model artifacts not found. Training model pipeline...
    venv\Scripts\python.exe train_model.py
    if errorlevel 1 (
        echo [ERROR] Model training failed!
        pause
        exit /b 1
    )
)

echo [INFO] Launching Customer Churn Predictor web app...
echo [INFO] Opening in your default web browser...
echo.
venv\Scripts\streamlit.exe run app.py
if errorlevel 1 (
    echo [ERROR] Streamlit exited with an error.
    pause
)
