@echo off
rem ============================================================
rem  Data Job Market Intelligence - one-click dashboard launcher
rem  Double-click this file to start the dashboard.
rem ============================================================
setlocal
cd /d "%~dp0"

rem --- Make sure Streamlit is installed for this Python --------
python -c "import streamlit" 1>nul 2>nul
if errorlevel 1 (
    echo.
    echo  Streamlit is not installed for the "python" on your PATH.
    echo  Open a terminal in this folder and run:
    echo.
    echo      pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem --- Skip Streamlit's one-time "enter your email" prompt -----
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" 1>nul 2>nul
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    >  "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo  ============================================================
echo    Data Job Market Intelligence
echo    Starting the dashboard... your browser will open shortly.
echo.
echo    Keep this window OPEN while you use the dashboard.
echo    Close it (or press Ctrl+C) to stop.
echo  ============================================================
echo.

rem  Use "python -m streamlit" (not bare "streamlit") so it always uses the
rem  same Python that has Streamlit installed.
python -m streamlit run app.py

echo.
echo  Dashboard stopped.
pause
endlocal
