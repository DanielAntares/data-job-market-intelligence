@echo off
rem ============================================================
rem  Refresh the data in one click:
rem  collect postings -> extract skills -> regenerate charts.
rem ============================================================
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo    Refreshing data (this can take a minute)...
echo  ============================================================
echo.

echo  [1/3] Collecting postings...
python -m src.collect || goto :err

echo  [2/3] Extracting skills...
python -m src.extract_skills || goto :err

echo  [3/3] Rendering charts...
python -m src.figures || goto :err

rem keep the showcase (docs/) charts in sync with the freshly rendered ones
copy /Y "reports\figures\*.png" "docs\figures\" >nul 2>nul

echo.
echo  Done. Launch run.bat to see the updated dashboard.
echo  (To publish the refresh: git add -A ^&^& git commit -m "refresh data" ^&^& git push)
pause
exit /b 0

:err
echo.
echo  Something went wrong. Make sure dependencies are installed:
echo      pip install -r requirements.txt
pause
exit /b 1
