@echo off
echo ==========================================
echo Stockout-Aware Retail Analytics Dashboard
echo ==========================================
echo.
cd /d "%~dp0.."
call .venv\Scripts\activate
streamlit run dashboard\dashboard.py
pause
