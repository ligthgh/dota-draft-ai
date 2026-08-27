@echo off
call .venv\Scripts\activate
python -m streamlit run src/recommend_app_v8.py
pause
