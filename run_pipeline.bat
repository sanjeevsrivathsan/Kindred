@echo off
call .venv\Scripts\activate
if not exist data\ml-25m\ratings.csv python scripts\download_data.py
python -m kindred.pipeline %*
pause
