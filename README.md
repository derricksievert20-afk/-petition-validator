# Petition Intake Checker V4

Conservative Streamlit prototype for petition intake review.

This version is intentionally less aggressive:
- Flags a required field only when the box appears visually blank
- Does not flag low OCR confidence by default
- Does not flag date format issues by default
- Checks duplicate names/VUIDs only when OCR text confidence is high enough

Deploy with Streamlit Cloud using `app.py`.
