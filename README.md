# Petition Validator V3

This version focuses on a clean reviewer report instead of dumping raw OCR confidence.

It flags:
- Page/row-specific missing required fields
- Duplicate printed names with page/row locations
- Duplicate VUIDs with page/row locations

Low OCR/illegible handwriting flags are off by default to reduce false positives.

Deploy on Streamlit with:
- Branch: `main`
- Main file path: `app.py`
