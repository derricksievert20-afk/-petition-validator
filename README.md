# Petition AI Validator V2

Template-aware Streamlit prototype for the Texas local political subdivision petition form.

## What changed in V2

- Looks for the signature table instead of OCRing the whole page.
- Ignores the title/instruction/affidavit sections.
- Splits the signature table into 10 signer rows.
- Uses ink density to detect blank fields, which reduces false missing-field flags.
- Uses OCR confidence only as a human-review signal.
- Checks duplicate printed names and duplicate VUIDs across all uploaded pages.
- Exports CSV and Excel reports.
- Shows a preview with the detected table/rows.

## Deploy on Streamlit Cloud

Upload these files to your GitHub repository:

- `app.py`
- `requirements.txt`
- `packages.txt`
- `README.md`

Then deploy with:

- Branch: `main`
- Main file path: `app.py`

## Important

This is still a prototype. Generic OCR will not read all handwriting correctly. A production version should integrate Google Vision, Azure Document Intelligence, or another handwriting OCR service.
