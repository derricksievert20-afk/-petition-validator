# Petition AI Validator

This is a Streamlit web app that accepts completed petition PDFs/images, OCRs the signature rows, and flags:

- Missing required fields
- Duplicate printed names
- Duplicate voter IDs
- Low-confidence / likely illegible handwriting
- Date format issues

## Best use
This works best with flat, scanned pages. Phone pictures should be taken straight-on with good lighting.

## Deploy as a shareable website
1. Create a GitHub repo.
2. Upload all files in this folder.
3. Go to Streamlit Community Cloud.
4. Deploy `app.py`.

The included `packages.txt` installs Tesseract on Streamlit Cloud.

## Run locally
Install Tesseract OCR first, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important note
This is a prototype. It should flag entries for human review, not automatically reject signatures.
