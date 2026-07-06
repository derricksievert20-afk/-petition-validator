import io
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image
from rapidfuzz import fuzz

st.set_page_config(page_title="Petition AI Validator", layout="wide")

DEFAULT_FIELDS = [
    "Signature",
    "Printed Name",
    "Residence Address",
    "City",
    "County",
    "Voter ID / DOB",
    "Date Signed",
]

@dataclass
class OCRCell:
    text: str
    conf: float


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_pdf(file_bytes: bytes, scale: float = 2.5) -> List[Image.Image]:
    pdf = pdfium.PdfDocument(file_bytes)
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale).to_pil()
        pages.append(bitmap.convert("RGB"))
    return pages


def load_uploaded_file(uploaded) -> List[Image.Image]:
    data = uploaded.read()
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        return render_pdf(data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return [img]


def preprocess_for_lines(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    return thr


def find_table_bounds(binary: np.ndarray) -> Tuple[int, int, int, int]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape[:2]
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if cw > 0.55 * w and ch > 0.15 * h and area > 0.1 * w * h:
            candidates.append((x, y, cw, ch))
    if candidates:
        x, y, cw, ch = max(candidates, key=lambda r: r[2] * r[3])
        return x, y, x + cw, y + ch
    return 0, int(h * 0.18), w, int(h * 0.95)


def detect_horizontal_lines(binary: np.ndarray) -> List[int]:
    h, w = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 12), 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = lines.sum(axis=1)
    ys = np.where(projection > projection.max() * 0.35)[0] if projection.max() > 0 else np.array([])
    grouped = []
    for y in ys:
        if not grouped or y - grouped[-1][-1] > 3:
            grouped.append([y])
        else:
            grouped[-1].append(y)
    centers = [int(np.mean(g)) for g in grouped]
    return centers


def detect_vertical_lines(binary: np.ndarray) -> List[int]:
    h, w = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 15)))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = lines.sum(axis=0)
    xs = np.where(projection > projection.max() * 0.35)[0] if projection.max() > 0 else np.array([])
    grouped = []
    for x in xs:
        if not grouped or x - grouped[-1][-1] > 3:
            grouped.append([x])
        else:
            grouped[-1].append(x)
    centers = [int(np.mean(g)) for g in grouped]
    return centers


def ocr_image_cell(img_bgr: np.ndarray) -> OCRCell:
    if img_bgr.size == 0:
        return OCRCell("", 0.0)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cfg = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DATAFRAME)
    if data is None or data.empty:
        return OCRCell("", 0.0)
    data = data.dropna(subset=["text"])
    texts = []
    confs = []
    for _, row in data.iterrows():
        text = str(row.get("text", "")).strip()
        try:
            conf = float(row.get("conf", -1))
        except Exception:
            conf = -1
        if text and conf >= 0:
            texts.append(text)
            confs.append(conf)
    return OCRCell(" ".join(texts).strip(), float(np.mean(confs)) if confs else 0.0)


def split_page_into_rows_and_cells(img: Image.Image, field_names: List[str], first_data_row: int = 1) -> Tuple[List[Dict], Image.Image]:
    bgr = pil_to_cv(img)
    binary = preprocess_for_lines(bgr)
    x1, y1, x2, y2 = find_table_bounds(binary)
    crop_bin = binary[y1:y2, x1:x2]
    crop_bgr = bgr[y1:y2, x1:x2]
    h, w = crop_bin.shape

    hlines = detect_horizontal_lines(crop_bin)
    vlines = detect_vertical_lines(crop_bin)

    # Fallbacks for forms where grid lines are faint.
    if len(hlines) < 5:
        row_count_guess = 11
        hlines = [int(i * h / row_count_guess) for i in range(row_count_guess + 1)]
    if len(vlines) < len(field_names) + 1:
        vlines = [int(i * w / len(field_names)) for i in range(len(field_names) + 1)]
    else:
        # Keep only left/right-ish table separators and normalize count if too many.
        vlines = sorted(vlines)
        if len(vlines) > len(field_names) + 1:
            idx = np.linspace(0, len(vlines) - 1, len(field_names) + 1).round().astype(int)
            vlines = [vlines[i] for i in idx]

    hlines = sorted(set([max(0, min(h - 1, y)) for y in hlines]))
    vlines = sorted(set([max(0, min(w - 1, x)) for x in vlines]))

    rows = []
    annotated = bgr.copy()
    page_row_num = 0
    for r in range(first_data_row, len(hlines) - 1):
        top, bottom = hlines[r], hlines[r + 1]
        if bottom - top < 18:
            continue
        page_row_num += 1
        row = {"page_row": page_row_num}
        confs = []
        for c, field in enumerate(field_names):
            if c >= len(vlines) - 1:
                break
            left, right = vlines[c], vlines[c + 1]
            pad = 5
            cell = crop_bgr[max(0, top+pad):max(0, bottom-pad), max(0, left+pad):max(0, right-pad)]
            ocr = ocr_image_cell(cell)
            row[field] = ocr.text
            row[field + " confidence"] = round(ocr.conf, 1)
            confs.append(ocr.conf)
        row["average_confidence"] = round(float(np.mean(confs)) if confs else 0, 1)
        rows.append(row)

        # draw row boxes on original image
        cv2.rectangle(annotated, (x1, y1 + top), (x2, y1 + bottom), (0, 255, 255), 2)
        cv2.putText(annotated, str(page_row_num), (x1 + 5, y1 + top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return rows, cv_to_pil(annotated)


def normalize_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z\s]", "", str(s).lower()).strip()
    return re.sub(r"\s+", " ", s)


def looks_like_date(s: str) -> bool:
    s = str(s).strip()
    return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", s))


def validate_rows(df: pd.DataFrame, required_fields: List[str], name_field: str, voter_field: str, date_field: str, low_conf: int) -> pd.DataFrame:
    issues = []
    norm_names = df[name_field].fillna("").map(normalize_name) if name_field in df else pd.Series([], dtype=str)
    voter_ids = df[voter_field].fillna("").astype(str).str.strip().str.lower() if voter_field in df else pd.Series([], dtype=str)

    for idx, row in df.iterrows():
        row_issues = []
        for f in required_fields:
            if f in df.columns and not str(row.get(f, "")).strip():
                row_issues.append(f"Missing {f}")
        if "average_confidence" in df.columns and float(row.get("average_confidence", 0) or 0) < low_conf:
            row_issues.append("Low OCR confidence / possible illegible handwriting")
        if date_field in df.columns and str(row.get(date_field, "")).strip() and not looks_like_date(row.get(date_field, "")):
            row_issues.append("Date may be invalid")

        nm = norm_names.iloc[idx] if idx < len(norm_names) else ""
        if nm:
            for j, other in enumerate(norm_names):
                if j == idx or not other:
                    continue
                if nm == other or fuzz.ratio(nm, other) >= 92:
                    row_issues.append(f"Possible duplicate name with row {j + 1}")
                    break
        vid = voter_ids.iloc[idx] if idx < len(voter_ids) else ""
        if vid and voter_ids.tolist().count(vid) > 1:
            dup_rows = [str(i + 1) for i, v in enumerate(voter_ids) if v == vid and i != idx]
            row_issues.append("Duplicate voter ID with row(s) " + ", ".join(dup_rows[:5]))

        issues.append("; ".join(row_issues) if row_issues else "OK")
    out = df.copy()
    out.insert(0, "Status", ["Flagged" if i != "OK" else "OK" for i in issues])
    out.insert(1, "Issues", issues)
    return out


st.title("Petition AI Validator")
st.caption("Upload completed petition pages. The app OCRs each row and flags missing info, duplicates, and likely illegible handwriting.")

with st.sidebar:
    st.header("Form setup")
    fields_text = st.text_area("Columns on the petition form, one per line", "\n".join(DEFAULT_FIELDS), height=170)
    field_names = [f.strip() for f in fields_text.splitlines() if f.strip()]
    first_data_row = st.number_input("Header rows to skip", min_value=0, max_value=5, value=1)
    low_conf = st.slider("Illegible confidence threshold", 0, 100, 45)
    st.divider()
    required_fields = st.multiselect("Required fields", field_names, default=[f for f in field_names if f != "Signature"])
    name_field = st.selectbox("Printed name field", field_names, index=field_names.index("Printed Name") if "Printed Name" in field_names else 0)
    voter_field = st.selectbox("Voter ID/DOB field", field_names, index=field_names.index("Voter ID / DOB") if "Voter ID / DOB" in field_names else 0)
    date_field = st.selectbox("Date signed field", field_names, index=field_names.index("Date Signed") if "Date Signed" in field_names else 0)

uploaded_files = st.file_uploader("Upload petition PDFs or images", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze uploaded petitions", type="primary"):
        all_rows = []
        annotated_images = []
        progress = st.progress(0)
        total = len(uploaded_files)
        page_global = 0
        for fi, uploaded in enumerate(uploaded_files):
            try:
                pages = load_uploaded_file(uploaded)
                for p_i, page in enumerate(pages, start=1):
                    page_global += 1
                    rows, annotated = split_page_into_rows_and_cells(page, field_names, int(first_data_row))
                    annotated_images.append((uploaded.name, p_i, annotated))
                    for r in rows:
                        r.insert if False else None
                        r["file"] = uploaded.name
                        r["page"] = p_i
                        all_rows.append(r)
            except Exception as e:
                st.error(f"Could not process {uploaded.name}: {e}")
            progress.progress((fi + 1) / total)

        if not all_rows:
            st.warning("No rows were detected. Try a clearer scan or lower the header rows to skip.")
        else:
            df = pd.DataFrame(all_rows)
            # Put file/page first
            first_cols = [c for c in ["file", "page", "page_row"] if c in df.columns]
            other_cols = [c for c in df.columns if c not in first_cols]
            df = df[first_cols + other_cols]
            result = validate_rows(df, required_fields, name_field, voter_field, date_field, low_conf)

            st.subheader("Flagged results")
            st.dataframe(result, use_container_width=True, height=420)

            flagged = result[result["Status"] == "Flagged"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows checked", len(result))
            c2.metric("Flagged rows", len(flagged))
            c3.metric("OK rows", len(result) - len(flagged))

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV report", csv, "petition_validation_report.csv", "text/csv")

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                result.to_excel(writer, index=False, sheet_name="Validation Report")
            st.download_button("Download Excel report", excel_buf.getvalue(), "petition_validation_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.subheader("Detected rows on uploaded pages")
            for name, page_no, ann in annotated_images[:10]:
                with st.expander(f"{name} — page {page_no}"):
                    st.image(ann, use_container_width=True)

else:
    st.info("Upload a completed petition PDF or image to start.")

st.warning("Prototype notice: This tool flags entries for human review. It should not be used to automatically reject voter signatures.")
