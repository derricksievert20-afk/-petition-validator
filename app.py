import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

st.set_page_config(page_title="Petition Intake Checker V5", layout="wide")

@dataclass
class CellResult:
    text: str
    confidence: float
    ink_ratio: float

FIELDS = ["Date Signed", "Signature", "Printed Name", "Residence Address", "County", "Voter ID / VUID", "Date of Birth"]
REQUIRED_DEFAULT = ["Printed Name", "Residence Address", "County", "Voter ID / VUID", "Date of Birth", "Date Signed"]

# Approximate Texas SOS petition signature table columns as fractions across the detected table width.
# These are deliberately conservative and can be adjusted in the sidebar.
COLUMN_FRACTIONS = [0.000, 0.075, 0.190, 0.345, 0.695, 0.780, 0.900, 1.000]

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pages_from_upload(uploaded) -> List[Image.Image]:
    data = uploaded.read()
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        if convert_from_bytes is None:
            st.error("PDF support is missing. Make sure pdf2image and poppler are installed.")
            return []
        return convert_from_bytes(data, dpi=250)
    return [Image.open(io.BytesIO(data)).convert("RGB")]

def normalize_page(img: np.ndarray) -> np.ndarray:
    # Rotate portrait-ish images to landscape if needed, because this form is normally landscape.
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def find_table_bbox(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the main signature table. Falls back to expected relative area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)

    # Horizontal and vertical line masks
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w//18), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h//30)))
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel)
    grid = cv2.add(horiz, vert)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        # Main table is wide and in upper/middle part, not the affidavit at bottom.
        if bw > 0.65*w and 0.12*h < y < 0.70*h and 0.12*h < bh < 0.55*h:
            candidates.append((area, x, y, bw, bh))
    if candidates:
        _, x, y, bw, bh = max(candidates)
        pad = 6
        return max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad)

    # Fallback based on this exact petition form layout.
    return int(0.035*w), int(0.32*h), int(0.965*w), int(0.74*h)

def find_row_bounds(table_img: np.ndarray, expected_rows: int = 10, skip_header_lines: int = 3) -> List[Tuple[int,int]]:
    """Use the known Texas SOS petition layout.

    Generic horizontal-line detection was over-sensitive and caused false flags.
    This version assumes the standard form: column headings at top, 10 signer rows
    below. The sidebar allows small adjustments if a scan is cropped differently.
    """
    h, w = table_img.shape[:2]
    data_top = int((0.28 + 0.015 * (skip_header_lines - 3)) * h)
    data_bottom = int(0.88 * h)
    row_h = (data_bottom - data_top) / expected_rows
    return [(int(data_top+i*row_h)+2, int(data_top+(i+1)*row_h)-2) for i in range(expected_rows)]

def crop_cell(row_img: np.ndarray, field_idx: int, left_margin_pct=0.0, right_margin_pct=0.0) -> np.ndarray:
    h, w = row_img.shape[:2]
    x1 = int(COLUMN_FRACTIONS[field_idx] * w)
    x2 = int(COLUMN_FRACTIONS[field_idx+1] * w)
    pad_x = 2
    pad_y = max(1, int(0.04*h))
    return row_img[pad_y:max(pad_y+1, h-pad_y), max(0,x1+pad_x):min(w,x2-pad_x)]

def ink_ratio(cell: np.ndarray) -> float:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # remove very light background/noise
    _, th = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    # reduce influence of table borders by ignoring edges
    h,w = th.shape
    if h > 8 and w > 8:
        th = th[3:h-3, 3:w-3]
    return float(np.count_nonzero(th)) / float(th.size) if th.size else 0.0

def ocr_cell(cell: np.ndarray) -> CellResult:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    config = "--psm 7 --oem 3"
    text = pytesseract.image_to_string(gray, config=config).strip()
    try:
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DATAFRAME)
        confs = pd.to_numeric(data["conf"], errors="coerce")
        confs = confs[confs >= 0]
        conf = float(confs.mean()) if len(confs) else 0.0
    except Exception:
        conf = 0.0
    return CellResult(text=text, confidence=conf, ink_ratio=ink_ratio(cell))

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_key(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def human_location(row) -> str:
    return f"page {int(row['Page'])}, row {int(row['Line'])}"

def make_clean_issue_summary(row) -> str:
    issues = row.get("Issues", "") or ""
    return issues if issues else "OK"

def analyze_page(img: Image.Image, filename: str, page_num: int, required_fields: List[str], low_conf_threshold: int, blank_threshold: float, skip_header_lines: int, validate_dates: bool, flag_low_conf: bool) -> Tuple[List[Dict], Image.Image]:
    cvimg = normalize_page(pil_to_cv(img))
    x1,y1,x2,y2 = find_table_bbox(cvimg)
    table = cvimg[y1:y2, x1:x2]
    row_bounds = find_row_bounds(table, expected_rows=10, skip_header_lines=skip_header_lines)

    # draw detected area/rows for review
    preview = cvimg.copy()
    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,180,0), 4)

    rows = []
    for idx, (ry1, ry2) in enumerate(row_bounds, start=1):
        abs_y1, abs_y2 = y1+ry1, y1+ry2
        cv2.rectangle(preview, (x1, abs_y1), (x2, abs_y2), (255,160,0), 2)
        row_img = table[ry1:ry2, :]
        record = {"File": filename, "Page": page_num, "Line": idx}
        issues = []
        low_conf_fields = []
        for fi, field in enumerate(FIELDS):
            cell = crop_cell(row_img, fi)
            res = ocr_cell(cell)
            text = clean_text(res.text)
            record[field] = text
            record[f"{field} confidence"] = round(res.confidence, 1)
            record[f"{field} ink"] = round(res.ink_ratio, 4)

            is_blank = res.ink_ratio < blank_threshold
            record[f"{field} filled"] = "No" if is_blank else "Yes"
            # Conservative intake mode: flag missing only when the box appears visually empty.
            if field in required_fields and is_blank:
                issues.append(f"Missing {field}")
            # Only call illegible if there is writing but OCR is weak; do not make it automatically invalid.
            if (not is_blank) and field in ["Printed Name", "Residence Address", "County", "Voter ID / VUID", "Date of Birth"] and res.confidence < low_conf_threshold:
                low_conf_fields.append(field)

        if flag_low_conf and low_conf_fields:
            issues.append("Needs human review: possible illegible " + ", ".join(low_conf_fields))

        if validate_dates:
            for df in ["Date Signed", "Date of Birth"]:
                val = record.get(df, "")
                if val and not re.search(r"\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}", val):
                    issues.append(f"Review {df} format")

        record["Issues"] = "; ".join(issues) if issues else ""
        record["Status"] = "Flagged" if issues else "OK"
        rows.append(record)
    return rows, cv_to_pil(preview)

st.title("Petition Intake Checker V5")
st.caption("Intake review focused on filled/missing fields and possible duplicates. This version uses the Texas petition layout instead of generic OCR rows.")

with st.sidebar:
    st.header("Form setup")
    st.write("This version is tuned for the Texas SOS local petition form shown in your project.")
    required = st.multiselect("Required fields", FIELDS, default=REQUIRED_DEFAULT)
    skip_header_lines = st.number_input("Row start adjustment", min_value=1, max_value=5, value=3, help="Use 3 normally. Increase/decrease only if the orange row boxes are shifted.")
    blank_threshold = st.slider("Blank field sensitivity", 0.0002, 0.020, 0.0010, 0.0002, help="Lower = fewer false missing-field flags. Increase only if truly blank boxes are not being caught.")
    duplicate_conf = st.slider("Minimum OCR confidence for duplicate checks", 0, 100, 55, help="Higher = fewer false duplicate alerts. Duplicate checks only use text above this confidence.")
    flag_low_conf = False
    low_conf = 0
    validate_dates = False

uploaded_files = st.file_uploader("Upload completed petition PDFs or images", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze uploaded petitions", type="primary"):
        all_rows = []
        previews = []
        progress = st.progress(0)
        total_pages = 0
        pages_by_file = []
        for f in uploaded_files:
            pages = pages_from_upload(f)
            pages_by_file.append((f.name, pages))
            total_pages += len(pages)
        done = 0
        for fname, pages in pages_by_file:
            for pi, page in enumerate(pages, start=1):
                rows, preview = analyze_page(page, fname, pi, required, low_conf, blank_threshold, int(skip_header_lines), validate_dates, flag_low_conf)
                all_rows.extend(rows)
                previews.append((fname, pi, preview))
                done += 1
                progress.progress(done / max(total_pages, 1))

        df = pd.DataFrame(all_rows)
        # Duplicate checks after OCR
        if not df.empty:
            df["Normalized Printed Name"] = df.apply(lambda r: norm_key(r.get("Printed Name", "")) if r.get("Printed Name confidence", 0) >= duplicate_conf and len(norm_key(r.get("Printed Name", ""))) >= 5 else "", axis=1)
            df["Normalized VUID"] = df.apply(lambda r: norm_key(r.get("Voter ID / VUID", "")) if r.get("Voter ID / VUID confidence", 0) >= duplicate_conf and len(norm_key(r.get("Voter ID / VUID", ""))) >= 4 else "", axis=1)

            # Duplicate checks with exact page/row references, e.g. page 1 row 1 and page 2 row 1 duplicate name James Smith.
            for key_col, display_col, label in [
                ("Normalized Printed Name", "Printed Name", "Duplicate name"),
                ("Normalized VUID", "Voter ID / VUID", "Duplicate VUID"),
            ]:
                groups = df[df[key_col].ne("")].groupby(key_col).groups
                for _, idxs in groups.items():
                    idxs = list(idxs)
                    if len(idxs) > 1:
                        shown_value = clean_text(str(df.loc[idxs[0], display_col]))
                        locs = [human_location(df.loc[i]) for i in idxs]
                        msg = f"{label}: {shown_value} appears on " + " and ".join(locs)
                        for i in idxs:
                            current = df.loc[i, "Issues"] or ""
                            df.loc[i, "Issues"] = (current + "; " if current else "") + msg

            df["Status"] = np.where(df["Issues"].fillna("").ne(""), "Flagged", "OK")
            df["Issue Summary"] = df.apply(make_clean_issue_summary, axis=1)

        st.subheader("Review results")
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows checked", len(df))
        c2.metric("Flagged rows", int((df["Status"] == "Flagged").sum()) if not df.empty else 0)
        c3.metric("OK rows", int((df["Status"] == "OK").sum()) if not df.empty else 0)
        clean_cols = ["Status", "Issue Summary", "File", "Page", "Line", "Date Signed filled", "Printed Name filled", "Residence Address filled", "County filled", "Voter ID / VUID filled", "Date of Birth filled", "Printed Name", "Voter ID / VUID"]
        clean_df = df[[c for c in clean_cols if c in df.columns]].copy()
        st.dataframe(clean_df, use_container_width=True)

        with st.expander("Show raw OCR details"):
            st.dataframe(df, use_container_width=True)

        csv = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV report", data=csv, file_name="petition_validation_report.csv", mime="text/csv")
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            clean_df.to_excel(writer, index=False, sheet_name="Validation Report")
        st.download_button("Download Excel report", data=xbuf.getvalue(), file_name="petition_validation_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Detected signature rows")
        st.caption("Green box = detected signature table. Orange boxes = rows analyzed. If these are off, adjust 'Header grid lines' or scan straighter.")
        for fname, pi, preview in previews[:10]:
            st.write(f"{fname} — page {pi}")
            st.image(preview, use_container_width=True)
else:
    st.info("Upload a petition PDF or image to start.")

st.warning("Prototype notice: this is an intake review tool. It checks whether boxes appear filled and looks for obvious duplicates; staff still make final decisions.")
