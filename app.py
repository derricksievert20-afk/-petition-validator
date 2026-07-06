import io
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import fitz
import pandas as pd
import streamlit as st
from PIL import Image
from rapidfuzz import fuzz

st.set_page_config(page_title="Petition Intake Review Assistant", layout="wide")

FIELDS = [
    "Date Signed", "Signature", "Printed Name", "Residence Address",
    "County", "VUID", "Date of Birth"
]
REQUIRED_DEFAULT = ["Date Signed", "Signature", "Printed Name", "Residence Address", "County", "VUID", "Date of Birth"]

# Approximate Texas petition table crop coordinates as percentages of page size.
# These can be adjusted in the sidebar if a scan is shifted.
DEFAULT = {
    "table_left": 0.045,
    "table_top": 0.325,
    "table_right": 0.965,
    "table_bottom": 0.755,
    "header_frac": 0.14,
    # relative widths inside table after the Name section header:
    "date_w": 0.080,
    "sig_w": 0.120,
    "printed_w": 0.150,
    "address_w": 0.335,
    "county_w": 0.090,
    "vuid_w": 0.115,
    "dob_w": 0.110,
}

@dataclass
class PageItem:
    file_name: str
    page_no: int
    image: Image.Image


def pdf_to_images(uploaded_file) -> List[Image.Image]:
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images


def load_uploaded(files) -> List[PageItem]:
    pages = []
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            for idx, img in enumerate(pdf_to_images(f), start=1):
                pages.append(PageItem(name, idx, img))
        else:
            img = Image.open(f).convert("RGB")
            pages.append(PageItem(name, 1, img))
    return pages


def crop_box(img: Image.Image, box_pct: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    l, t, r, b = box_pct
    return img.crop((int(l*w), int(t*h), int(r*w), int(b*h)))


def get_field_boxes(settings: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    left = settings["table_left"]
    right = settings["table_right"]
    width = right - left
    rels = [
        ("Date Signed", settings["date_w"]),
        ("Signature", settings["sig_w"]),
        ("Printed Name", settings["printed_w"]),
        ("Residence Address", settings["address_w"]),
        ("County", settings["county_w"]),
        ("VUID", settings["vuid_w"]),
        ("Date of Birth", settings["dob_w"]),
    ]
    total = sum(v for _, v in rels)
    x = left
    boxes = {}
    for field, frac in rels:
        w = width * (frac / total)
        boxes[field] = (x, x + w)
        x += w
    return boxes


def row_field_crop(img, settings, row_num, field):
    top = settings["table_top"]
    bottom = settings["table_bottom"]
    header_h = (bottom - top) * settings["header_frac"]
    body_top = top + header_h
    row_h = (bottom - body_top) / 10
    y1 = body_top + (row_num - 1) * row_h
    y2 = body_top + row_num * row_h
    x1, x2 = get_field_boxes(settings)[field]
    pad_x = 0.003
    pad_y = 0.003
    return crop_box(img, (max(0, x1+pad_x), max(0, y1+pad_y), min(1, x2-pad_x), min(1, y2-pad_y)))


def normalize_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_vuid(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def duplicate_issues(df: pd.DataFrame, fuzzy_threshold: int) -> List[Dict]:
    issues = []
    # exact VUID dupes
    vmap = {}
    for _, r in df.iterrows():
        v = normalize_vuid(r.get("VUID", ""))
        if len(v) >= 5:
            vmap.setdefault(v, []).append(r)
    for v, rows in vmap.items():
        if len(rows) > 1:
            locs = ", ".join([f"Page {x['Page']}, Row {x['Row']}" for x in rows])
            issues.append({"Issue Type":"Possible duplicate VUID", "Details":f"VUID {v} appears on {locs}", "Rows": locs})

    # exact/fuzzy name dupes
    rows = list(df.to_dict("records"))
    seen_pairs = set()
    for i in range(len(rows)):
        n1 = normalize_name(rows[i].get("Printed Name", ""))
        if len(n1) < 4:
            continue
        for j in range(i+1, len(rows)):
            n2 = normalize_name(rows[j].get("Printed Name", ""))
            if len(n2) < 4:
                continue
            score = 100 if n1 == n2 else fuzz.token_sort_ratio(n1, n2)
            if score >= fuzzy_threshold:
                key = tuple(sorted([(rows[i]["Page"], rows[i]["Row"]), (rows[j]["Page"], rows[j]["Row"])]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                issues.append({
                    "Issue Type":"Possible duplicate printed name",
                    "Details":f"{rows[i].get('Printed Name','')} / {rows[j].get('Printed Name','')} match score {score}",
                    "Rows":f"Page {rows[i]['Page']}, Row {rows[i]['Row']} and Page {rows[j]['Page']}, Row {rows[j]['Row']}"
                })
    return issues

st.title("Petition Intake Review Assistant")
st.caption("Goal: reduce line-by-line scanning by organizing entries, checking whether fields appear filled, and finding possible duplicates. Not voter eligibility verification.")

with st.sidebar:
    st.header("Review setup")
    required_fields = st.multiselect("Fields to check", FIELDS, default=REQUIRED_DEFAULT)
    fuzzy_threshold = st.slider("Duplicate name sensitivity", 80, 100, 94, help="Higher = fewer false duplicate flags")
    st.divider()
    st.subheader("Table alignment")
    st.caption("Use defaults first. Adjust only if row crops don't line up with the form.")
    settings = DEFAULT.copy()
    settings["table_left"] = st.slider("Table left", 0.00, 0.20, float(DEFAULT["table_left"]), 0.005)
    settings["table_top"] = st.slider("Table top", 0.20, 0.45, float(DEFAULT["table_top"]), 0.005)
    settings["table_right"] = st.slider("Table right", 0.80, 1.00, float(DEFAULT["table_right"]), 0.005)
    settings["table_bottom"] = st.slider("Table bottom", 0.60, 0.90, float(DEFAULT["table_bottom"]), 0.005)
    settings["header_frac"] = st.slider("Header height inside table", 0.05, 0.25, float(DEFAULT["header_frac"]), 0.01)

files = st.file_uploader("Upload petition PDFs/images", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if not files:
    st.info("Upload completed petition pages to begin.")
    st.stop()

pages = load_uploaded(files)
st.success(f"Loaded {len(pages)} page(s).")

st.subheader("1) Confirm fields are filled and type key duplicate fields")
st.caption("This avoids false flags from bad handwriting OCR. You only mark missing boxes and type/correct Printed Name and VUID for duplicate checking.")

records = []
for p_idx, page in enumerate(pages):
    with st.expander(f"{page.file_name} — Page {page.page_no}", expanded=(p_idx == 0)):
        st.image(page.image, caption="Full page preview", use_container_width=True)
        for row in range(1, 11):
            st.markdown(f"**Row {row}**")
            cols = st.columns([1.2, 1.2, 1.2, 2, 1, 1.2, 1.2])
            row_data = {"File": page.file_name, "Page": page.page_no, "Row": row}
            filled = {}
            for idx, field in enumerate(FIELDS):
                with cols[idx]:
                    st.image(row_field_crop(page.image, settings, row, field), caption=field, use_container_width=True)
                    filled[field] = st.checkbox("Filled", value=True, key=f"filled_{p_idx}_{row}_{field}")
            c1, c2 = st.columns(2)
            with c1:
                row_data["Printed Name"] = st.text_input("Printed Name for duplicate check", key=f"name_{p_idx}_{row}")
            with c2:
                row_data["VUID"] = st.text_input("VUID for duplicate check", key=f"vuid_{p_idx}_{row}")
            for field in FIELDS:
                row_data[f"{field} filled"] = "Yes" if filled[field] else "No"
            missing = [f for f in required_fields if not filled.get(f, True)]
            row_data["Missing Fields"] = "; ".join(missing)
            row_data["Status"] = "Flagged" if missing else "OK"
            row_data["Issue Summary"] = "OK" if not missing else "Missing " + "; Missing ".join(missing)
            records.append(row_data)
            st.divider()

if st.button("Generate intake report", type="primary"):
    df = pd.DataFrame(records)
    missing_df = df[df["Missing Fields"].astype(str).str.len() > 0].copy()
    dup_issues = duplicate_issues(df, fuzzy_threshold)
    dup_df = pd.DataFrame(dup_issues) if dup_issues else pd.DataFrame(columns=["Issue Type", "Details", "Rows"])

    st.subheader("2) Clean issue list")
    st.metric("Rows reviewed", len(df))
    st.metric("Rows with missing fields", len(missing_df))
    st.metric("Possible duplicate issues", len(dup_df))

    if len(missing_df):
        st.markdown("### Missing field flags")
        show = missing_df[["File", "Page", "Row", "Issue Summary"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.success("No missing required fields were marked.")

    if len(dup_df):
        st.markdown("### Duplicate flags")
        st.dataframe(dup_df, use_container_width=True, hide_index=True)
    else:
        st.success("No possible duplicates found from entered names/VUIDs.")

    full_report = df.copy()
    csv = full_report.to_csv(index=False).encode("utf-8")
    st.download_button("Download reviewed rows CSV", csv, "petition_review_rows.csv", "text/csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Reviewed Rows")
        missing_df.to_excel(writer, index=False, sheet_name="Missing Fields")
        dup_df.to_excel(writer, index=False, sheet_name="Duplicate Flags")
    st.download_button("Download Excel report", output.getvalue(), "petition_intake_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
