import io, re
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

st.set_page_config(page_title="Petition Intake Review V6", layout="wide")

FIELDS = ["Date Signed", "Signature", "Printed Name", "Residence Address", "County", "VUID", "Date of Birth"]
REQUIRED_DEFAULT = ["Date Signed", "Printed Name", "Residence Address", "County", "VUID", "Date of Birth"]
# Tuned for Texas SOS petition table in landscape orientation.
COLUMN_FRACTIONS = [0.000, 0.075, 0.190, 0.345, 0.695, 0.780, 0.900, 1.000]

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pages_from_upload(uploaded) -> List[Image.Image]:
    data = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        if convert_from_bytes is None:
            st.error("PDF support needs pdf2image/poppler. Upload JPG/PNG or redeploy with packages.txt included.")
            return []
        return convert_from_bytes(data, dpi=250)
    return [Image.open(io.BytesIO(data)).convert("RGB")]

def normalize_page(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def find_table_bbox(img: np.ndarray) -> Tuple[int,int,int,int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    blur = cv2.GaussianBlur(gray,(3,3),0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,12)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(max(50,w//14),1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(25,h//22)))
    grid = cv2.add(cv2.morphologyEx(th,cv2.MORPH_OPEN,horiz_kernel), cv2.morphologyEx(th,cv2.MORPH_OPEN,vert_kernel))
    contours,_ = cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,bw,bh = cv2.boundingRect(c)
        if bw > .65*w and .18*h < y < .62*h and .15*h < bh < .45*h:
            candidates.append((bw*bh,x,y,bw,bh))
    if candidates:
        _,x,y,bw,bh = max(candidates)
        pad=8
        return max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad)
    return int(.035*w), int(.32*h), int(.965*w), int(.735*h)

def row_bounds(table_img: np.ndarray, row_start: float, row_end: float, rows_per_page: int):
    h,w = table_img.shape[:2]
    top = int(row_start*h); bottom=int(row_end*h)
    rh=(bottom-top)/rows_per_page
    return [(int(top+i*rh)+2, int(top+(i+1)*rh)-2) for i in range(rows_per_page)]

def crop_cell(row_img: np.ndarray, field_idx: int) -> np.ndarray:
    h,w = row_img.shape[:2]
    x1 = int(COLUMN_FRACTIONS[field_idx]*w)+2
    x2 = int(COLUMN_FRACTIONS[field_idx+1]*w)-2
    py=max(1,int(.05*h))
    return row_img[py:max(py+1,h-py), max(0,x1):min(w,x2)]

def ink_ratio(cell: np.ndarray) -> float:
    gray = cv2.cvtColor(cell,cv2.COLOR_BGR2GRAY)
    # adaptive makes this much less sensitive to shadow/yellow paper
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    h,w = th.shape
    if h>10 and w>10:
        th = th[4:h-4,4:w-4]
    # remove table-line residue/noise
    kernel=np.ones((2,2),np.uint8)
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
    return float(np.count_nonzero(th))/float(th.size) if th.size else 0.0

def ocr(cell: np.ndarray) -> str:
    gray=cv2.cvtColor(cell,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    txt=pytesseract.image_to_string(gray,config="--psm 7 --oem 3").strip()
    return re.sub(r"\s+"," ",txt)

def norm(s):
    return re.sub(r"[^A-Z0-9]","",str(s or "").upper())

def analyze_files(uploaded_files, rows_per_page, row_start, row_end, blank_threshold):
    rows=[]; previews=[]; field_crops=[]
    for f in uploaded_files:
        pages=pages_from_upload(f)
        for pnum,page in enumerate(pages, start=1):
            cvimg=normalize_page(pil_to_cv(page))
            x1,y1,x2,y2=find_table_bbox(cvimg)
            table=cvimg[y1:y2,x1:x2]
            preview=cvimg.copy(); cv2.rectangle(preview,(x1,y1),(x2,y2),(0,180,0),4)
            for line,(r1,r2) in enumerate(row_bounds(table,row_start,row_end,rows_per_page), start=1):
                row_img=table[r1:r2,:]
                cv2.rectangle(preview,(x1,y1+r1),(x2,y1+r2),(255,160,0),2)
                rec={"File":f.name,"Page":pnum,"Row":line}
                rec["Location"] = f"Page {pnum}, Row {line}"
                for i,field in enumerate(FIELDS):
                    cell=crop_cell(row_img,i)
                    ratio=ink_ratio(cell)
                    rec[field+" Filled"] = ratio >= blank_threshold
                    rec[field+" Ink"] = round(ratio,4)
                    # Use OCR as a starter only. User can edit it before duplicate report.
                    rec[field] = ocr(cell) if field in ["Printed Name","VUID","Residence Address"] and ratio >= blank_threshold else ""
                rows.append(rec)
            previews.append((f.name,pnum,cv_to_pil(preview)))
    return pd.DataFrame(rows), previews

def build_issues(df, required):
    issue_rows=[]
    for _,r in df.iterrows():
        missing=[]
        for field in required:
            filled_col=field+" Filled"
            if filled_col in r and not bool(r[filled_col]):
                missing.append(field)
        if missing:
            issue_rows.append({"Issue":"Missing information","Location":r["Location"],"Details":"Missing " + ", ".join(missing)})
    # Duplicates use the editable text, not raw OCR guesses.
    for col,label,minlen in [("Printed Name","Possible duplicate name",5),("VUID","Possible duplicate VUID",4)]:
        if col not in df: continue
        temp=df.copy()
        temp["_key"]=temp[col].map(norm)
        temp=temp[temp["_key"].str.len()>=minlen]
        for key,g in temp.groupby("_key"):
            if len(g)>1:
                val=str(g.iloc[0][col])
                locs="; ".join(g["Location"].astype(str).tolist())
                issue_rows.append({"Issue":label,"Location":locs,"Details":f"{label}: {val}"})
    return pd.DataFrame(issue_rows)

st.title("Petition Intake Review V6")
st.caption("Designed for your team’s goal: reduce line-by-line scanning by showing missing fields and duplicate entries. It is not voter verification.")

with st.sidebar:
    st.header("Settings")
    required=st.multiselect("Fields to check as required", FIELDS, default=REQUIRED_DEFAULT)
    st.write("Use these only if orange row boxes do not line up.")
    rows_per_page=st.number_input("Rows per petition page",1,20,10)
    row_start=st.slider("Top of signer rows",0.15,0.50,0.29,0.005)
    row_end=st.slider("Bottom of signer rows",0.60,0.98,0.88,0.005)
    blank_threshold=st.slider("Blank box threshold",0.0001,0.0200,0.0007,0.0001)

uploaded=st.file_uploader("Upload petition PDF/images", type=["pdf","png","jpg","jpeg","tif","tiff"], accept_multiple_files=True)
if uploaded and st.button("Analyze pages", type="primary"):
    df,previews=analyze_files(uploaded, int(rows_per_page), row_start, row_end, blank_threshold)
    st.session_state["df"]=df
    st.session_state["previews"]=previews

if "df" in st.session_state:
    st.subheader("Step 1: Check the row boxes")
    st.caption("Green = table found. Orange = rows checked. If orange boxes are shifted, adjust Top/Bottom of signer rows and re-analyze.")
    for fname,pnum,img in st.session_state.get("previews",[])[:6]:
        with st.expander(f"Detected rows: {fname} page {pnum}", expanded=False):
            st.image(img, use_container_width=True)

    st.subheader("Step 2: Review/edit names and VUIDs for duplicate checking")
    st.caption("OCR handwriting will not be perfect. Fix names/VUIDs here, then the duplicate report below will update accurately.")
    show_cols=["File","Page","Row","Printed Name","VUID","Residence Address"] + [f+" Filled" for f in REQUIRED_DEFAULT if f+" Filled" in st.session_state["df"].columns]
    edited=st.data_editor(st.session_state["df"][show_cols], num_rows="fixed", use_container_width=True, key="editor")

    # merge edited text/filled fields back for issue generation
    working=st.session_state["df"].copy()
    for c in edited.columns:
        working[c]=edited[c]
    issues=build_issues(working, required)
    st.subheader("Clean issue report")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows checked", len(working))
    c2.metric("Issues found", len(issues))
    c3.metric("Rows without missing required fields", len(working)-sum(working.apply(lambda r:any((f+" Filled" in working.columns and not bool(r[f+" Filled"])) for f in required),axis=1)))
    if issues.empty:
        st.success("No missing required fields or duplicates found based on the current table.")
    else:
        st.dataframe(issues, use_container_width=True, hide_index=True)
    st.download_button("Download issue report CSV", issues.to_csv(index=False).encode("utf-8"), "petition_intake_issue_report.csv", "text/csv")
    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
        issues.to_excel(writer,index=False,sheet_name="Issues")
        working.to_excel(writer,index=False,sheet_name="Reviewed Rows")
    st.download_button("Download Excel report", xbuf.getvalue(), "petition_intake_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload a petition file and click Analyze pages.")

st.warning("Important: Handwritten OCR is used only as a starter. For accurate duplicate checking, staff should quickly correct the Printed Name/VUID fields in the editable table.")
