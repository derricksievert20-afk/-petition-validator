import io, re, zipfile
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title='Petition Intake Review V8', layout='wide')
st.title('Petition Intake Review V8')
st.caption('Human-in-the-loop intake tool: mark filled fields, enter names/VUIDs, then generate a clean missing-fields + duplicate report.')

FIELDS = ['Date Signed','Signature','Printed Name','Residence Address','County','VUID','Date of Birth']
REQUIRED = ['Date Signed','Signature','Printed Name','Residence Address','County','VUID','Date of Birth']

def pdf_to_images(uploaded):
    try:
        import fitz
        data = uploaded.read()
        doc = fitz.open(stream=data, filetype='pdf')
        imgs=[]
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
            imgs.append((i+1, Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')))
        return imgs
    except Exception as e:
        st.error(f'Could not read PDF: {e}')
        return []

def load_uploaded_files(files):
    pages=[]
    for f in files:
        if f.type == 'application/pdf' or f.name.lower().endswith('.pdf'):
            for pnum,img in pdf_to_images(f):
                pages.append({'file': f.name, 'page': pnum, 'image': img})
        else:
            try:
                pages.append({'file': f.name, 'page': 1, 'image': Image.open(f).convert('RGB')})
            except Exception as e:
                st.warning(f'Could not read {f.name}: {e}')
    return pages

def crop_rows(img, top_pct, bottom_pct, left_pct, right_pct, rows):
    w,h=img.size
    x1=int(w*left_pct/100); x2=int(w*right_pct/100)
    y1=int(h*top_pct/100); y2=int(h*bottom_pct/100)
    table=img.crop((x1,y1,x2,y2))
    tw,th=table.size
    out=[]
    for r in range(rows):
        ry1=int(th*r/rows); ry2=int(th*(r+1)/rows)
        out.append(table.crop((0,ry1,tw,ry2)))
    return out

def norm_name(s):
    s=(s or '').lower().strip()
    s=re.sub(r'[^a-z\s]', '', s)
    return re.sub(r'\s+', ' ', s)

def norm_vuid(s):
    return re.sub(r'\D', '', s or '')

with st.sidebar:
    st.header('Form crop setup')
    st.write('Adjust these if row crops do not line up with the petition table.')
    rows_per_page=st.number_input('Signature rows per page', 1, 30, 10)
    top_pct=st.slider('Table top %', 20, 80, 38)
    bottom_pct=st.slider('Table bottom %', 45, 95, 72)
    left_pct=st.slider('Table left %', 0, 30, 5)
    right_pct=st.slider('Table right %', 70, 100, 96)
    st.divider()
    st.write('Duplicate checks use only typed Printed Name and VUID fields.')

files=st.file_uploader('Upload petition PDFs or images', type=['pdf','png','jpg','jpeg','tif','tiff'], accept_multiple_files=True)

if not files:
    st.info('Upload petition pages to begin. This version intentionally avoids unreliable automatic rejection and uses quick staff review.')
    st.stop()

pages=load_uploaded_files(files)
if not pages:
    st.stop()

st.subheader('Step 1: Review rows')
st.write('For each row, uncheck any field that is visibly blank. Type Printed Name and VUID only when readable so duplicates can be checked.')

entries=[]
for pi,p in enumerate(pages):
    st.markdown(f'### {p["file"]} — Page {p["page"]}')
    rows=crop_rows(p['image'], top_pct, bottom_pct, left_pct, right_pct, int(rows_per_page))
    for ri,row_img in enumerate(rows, start=1):
        with st.expander(f'Row {ri}', expanded=False):
            st.image(row_img, use_container_width=True)
            cols=st.columns(len(FIELDS))
            filled={}
            for ci,field in enumerate(FIELDS):
                filled[field]=cols[ci].checkbox(field, value=True, key=f'filled_{pi}_{ri}_{field}')
            c1,c2=st.columns(2)
            printed=c1.text_input('Printed Name for duplicate check', key=f'name_{pi}_{ri}')
            vuid=c2.text_input('VUID for duplicate check', key=f'vuid_{pi}_{ri}')
            entries.append({'File':p['file'],'Page':p['page'],'Row':ri,'Printed Name':printed,'VUID':vuid, **{f'{f} filled':filled[f] for f in FIELDS}})

if st.button('Generate clean review report', type='primary'):
    issues=[]
    for e in entries:
        missing=[f for f in REQUIRED if not e.get(f'{f} filled', False)]
        if missing:
            issues.append({'Issue Type':'Missing field','Location':f"Page {e['Page']}, Row {e['Row']}", 'Details':', '.join(missing), 'File':e['File']})
    names={}
    vuids={}
    for e in entries:
        n=norm_name(e.get('Printed Name'))
        v=norm_vuid(e.get('VUID'))
        if n: names.setdefault(n,[]).append(e)
        if v: vuids.setdefault(v,[]).append(e)
    for n, group in names.items():
        if len(group)>1:
            locs='; '.join([f"Page {g['Page']}, Row {g['Row']}" for g in group])
            display=next((g['Printed Name'] for g in group if g['Printed Name']), n)
            issues.append({'Issue Type':'Possible duplicate printed name','Location':locs,'Details':display,'File':'Multiple' if len({g['File'] for g in group})>1 else group[0]['File']})
    for v, group in vuids.items():
        if len(group)>1:
            locs='; '.join([f"Page {g['Page']}, Row {g['Row']}" for g in group])
            issues.append({'Issue Type':'Possible duplicate VUID','Location':locs,'Details':v,'File':'Multiple' if len({g['File'] for g in group})>1 else group[0]['File']})
    df=pd.DataFrame(issues) if issues else pd.DataFrame(columns=['Issue Type','Location','Details','File'])
    st.subheader('Clean report')
    st.dataframe(df, use_container_width=True)
    st.download_button('Download CSV report', df.to_csv(index=False).encode(), 'petition_intake_report.csv', 'text/csv')
    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Review Report')
        pd.DataFrame(entries).to_excel(writer, index=False, sheet_name='Reviewed Entries')
    st.download_button('Download Excel report', xbuf.getvalue(), 'petition_intake_report.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
