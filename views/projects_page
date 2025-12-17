import streamlit as st
from datetime import date
from db import list_projects, create_project
from r2 import get_s3, project_prefix, ensure_project_marker, list_files, upload_file, download_bytes, delete_key

def render_projects_page():
    st.title("Projects")
    s3 = get_s3()

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Loo projekt")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        name = st.text_input("Projekti nimi", placeholder="nt Objekt_01")
    with c2:
        start = st.date_input("Algus (valikuline)", value=None)
    with c3:
        end = st.date_input("Lõpp (tähtaeg)", value=date.today())

    if st.button("Loo projekt", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta projekti nimi.")
        else:
            create_project(name.strip(), start_date=start, end_date=end)
            ensure_project_marker(s3, project_prefix(name.strip()))
            st.success("Projekt loodud.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    projects = list_projects()
    if not projects:
        st.info("Pole veel projekte.")
        return

    proj_names = [p["name"] for p in projects]
    selected_name = st.selectbox("Vali projekt", proj_names)
    p = next(x for x in projects if x["name"] == selected_name)
    prefix = project_prefix(selected_name)

    st.caption(f"Projekt: **{p['name']}** • Tähtaeg: **{p.get('end_date')}**")

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("📤 Failid (Cloudflare R2)")
    uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True)
    if uploads:
        for up in uploads:
            upload_file(s3, prefix, up)
        st.success(f"Üles laaditud {len(uploads)} faili.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📄 Projekti failid")
    files = list_files(s3, prefix)
    if not files:
        st.info("Selles projektis pole veel faile.")
        return

    for f in files:
        a,b,c = st.columns([6,2,2])
        with a:
            st.write(f"📄 {f['name']}")
            st.caption(f"{f['size']/1024:.1f} KB")
        with b:
            data = download_bytes(s3, f["key"])
            st.download_button("⬇️ Laadi alla", data=data, file_name=f["name"], key=f"dl_{f['key']}", use_container_width=True)
        with c:
            if st.button("🗑️ Kustuta", key=f"del_{f['key']}", use_container_width=True):
                delete_key(s3, f["key"])
                st.rerun()
