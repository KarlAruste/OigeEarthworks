import streamlit as st
from datetime import date
from db import list_projects, create_project, get_project, set_project_landxml
from r2 import get_s3, project_prefix, ensure_project_marker, list_files, upload_file, download_bytes, delete_key
from landxml import parse_landxml_total_volume

def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    if "project_id" not in st.session_state:
        st.session_state["project_id"] = None

    # --- Create project ---
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

    # --- Project list ---
    st.subheader("📌 Minu projektid")
    if not projects:
        st.info("Pole veel projekte.")
        return

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.caption("Vali projekt:")
        for p in projects:
            if st.button(p["name"], use_container_width=True, key=f"projbtn_{p['id']}"):
                st.session_state["project_id"] = p["id"]
                st.rerun()

    with right:
        if not st.session_state["project_id"]:
            st.info("Vali vasakult projekt.")
            return

        p = get_project(st.session_state["project_id"])
        if not p:
            st.session_state["project_id"] = None
            st.rerun()

        st.subheader(p["name"])
        st.caption(f"Tähtaeg: {p.get('end_date')}")

        # Planned volume (from LandXML)
        planned = p.get("planned_volume_m3")
        if planned is not None:
            st.success(f"Planeeritud maht (LandXML): {float(planned):.2f} m³")
        else:
            st.info("Planeeritud maht puudub. Laadi LandXML üles.")

        # --- LandXML upload ---
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📄 LandXML (planeeritud mahu arvutus)")
        landxml = st.file_uploader("Laadi üles LandXML (.xml)", type=["xml"])
        if landxml and st.button("Salvesta LandXML", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)
            total_vol = parse_landxml_total_volume(xml_bytes)
            set_project_landxml(p["id"], key, total_vol)
            if total_vol is None:
                st.warning("LandXML-ist ei leitud kindlat mahu väärtust. Fail salvestati, aga mahtu ei suutnud tuvastada.")
            else:
                st.success(f"LandXML salvestatud. Leitud maht: {total_vol:.2f} m³")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # --- R2 files ---
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📤 Failid (Cloudflare R2)")
        uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
        if uploads:
            prefix = project_prefix(p["name"])
            for up in uploads:
                upload_file(s3, prefix, up)
            st.success(f"Üles laaditud {len(uploads)} faili.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📄 Projekti failid")
        prefix = project_prefix(p["name"])
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
