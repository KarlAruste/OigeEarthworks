# views/projects_view.py
import io
import pandas as pd
import streamlit as st

from db import init_db, create_project, list_projects, get_project
from landxml import (
    read_landxml_tin_from_bytes,
    build_tin_index,
    parse_alignment_from_bytes,
    compute_pk_table,
)


def _header(text: str):
    st.markdown(f"### {text}")

def _block_start():
    st.markdown('<div class="block">', unsafe_allow_html=True)

def _block_end():
    st.markdown("</div>", unsafe_allow_html=True)

def _float_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def render_projects_view():
    init_db()

    # --- Create project ---
    _block_start()
    _header("➕ Loo projekt")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pname = st.text_input("Projekti nimi", placeholder="nt Tapa_Objekt_01")
    with c2:
        sdate = st.date_input("Algus (valikuline)", value=None)
    with c3:
        edate = st.date_input("Lõpp (tähtaeg)", value=None)

    if st.button("Loo projekt", use_container_width=True):
        if not pname.strip():
            st.error("Sisesta projekti nimi.")
        else:
            create_project(pname.strip(), sdate, edate)
            st.success("Projekt loodud.")
            st.rerun()
    _block_end()

    st.markdown("")

    # --- Projects list ---
    projs = list_projects()
    _block_start()
    _header("📌 Minu projektid")
    if not projs:
        st.info("Projekte pole veel.")
        _block_end()
        return

    # pick active
    labels = [f"#{p['id']} — {p['name']}" for p in projs]
    ids = [int(p["id"]) for p in projs]
    cur_id = st.session_state.get("active_project_id")
    if cur_id not in ids:
        cur_id = ids[0]
        st.session_state["active_project_id"] = cur_id

    sel = st.selectbox("Vali projekt", options=ids, format_func=lambda pid: next((l for i, l in zip(ids, labels) if i == pid), str(pid)))
    st.session_state["active_project_id"] = int(sel)

    proj = get_project(int(sel))
    st.caption(f"Aktiivne projekt: **{proj['name']}** (ID {proj['id']})")

    _block_end()

    st.markdown("")

    tabs = st.tabs(["Üldine", "Mahud", "Failid"])
    # ----------------------------------------------------------
    # Üldine
    # ----------------------------------------------------------
    with tabs[0]:
        _block_start()
        _header("Üldinfo")
        st.write(f"**Projekt:** {proj['name']}")
        st.write(f"**Algus:** {proj.get('start_date')}")
        st.write(f"**Lõpp:** {proj.get('end_date')}")
        _block_end()

    # ----------------------------------------------------------
    # Mahud
    # ----------------------------------------------------------
    with tabs[1]:
        _block_start()
        _header("Mahud (LandXML pinnamudel + Civil3D Alignment)")

        st.markdown("**1) Pinnamudel (LandXML Surface / TIN)**")
        surf_file = st.file_uploader("Laadi pinnamudeli LandXML (.xml / .landxml)", type=["xml", "landxml"], key="surf_upl")
        if surf_file:
            surf_bytes = surf_file.read()
            st.session_state["surf_bytes"] = surf_bytes
            st.success(f"Pinnamudel laetud: {surf_file.name} ({len(surf_bytes)/1024:.1f} KB)")

        if "surf_bytes" not in st.session_state:
            st.warning("Lae pinnamudel (Surface LandXML), et saaks arvutada.")
            _block_end()
            return

        # Build TIN index once
        if st.session_state.get("tin_idx") is None or st.session_state.get("tin_idx_src") != hash(st.session_state["surf_bytes"]):
            pts, faces = read_landxml_tin_from_bytes(st.session_state["surf_bytes"])
            idx = build_tin_index(pts, faces)
            st.session_state["tin_idx"] = idx
            st.session_state["tin_idx_src"] = hash(st.session_state["surf_bytes"])
            st.caption(f"Punkte: {len(pts)} | Faces: {len(faces)}")

        st.markdown("---")
        st.markdown("**2) Telg / Alignment (LandXML Alignments)**")
        aln_file = st.file_uploader("Laadi telje LandXML (Alignment)", type=["xml", "landxml"], key="aln_upl")
        if aln_file:
            aln_bytes = aln_file.read()
            st.session_state["aln_bytes"] = aln_bytes
            st.success(f"Telg laetud: {aln_file.name} ({len(aln_bytes)/1024:.1f} KB)")

        if "aln_bytes" not in st.session_state:
            st.warning("Lae telje LandXML (Alignment), et saaks PK tabeli teha.")
            _block_end()
            return

        # parse alignment once
        if st.session_state.get("alignment") is None or st.session_state.get("alignment_src") != hash(st.session_state["aln_bytes"]):
            aln = parse_alignment_from_bytes(st.session_state["aln_bytes"])
            st.session_state["alignment"] = aln
            st.session_state["alignment_src"] = hash(st.session_state["aln_bytes"])

        aln = st.session_state["alignment"]
        st.info(f"Telg: **{aln['name']}** | Pikkus (Civil): **{aln['length']:.3f} m**")

        st.markdown("---")
        st.markdown("**3) Arvutusparameetrid**")

        c1, c2, c3 = st.columns(3)
        with c1:
            pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
            cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=5.0, value=17.0, step=1.0)
