# views/projects_view.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd

from db import (
    list_projects,
    create_project,
    get_project,
    set_project_landxml,
)
from r2 import (
    get_s3,
    project_prefix,
    ensure_project_marker,
    upload_file,
    download_bytes,
    list_files,
    delete_key,
)

from landxml import (
    read_landxml_tin_from_bytes,
    read_axis_from_bytes,
    compute_pk_table_from_landxml,
    polyline_length,
)


# -------------------------
# Small helpers
# -------------------------
def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


def _nice_num(x: float, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def _list_project_landxml_keys(s3, proj_name: str):
    """Return list of dicts from R2 landxml/ folder."""
    prefix = project_prefix(proj_name) + "landxml/"
    files = list_files(s3, prefix) or []
    # keep only xml/landxml
    out = []
    for f in files:
        n = (f.get("name") or "").lower()
        if n.endswith(".xml") or n.endswith(".landxml"):
            out.append(f)
    return out


def _list_project_axis_keys(s3, proj_name: str):
    """Return list of dicts from R2 axis/ folder."""
    prefix = project_prefix(proj_name) + "axis/"
    files = list_files(s3, prefix) or []
    out = []
    for f in files:
        n = (f.get("name") or "").lower()
        if n.endswith(".xml") or n.endswith(".landxml") or n.endswith(".csv") or n.endswith(".txt"):
            out.append(f)
    return out


# -------------------------
# Main view
# -------------------------
def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # Session defaults
    st.session_state.setdefault("active_project_id", None)

    # Mahud state
    st.session_state.setdefault("mahud_surface_key", None)
    st.session_state.setdefault("mahud_surface_bytes", None)

    st.session_state.setdefault("mahud_axis_key", None)
    st.session_state.setdefault("mahud_axis_xy", None)
    st.session_state.setdefault("mahud_axis_info", None)

    # ---------------- Create project ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Loo projekt")

    c1, c2, c3 = st.columns([2, 1, 1])
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

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Project list ----------------
    st.subheader("📌 Minu projektid")
    if not projects:
        st.info("Pole veel projekte.")
        return

    left, right = st.columns([1, 3], gap="large")

    with left:
        st.caption("Vali projekt:")
        for proj in projects:
            if st.button(proj["name"], use_container_width=True, key=f"projbtn_{proj['id']}"):
                st.session_state["active_project_id"] = proj["id"]

                # reset mahud view state on project switch
                st.session_state["mahud_surface_key"] = None
                st.session_state["mahud_surface_bytes"] = None
                st.session_state["mahud_axis_key"] = None
                st.session_state["mahud_axis_xy"] = None
                st.session_state["mahud_axis_info"] = None

                st.rerun()

    with right:
        pid = st.session_state.get("active_project_id")
        if not pid:
            st.info("Vali vasakult projekt.")
            return

        p = get_project(pid)
        if not p:
            st.session_state["active_project_id"] = None
            st.rerun()

        st.subheader(p["name"])
        st.caption(f"Tähtaeg: {p.get('end_date')}")

        tab_general, tab_mahud, tab_files = st.tabs(["Üldine", "Mahud", "Failid"])

        # ---------------- ÜLDINE ----------------
        with tab_general:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📊 Projekti salvestatud planeeritud väärtused (DB)")

            if p.get("planned_volume_m3") is None:
                st.info("Planeeritud mahtu pole veel arvutatud.")
            else:
                st.write(f"**Planned length:** {float(p.get('planned_length_m') or 0):.2f} m")
                st.write(f"**Planned area:** {float(p.get('planned_area_m2') or 0):.3f} m²")
                st.write(f"**Planned volume:** {float(p.get('planned_volume_m3') or 0):.3f} m³")
                st.caption(f"LandXML key: {p.get('landxml_key') or '-'}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- MAHUD ----------------
        with tab_mahud:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("🧮 Mahud (LandXML + Telg/Alignment import)")

            st.caption("Siin valid või laed pinnamudeli (LandXML) ja impordid telje (Civil3D polyline/alignment).")

            # ---- 1) Surface: choose existing from R2 ----
            st.markdown("### 1) Pinnamudel (LandXML)")

            landxml_files = _list_project_landxml_keys(s3, p["name"])
            options = ["— vali R2-st —"] + [f"{f['name']}  ({f['size']/1024:.1f} KB)" for f in landxml_files]
            sel = st.selectbox("Vali varem üles laetud LandXML", options, index=0)

            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("Ava valitud LandXML", use_container_width=True):
                    if sel.startswith("—"):
                        st.warning("Vali enne LandXML fail.")
                    else:
                        idx = options.index(sel) - 1
                        key = landxml_files[idx]["key"]
                        xml_bytes = download_bytes(s3, key)
                        st.session_state["mahud_surface_key"] = key
                        st.session_state["mahud_surface_bytes"] = xml_bytes
                        st.success(f"Avatud: {landxml_files[idx]['name']}")
                        st.rerun()

            with cB:
                up = st.file_uploader("...või lae uus LandXML", type=["xml", "landxml"], key="mahud_landxml_upload")
                if up and st.button("Salvesta uus LandXML R2-sse", use_container_width=True):
                    prefix = project_prefix(p["name"]) + "landxml/"
                    key = upload_file(s3, prefix, up)
                    st.success("Salvestatud. Vali see nüüd rippmenüüst ja vajuta 'Ava'.")
                    st.rerun()

            xml_bytes = st.session_state.get("mahud_surface_bytes")
            xml_key = st.session_state.get("mahud_surface_key")

            if xml_bytes:
                # show quick stats
                try:
                    pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)  # (E,N,Z)
                    xyz_show = _downsample_points(xyz, 20000)
                    st.caption(
                        f"LandXML punkte: {len(xyz):,} | Kuvan: {len(xyz_show):,} | "
                        f"E min/max: {_nice_num(np.min(xyz[:,0]))} / {_nice_num(np.max(xyz[:,0]))} | "
                        f"N min/max: {_nice_num(np.min(xyz[:,1]))} / {_nice_num(np.max(xyz[:,1]))}"
                    )
                except Exception as e:
                    st.error(f"LandXML lugemine ebaõnnestus: {e}")
                    st.stop()
            else:
                st.info("Vali või lae LandXML, et telge importida ja maht arvutada.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.divider()

            # ---- 2) Axis import ----
            st.markdown("### 2) Telg / Alignment (import Civil3D-st)")

            axis_files = _list_project_axis_keys(s3, p["name"])
            axis_opts = ["— vali R2-st —"] + [f"{f['name']}  ({f['size']/1024:.1f} KB)" for f in axis_files]
            axis_sel = st.selectbox("Vali varem üles laetud telg", axis_opts, index=0)

            col1, col2 = st.columns([1, 1])

            with col1:
                force_swap = st.checkbox("Telje koordinaadid on vahetuses (N=Easting ja E=Northing) → Swap E/N", value=True)

                if st.button("Ava valitud telg", use_container_width=True):
                    if axis_sel.startswith("—"):
                        st.warning("Vali enne telje fail.")
                    else:
                        i = axis_opts.index(axis_sel) - 1
                        key = axis_files[i]["key"]
                        b = download_bytes(s3, key)

                        info = read_axis_from_bytes(b, filename=axis_files[i]["name"], force_swap_en=force_swap)
                        st.session_state["mahud_axis_key"] = key
                        st.session_state["mahud_axis_xy"] = info["axis_xy"]
                        st.session_state["mahud_axis_info"] = info
                        st.success(f"Telg avatud: {axis_files[i]['name']}")
                        st.rerun()

            with col2:
                axis_up = st.file_uploader(
                    "Laadi telje fail (.xml/.landxml alignment või .csv/.txt punktid)",
                    type=["xml", "landxml", "csv", "txt"],
                    key="mahud_axis_upload",
                )
                if axis_up and st.button("Salvesta telg R2-sse", use_container_width=True):
                    prefix = project_prefix(p["name"]) + "axis/"
                    key = upload_file(s3, prefix, axis_up)
                    st.success("Telg salvestatud. Vali see rippmenüüst ja vajuta 'Ava'.")
                    st.rerun()

            axis_xy = st.session_state.get("mahud_axis_xy")
            axis_info = st.session_state.get("mahud_axis_info")

            if axis_xy and len(axis_xy) >= 2:
                declared = axis_info.get("declared_length_m") if isinstance(axis_info, dict) else None
                st.success(
                    f"Telg punkte: {len(axis_xy)} | "
                    f"Pikkus (arvutatud): {polyline_length(axis_xy):.2f} m"
                    + (f" | Alignment length attr: {declared:.2f} m" if isinstance(declared, (int, float)) else "")
                )

                with st.expander("Näita telje esimesed 10 punkti"):
                    df_axis = pd.DataFrame(axis_xy[:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)
            else:
                st.info("Impordi telg/alignment, et saaks PK tabeli arvutada.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.divider()

            # ---- 3) Parameters + compute ----
            st.markdown("### 3) Arvutusparameetrid")

            c1, c2, c3 = st.columns(3)
            with c1:
                pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
                cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=17.0, step=1.0)
                sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.1, step=0.01)
            with c2:
                tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005)
                min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.2, step=0.05)
                min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.3, step=0.05)
            with c3:
                slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2")
                bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05)

            if st.button("Arvuta PK tabel", use_container_width=True):
                try:
                    with st.spinner("Arvutan PK tabelit..."):
                        res = compute_pk_table_from_landxml(
                            xml_bytes=xml_bytes,
                            axis_xy_abs=axis_xy,
                            pk_step=float(pk_step),
                            cross_len=float(cross_len),
                            sample_step=float(sample_step),
                            tol=float(tol),
                            min_run=float(min_run),
                            min_depth_from_bottom=float(min_depth),
                            slope_text=slope,
                            bottom_w=float(bottom_w),
                        )

                    df = pd.DataFrame(res["rows"])
                    st.success(f"✅ Kokku maht: {res['total_volume_m3']:.3f} m³")
                    st.write(f"Telje pikkus: **{res['axis_length_m']:.2f} m** | PK-sid: **{res['count']}**")

                    planned_area = None
                    if res["axis_length_m"] > 0 and res["total_volume_m3"] is not None:
                        planned_area = float(res["total_volume_m3"] / res["axis_length_m"])

                    key_to_save = xml_key or (p.get("landxml_key") or "")
                    set_project_landxml(
                        p["id"],
                        landxml_key=key_to_save,
                        planned_volume_m3=float(res["total_volume_m3"]),
                        planned_length_m=float(res["axis_length_m"]),
                        planned_area_m2=planned_area,
                    )

                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False, sep=";").encode("utf-8")
                    st.download_button(
                        "⬇️ Lae alla CSV",
                        data=csv,
                        file_name=f"{p['name']}_pk_tabel.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.info("Tulemused salvestati projekti planned_* väljade alla (DB).")
                except Exception as e:
                    st.error(f"Arvutus ebaõnnestus: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- FAILID ----------------
        with tab_files:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📤 Failid (Cloudflare R2)")

            uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
            if uploads:
                prefix = project_prefix(p["name"])
                for up in uploads:
                    upload_file(s3, prefix, up)
                st.success(f"Üles laaditud {len(uploads)} faili.")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📄 Projekti failid")
            prefix = project_prefix(p["name"])
            files = list_files(s3, prefix)

            if not files:
                st.info("Selles projektis pole veel faile.")
                return

            for f in files:
                a, bcol, c = st.columns([6, 2, 2])
                with a:
                    st.write(f"📄 {f['name']}")
                    st.caption(f"{f['size'] / 1024:.1f} KB")
                with bcol:
                    data = download_bytes(s3, f["key"])
                    st.download_button(
                        "⬇️ Laadi alla",
                        data=data,
                        file_name=f["name"],
                        key=f"dl_{f['key']}",
                        use_container_width=True,
                    )
                with c:
                    if st.button("🗑️ Kustuta", key=f"del_{f['key']}", use_container_width=True):
                        delete_key(s3, f["key"])
                        st.rerun()
