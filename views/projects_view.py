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
    compute_pk_table_from_landxml,
    parse_axis_points_from_bytes,
    polyline_length,
)


# -------------------------
# small helpers
# -------------------------
def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


def _list_landxml_keys_for_project(s3, project_name: str) -> list[dict]:
    """
    Returns list of objects under landxml/ prefix.
    Each dict: {name,key,size}
    """
    prefix = project_prefix(project_name) + "landxml/"
    files = list_files(s3, prefix)
    # keep only xml/landxml
    out = []
    for f in files:
        nm = (f.get("name") or "").lower()
        if nm.endswith(".xml") or nm.endswith(".landxml"):
            out.append(f)
    # newest first (list_files already likely sorted, but keep stable)
    return out


# -------------------------
# Main view
# -------------------------
def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # session defaults
    st.session_state.setdefault("active_project_id", None)

    # Mahud: cached selected surface and axis
    st.session_state.setdefault("mahud_surface_key", None)
    st.session_state.setdefault("mahud_surface_bytes", None)
    st.session_state.setdefault("mahud_axis_name", None)
    st.session_state.setdefault("mahud_axis_xy", None)  # list[(E,N)]
    st.session_state.setdefault("mahud_swap_axis_en", False)

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

                # reset Mahud state when switching project
                st.session_state["mahud_surface_key"] = None
                st.session_state["mahud_surface_bytes"] = None
                st.session_state["mahud_axis_name"] = None
                st.session_state["mahud_axis_xy"] = None
                st.session_state["mahud_swap_axis_en"] = False

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

        tabs = st.tabs(["Üldine", "Mahud", "Failid"])

        # =========================================================
        # TAB 1: ÜLDINE
        # =========================================================
        with tabs[0]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📌 Projekti info")

            p2 = get_project(pid)
            if p2 and p2.get("planned_volume_m3") is not None:
                st.markdown("### Salvestatud planeeritud väärtused (Mahud)")
                st.write(f"**Planned length:** {float(p2['planned_length_m'] or 0):.2f} m")
                st.write(f"**Planned area:** {float(p2['planned_area_m2'] or 0):.3f} m²")
                st.write(f"**Planned volume:** {float(p2['planned_volume_m3'] or 0):.3f} m³")
                if p2.get("landxml_key"):
                    st.caption(f"LandXML key: {p2.get('landxml_key')}")
            else:
                st.info("Mahte pole veel arvutatud. Mine tab’i **Mahud**.")

            st.markdown("</div>", unsafe_allow_html=True)

        # =========================================================
        # TAB 2: MAHUD
        # =========================================================
        with tabs[1]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("🧮 Mahud (LandXML + telg Civil3D-st)")

            st.caption(
                "1) Vali või laadi üles LandXML pind (R2). "
                "2) Impordi telg (Civil3D Alignment/Polyline LandXML või CSV/TXT). "
                "3) Arvuta PK tabel ja kogu maht."
            )

            # ---------------------------
            # 1) Surface: upload + select existing
            # ---------------------------
            st.markdown("### 1) LandXML pind (R2)")

            landxml_files = _list_landxml_keys_for_project(s3, p["name"])
            existing_names = [f["name"] for f in landxml_files]

            cA, cB = st.columns([2, 2])
            with cA:
                chosen = st.selectbox(
                    "Vali varem üles laaditud LandXML",
                    options=["— vali —"] + existing_names,
                    index=0,
                )
            with cB:
                up = st.file_uploader(
                    "Või laadi uus LandXML üles",
                    type=["xml", "landxml"],
                    key="mahud_landxml_upload",
                )

            if up is not None:
                if st.button("Salvesta LandXML R2-sse", use_container_width=True):
                    prefix = project_prefix(p["name"]) + "landxml/"
                    key = upload_file(s3, prefix, up)
                    st.success("LandXML salvestatud R2-sse.")
                    # refresh select
                    st.session_state["mahud_surface_key"] = key
                    st.session_state["mahud_surface_bytes"] = download_bytes(s3, key)
                    st.rerun()

            # if choose existing
            if chosen and chosen != "— vali —":
                key = next((f["key"] for f in landxml_files if f["name"] == chosen), None)
                if key and key != st.session_state.get("mahud_surface_key"):
                    st.session_state["mahud_surface_key"] = key
                    st.session_state["mahud_surface_bytes"] = download_bytes(s3, key)

            xml_bytes = st.session_state.get("mahud_surface_bytes")
            if xml_bytes:
                # show quick stats
                try:
                    pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)  # (E,N,Z)
                    xyz_show = _downsample_points(xyz, max_pts=20000)
                    x_min = float(np.nanmin(xyz_show[:, 0])); x_max = float(np.nanmax(xyz_show[:, 0]))
                    y_min = float(np.nanmin(xyz_show[:, 1])); y_max = float(np.nanmax(xyz_show[:, 1]))
                    st.caption(f"LandXML punktid: {len(xyz)} | E min/max: {x_min:.3f} / {x_max:.3f} | N min/max: {y_min:.3f} / {y_max:.3f}")
                except Exception as e:
                    st.error(f"LandXML lugemine ebaõnnestus: {e}")
                    st.stop()
            else:
                st.info("Vali või lae LandXML, et jätkata.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.divider()

            # ---------------------------
            # 2) Axis import
            # ---------------------------
            st.markdown("### 2) Telg (Civil3D alignment/polyline)")

            swap_axis = st.checkbox(
                "Telje koordinaadid on vahetuses (N=Easting ja E=Northing) → Swap E/N",
                value=bool(st.session_state.get("mahud_swap_axis_en", False)),
            )
            st.session_state["mahud_swap_axis_en"] = bool(swap_axis)

            axis_file = st.file_uploader(
                "Laadi telje fail (.xml/.landxml alignment või .csv/.txt punktid)",
                type=["xml", "landxml", "csv", "txt"],
                key="mahud_axis_upload",
            )

            if axis_file is not None:
                if st.button("Impordi telg", use_container_width=True):
                    try:
                        axis_bytes = axis_file.getvalue()
                        axis_xy = parse_axis_points_from_bytes(
                            axis_bytes,
                            filename=axis_file.name,
                            force_swap_en=swap_axis,
                        )
                        st.session_state["mahud_axis_name"] = axis_file.name
                        st.session_state["mahud_axis_xy"] = axis_xy
                        st.success(f"Telg imporditud: {axis_file.name} | Punkte: {len(axis_xy)} | Pikkus: {polyline_length(axis_xy):.2f} m")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            axis_xy_abs = st.session_state.get("mahud_axis_xy")
            if axis_xy_abs and len(axis_xy_abs) >= 2:
                st.caption(f"Telg: {st.session_state.get('mahud_axis_name')} | Punkte: {len(axis_xy_abs)} | Pikkus: {polyline_length(axis_xy_abs):.2f} m")
                df_axis = pd.DataFrame(axis_xy_abs, columns=["E", "N"])
                with st.expander("Näita telje esimesed 10 punkti"):
                    st.dataframe(df_axis.head(10), use_container_width=True)
            else:
                st.info("Impordi telg, et arvutada mahud.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.divider()

            # ---------------------------
            # 3) Parameters + compute
            # ---------------------------
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
                    with st.spinner("Arvutan PK-d ja mahtu..."):
                        res = compute_pk_table_from_landxml(
                            xml_bytes=xml_bytes,
                            axis_xy_abs=axis_xy_abs,
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

                    key = st.session_state.get("mahud_surface_key") or (p.get("landxml_key") or "")
                    set_project_landxml(
                        p["id"],
                        landxml_key=key,
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

        # =========================================================
        # TAB 3: FAILID (R2)
        # =========================================================
        with tabs[2]:
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
