# views/projects_view.py
import streamlit as st
from datetime import date
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
    compute_pk_table_from_landxml,
    read_axis_polyline_from_csv_bytes,
    read_axis_from_landxml_alignment_bytes,
    polyline_length,
)


def _list_landxml_keys_for_project(s3, proj_name: str):
    prefix = project_prefix(proj_name) + "landxml/"
    files = list_files(s3, prefix)
    # keep only xml/landxml
    out = []
    for f in files:
        n = (f.get("name") or "").lower()
        if n.endswith(".xml") or n.endswith(".landxml"):
            out.append(f)
    # newest first if list_files returns that way; if not, it’s ok.
    return out


def _axis_from_uploaded_file(file_name: str, data: bytes):
    low = (file_name or "").lower()
    if low.endswith(".csv") or low.endswith(".txt"):
        return read_axis_polyline_from_csv_bytes(data)
    if low.endswith(".xml") or low.endswith(".landxml"):
        # treat as Alignment LandXML
        return read_axis_from_landxml_alignment_bytes(data)
    raise ValueError("Telg: toeta ainult .csv/.txt (punktid E;N) või .xml/.landxml (Alignment).")


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # Session defaults
    st.session_state.setdefault("active_project_id", None)

    # Mahud tab state
    st.session_state.setdefault("selected_landxml_key", None)
    st.session_state.setdefault("axis_points_abs", None)  # [(E,N)]
    st.session_state.setdefault("axis_name", None)

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

                # reset per project
                st.session_state["selected_landxml_key"] = proj.get("landxml_key") or None
                st.session_state["axis_points_abs"] = None
                st.session_state["axis_name"] = None

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

        tab_general, tab_volumes, tab_files = st.tabs(["Üldine", "Mahud", "Failid"])

        # =========================
        # ÜLDINE
        # =========================
        with tab_general:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("ℹ️ Projekti info")

            st.write(f"**Projekt:** {p['name']}")
            st.write(f"**Algus:** {p.get('start_date')}")
            st.write(f"**Tähtaeg:** {p.get('end_date')}")

            if p.get("planned_volume_m3") is not None:
                st.markdown("### Salvestatud planeeritud väärtused (viimane arvutus)")
                st.write(f"**Planned length:** {float(p.get('planned_length_m') or 0):.2f} m")
                st.write(f"**Planned area:** {float(p.get('planned_area_m2') or 0):.3f} m²")
                st.write(f"**Planned volume:** {float(p.get('planned_volume_m3') or 0):.3f} m³")

            st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # MAHUD (LandXML + axis upload + compute)
        # =========================
        with tab_volumes:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📐 Mahud (LandXML + telg)")

            # ---- 1) LANDXML: upload + select existing ----
            st.markdown("#### 1) Vali pinnamudel (LandXML)")

            landxml_files = _list_landxml_keys_for_project(s3, p["name"])
            options = []
            key_by_label = {}

            # show existing in R2
            for f in landxml_files:
                label = f"{f['name']}  ({f['size']/1024:.1f} KB)"
                options.append(label)
                key_by_label[label] = f["key"]

            # include DB-saved key if not listed for some reason
            if p.get("landxml_key") and p["landxml_key"] not in (x["key"] for x in landxml_files):
                label = f"{p['landxml_key'].split('/')[-1]}  (DB)"
                options.insert(0, label)
                key_by_label[label] = p["landxml_key"]

            selected_label = None
            if options:
                # try set default
                default_key = st.session_state.get("selected_landxml_key") or p.get("landxml_key")
                default_idx = 0
                if default_key:
                    for i, lab in enumerate(options):
                        if key_by_label.get(lab) == default_key:
                            default_idx = i
                            break
                selected_label = st.selectbox("Üleslaetud pinnamudelid", options, index=default_idx)
                st.session_state["selected_landxml_key"] = key_by_label[selected_label]

            # upload new landxml
            new_landxml = st.file_uploader("Laadi uus LandXML (.xml/.landxml)", type=["xml", "landxml"], key="landxml_upload_volumes")
            if new_landxml and st.button("Salvesta LandXML R2-sse", use_container_width=True):
                prefix = project_prefix(p["name"]) + "landxml/"
                key = upload_file(s3, prefix, new_landxml)
                st.session_state["selected_landxml_key"] = key
                st.success("LandXML salvestatud. Vali see listist ja arvuta.")
                st.rerun()

            landxml_key = st.session_state.get("selected_landxml_key")
            if not landxml_key:
                st.info("Vali või lae LandXML pinnamudel.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # ---- 2) AXIS: upload polyline/alignment ----
            st.markdown("#### 2) Telg (Civil3D polyline / alignment)")

            st.caption(
                "Soovitus: ekspordi Civil3D-st telje punktid CSV-ks (E;N igal real). "
                "Või ekspordi alignment LandXML-ina (CoordGeom Line/Curve Start/End)."
            )

            axis_file = st.file_uploader(
                "Laadi teljefail (.csv/.txt punktidega või alignment .xml/.landxml)",
                type=["csv", "txt", "xml", "landxml"],
                key="axis_upload",
            )

            if axis_file and st.button("Impordi telg", use_container_width=True):
                data = axis_file.getvalue()
                axis_pts = _axis_from_uploaded_file(axis_file.name, data)
                st.session_state["axis_points_abs"] = axis_pts
                st.session_state["axis_name"] = axis_file.name
                st.success(f"Telg imporditud: {len(axis_pts)} punkti, pikkus {polyline_length(axis_pts):.2f} m")
                st.rerun()

            axis_pts = st.session_state.get("axis_points_abs")
            if axis_pts:
                st.write(f"**Telg:** {st.session_state.get('axis_name')} | Punkte: {len(axis_pts)} | Pikkus: {polyline_length(axis_pts):.2f} m")
                with st.expander("Näita telje esimesed 10 punkti"):
                    df_axis = pd.DataFrame(axis_pts[:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)
            else:
                st.warning("Impordi telg enne arvutamist.")

            # ---- 3) Compute parameters ----
            st.markdown("#### 3) Arvutusparameetrid")
            c1, c2, c3 = st.columns(3)
            with c1:
                pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
                cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=25.0, step=1.0)
                sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.1, step=0.01)
            with c2:
                tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005)
                min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.2, step=0.05)
                min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.3, step=0.05)
            with c3:
                slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2")
                bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05)

            # ---- 4) Compute ----
            if st.button("Arvuta PK tabel (LandXML + telg)", use_container_width=True):
                if not axis_pts or len(axis_pts) < 2:
                    st.error("Telg puudub või liiga lühike.")
                else:
                    xml_bytes = download_bytes(s3, landxml_key)

                    res = compute_pk_table_from_landxml(
                        xml_bytes=xml_bytes,
                        axis_xy_abs=axis_pts,
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

                    # Save to DB (and save selected landxml key too)
                    set_project_landxml(
                        p["id"],
                        landxml_key=landxml_key,
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

                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # FAILID (R2 upload/list)
        # =========================
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
