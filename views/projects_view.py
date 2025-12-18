# views/projects_view.py
import streamlit as st
from datetime import date

from db import list_projects, create_project, get_project, set_project_top_width
from r2 import (
    get_s3,
    project_prefix,
    ensure_project_marker,
    upload_file,
    download_bytes,
    list_files,
    delete_key,
)

from landxml import compute_cross_sections_every_5m


def _area_trapezoid(depth: float, b: float, n: float) -> float:
    # A = b*d + n*d^2 (trapeets: bottom=b, slopes 1:n)
    return float(b * depth + n * depth * depth)


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    if "project_id" not in st.session_state:
        st.session_state["project_id"] = None

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
                st.session_state["project_id"] = proj["id"]
                st.rerun()

    with right:
        pid = st.session_state.get("project_id")
        if not pid:
            st.info("Vali vasakult projekt.")
            return

        p = get_project(pid)
        if not p:
            st.session_state["project_id"] = None
            st.rerun()

        st.subheader(p["name"])
        st.caption(f"Tähtaeg: {p.get('end_date')}")

        # ---------------- LandXML upload (TOP WIDTH from cross-sections) ----------------
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📄 LandXML → pealmine laius (EG–EG)")

        landxml = st.file_uploader("Laadi üles LandXML (.xml)", type=["xml"], key="landxml_upload")

        with st.expander("Ristlõike seaded (valikuline)"):
            step_m = st.number_input("Samm (m) – iga X meetri tagant ristlõige", min_value=1.0, value=5.0, step=1.0)
            slice_thickness_m = st.number_input("Slice paksus (m)", min_value=0.2, value=1.0, step=0.1)

            cA, cB, cC, cD = st.columns(4)
            with cA:
                edge_band_pct = st.slider("Serva band %", 5, 25, 12, 1)
            with cB:
                edge_z_pct = st.slider("Serva Z %", 70, 99, 92, 1)
            with cC:
                bottom_z_pct = st.slider("Põhja Z %", 1, 20, 3, 1)
            with cD:
                min_pts = st.slider("Min punkte slice’is", 20, 300, 60, 10)

        if landxml and st.button("Salvesta LandXML & loe laius", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)

            xml_bytes = download_bytes(s3, key)

            res = compute_cross_sections_every_5m(
                xml_bytes,
                step_m=float(step_m),
                slice_thickness_m=float(slice_thickness_m),
                edge_band_pct=float(edge_band_pct),
                edge_z_pct=float(edge_z_pct),
                bottom_z_pct=float(bottom_z_pct),
                min_points_in_slice=int(min_pts),
            )

            if not res:
                # salvesta key DB-sse width None (või jäta vana alles)
                set_project_top_width(p["id"], key, None)
                st.warning("Ei suutnud TIN-ist ristlõikeid/pealmist laiust hinnata. Fail salvestati.")
                st.rerun()

            top_w = float(res.get("top_width_avg_m")) if res.get("top_width_avg_m") is not None else None
            valid = len(res.get("sections", []))
            axis_len = float(res.get("axis_length_m", 0.0))

            set_project_top_width(p["id"], key, top_w)

            if top_w is None:
                st.warning(f"Ristlõiked leiti ({valid}), aga pealmist laiust ei saanud usaldusväärselt arvutada.")
            else:
                st.success(f"Pealmine laius (EG–EG): ~ {top_w:.2f} m (valid slices: {valid})")
                st.caption(f"TIN telje pikkus (auto): ~ {axis_len:.2f} m")

            st.rerun()

        # Näita salvestatud tulemus
        if p.get("top_width_m") is not None:
            st.write(f"**TIN pealmine laius (EG–EG):** {float(p['top_width_m']):.2f} m")
        else:
            st.info("Pealmine laius puudub. Laadi LandXML üles.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Volume from user params ----------------
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("🧮 Maht sinu parameetritest")

        st.caption("Sisesta pikkus, põhi, nõlv ja sügavuse vahemik. Arvutame mahu (min/max/keskmine).")

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            L = st.number_input("Pikkus L (m)", min_value=0.0, value=50.0, step=1.0)
        with c2:
            b = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.4, step=0.1)
        with c3:
            slope_n = st.number_input("Nõlv 1:n (n)", min_value=0.1, value=2.0, step=0.1)
        with c4:
            d_min = st.number_input("Sügavus min (m)", min_value=0.0, value=1.5, step=0.1)

        d_max = st.number_input("Sügavus max (m)", min_value=0.0, value=2.5, step=0.1)

        if d_max < d_min:
            st.error("Sügavus max peab olema >= sügavus min.")
        else:
            A_min = _area_trapezoid(d_min, b, slope_n)
            A_max = _area_trapezoid(d_max, b, slope_n)
            A_avg = (A_min + A_max) / 2.0

            V_min = A_min * L
            V_max = A_max * L
            V_avg = A_avg * L

            st.write(f"**Ristlõige min:** {A_min:.2f} m²")
            st.write(f"**Ristlõige max:** {A_max:.2f} m²")
            st.write(f"**Ristlõige keskmine:** {A_avg:.2f} m²")

            st.write(f"**Maht min:** {V_min:.1f} m³")
            st.write(f"**Maht max:** {V_max:.1f} m³")
            st.write(f"**Maht keskmine:** {V_avg:.1f} m³")

            # top width from params
            T_min = b + 2.0 * slope_n * d_min
            T_max = b + 2.0 * slope_n * d_max
            st.caption(f"Sinu parameetritest pealmine laius: {T_min:.2f} – {T_max:.2f} m")

            # compare with TIN top width
            if p.get("top_width_m") is not None:
                tw = float(p["top_width_m"])
                if not (T_min <= tw <= T_max):
                    st.warning("TIN pealmine laius ei jää sinu parameetrite vahemikku. Kontrolli b/n/d või TIN servi.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- R2 file upload ----------------
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

        # ---------------- File list ----------------
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
