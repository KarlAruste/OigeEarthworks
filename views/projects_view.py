# views/projects_view.py
import streamlit as st
from datetime import date

from db import list_projects, create_project, get_project, set_project_landxml
from r2 import (
    get_s3,
    project_prefix,
    ensure_project_marker,
    list_files,
    upload_file,
    download_bytes,
    delete_key,
)
from landxml import estimate_length_area_volume_from_tin


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    if "project_id" not in st.session_state:
        st.session_state["project_id"] = None

    # --- Create project ---
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

        # Planned geometry summary
        length = p.get("planned_length_m")
        area = p.get("planned_area_m2")
        vol = p.get("planned_volume_m3")

        if length is not None:
            st.write(f"**Pikkus:** {float(length):.2f} m")
        if area is not None:
            st.write(f"**Keskmine ristlõige:** {float(area):.2f} m²")
        if vol is not None:
            st.write(f"**Planeeritud maht:** {float(vol):.1f} m³")

        if vol is None:
            st.info("Planeeritud maht puudub. Laadi LandXML üles.")

        # --- LandXML upload ---
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📄 LandXML (TIN -> pikkus/ristlõige/maht)")

        landxml = st.file_uploader(
            "Laadi üles LandXML (.xml)",
            type=["xml"],
            key="landxml_upload",
            help="Lae üles kraavi TIN LandXML. Rakendus proovib leida servad ja arvutada ristlõike."
        )

        with st.expander("Arvutuse seaded (valikuline)"):
            n_bins = st.slider("Lõigete arv", min_value=10, max_value=120, value=30, step=5)
            edge_tail = st.slider("Serva tail %", min_value=5, max_value=25, value=10, step=1)
            slice_ratio = st.slider("Slice paksus (ratio %)", min_value=1, max_value=10, value=3, step=1) / 100.0
            edge_z_pct = st.slider("Serva Z percentiil (kõrgus)", 90, 99, 97, 1)
            fallback_top = st.slider("Fallback top %", 85, 99, 95, 1)

        if landxml and st.button("Salvesta & arvuta", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)

            xml_bytes = download_bytes(s3, key)

            length_m, area_m2, vol_m3 = estimate_length_area_volume_from_tin(
            xml_bytes,
            n_bins=int(n_bins),
            slice_thickness_ratio=float(slice_ratio),
            edge_tail_pct=float(edge_tail),
            edge_z_pct=float(edge_z_pct),
            top_percentile_fallback=float(fallback_top),
         )



            set_project_landxml(p["id"], key, vol_m3, length_m, area_m2)

            if vol_m3 is None or area_m2 is None:
    if length_m is not None:
        st.warning(
            f"TIN-ist ei suutnud kindlalt ristlõiget/mahtu hinnata. "
            f"Pikkus ~ {length_m:.1f} m. "
            "See tähendab tavaliselt, et servad pole selles pinnas "
            "või TIN on ainult kraavi põhi."
        )
    else:
        st.warning(
            "TIN-ist ei suutnud kindlalt ristlõiget/mahtu hinnata. "
            "Fail salvestati."
        )
else:
    st.success(
        f"Leitud: pikkus ~ {length_m:.2f} m, "
        f"ristlõige ~ {area_m2:.2f} m², "
        f"maht ~ {vol_m3:.1f} m³"
    )

        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown("</div>", unsafe_allow_html=True)

        # --- File list ---
        st.subheader("📄 Projekti failid")
        prefix = project_prefix(p["name"])
        files = list_files(s3, prefix)
        if not files:
            st.info("Selles projektis pole veel faile.")
            return

        for f in files:
            a, b, c = st.columns([6, 2, 2])
            with a:
                st.write(f"📄 {f['name']}")
                st.caption(f"{f['size'] / 1024:.1f} KB")
            with b:
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
