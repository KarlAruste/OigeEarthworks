# views/projects_view.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from db import list_projects, create_project, get_project, set_project_landxml
from r2 import (
    get_s3, project_prefix, ensure_project_marker,
    upload_file, download_bytes, list_files, delete_key
)

from streamlit_plotly_events import plotly_events

from landxml import (
    read_landxml_tin_from_bytes,
    build_tin_index,
    nearest_point_xyz,
    polyline_length,
    compute_pk_table_from_landxml,
)


def _downsample_xy(xy: np.ndarray, max_pts: int = 60000) -> np.ndarray:
    if xy.shape[0] <= max_pts:
        return xy
    idx = np.random.choice(xy.shape[0], size=max_pts, replace=False)
    return xy[idx]


def _make_local_coords(xy_abs: np.ndarray):
    # local origin = min corner (stabiilne)
    x0 = float(np.min(xy_abs[:, 0]))
    y0 = float(np.min(xy_abs[:, 1]))
    xy_local = xy_abs.copy()
    xy_local[:, 0] -= x0
    xy_local[:, 1] -= y0
    return (x0, y0), xy_local


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # session
    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)

    st.session_state.setdefault("tin_pts_abs", None)   # Nx2 abs (E,N)
    st.session_state.setdefault("tin_origin", None)    # (E0,N0)
    st.session_state.setdefault("tin_pts_local", None) # Nx2 local

    st.session_state.setdefault("tin_index", None)     # TinIndex (for snapping)

    st.session_state.setdefault("axis_abs", [])        # [(E,N), ...]
    st.session_state.setdefault("axis_finished", False)

    # ---------------- Create project ----------------
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

    st.divider()

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
                st.session_state["axis_abs"] = []
                st.session_state["axis_finished"] = False
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["tin_pts_abs"] = None
                st.session_state["tin_pts_local"] = None
                st.session_state["tin_origin"] = None
                st.session_state["tin_index"] = None
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

        # ---------------- LandXML upload ----------------
        st.subheader("📄 LandXML → salvesta ja ava mudel")
        landxml = st.file_uploader("Laadi üles LandXML (.xml/.landxml)", type=["xml", "landxml"], key="landxml_upload")

        if landxml and st.button("Salvesta LandXML (R2) & lae mudel", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)

            pts, faces = read_landxml_tin_from_bytes(xml_bytes)
            xyz = np.array(list(pts.values()), dtype=float)  # (E,N,Z)

            xy_abs = xyz[:, :2]
            origin, xy_local = _make_local_coords(xy_abs)

            st.session_state["landxml_bytes"] = xml_bytes
            st.session_state["landxml_key"] = key
            st.session_state["tin_pts_abs"] = xy_abs
            st.session_state["tin_origin"] = origin
            st.session_state["tin_pts_local"] = xy_local

            st.session_state["tin_index"] = build_tin_index(pts, faces)

            st.session_state["axis_abs"] = []
            st.session_state["axis_finished"] = False

            st.success(f"LandXML laaditud. Punkte: {xy_abs.shape[0]}")
            st.rerun()

        st.divider()

        # ---------------- Axis drawing ----------------
        st.subheader("🧭 Telje joonistamine (polyline) + PK tabel")

        xml_bytes = st.session_state.get("landxml_bytes")
        xy_local = st.session_state.get("tin_pts_local")
        origin = st.session_state.get("tin_origin")
        tin_index = st.session_state.get("tin_index")

        if xml_bytes is None or xy_local is None or origin is None or tin_index is None:
            st.info("Lae LandXML üles, et telge joonistada.")
        else:
            E0, N0 = origin

            # controls
            cA, cB, cC, cD = st.columns([1, 1, 1, 2])
            with cA:
                if st.button("Alusta / joonista telg", use_container_width=True):
                    st.session_state["axis_abs"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with cB:
                if st.button("Undo (eemalda viimane)", use_container_width=True):
                    if st.session_state["axis_abs"]:
                        st.session_state["axis_abs"] = st.session_state["axis_abs"][:-1]
                        st.session_state["axis_finished"] = False
                        st.rerun()
            with cC:
                if st.button("Tühjenda telg", use_container_width=True):
                    st.session_state["axis_abs"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with cD:
                if st.button("Lõpeta telg", use_container_width=True):
                    if len(st.session_state["axis_abs"]) < 2:
                        st.warning("Telg peab olema vähemalt 2 punktiga.")
                    else:
                        st.session_state["axis_finished"] = True
                        st.success("Telg lõpetatud.")
                        st.rerun()

            axis_abs = st.session_state["axis_abs"]
            axis_finished = st.session_state["axis_finished"]

            # plot points
            xy_show = _downsample_xy(xy_local, max_pts=60000)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=xy_show[:, 0], y=xy_show[:, 1],
                mode="markers",
                marker=dict(size=3, opacity=0.6),
                name="TIN punktid",
                hoverinfo="skip"
            ))

            # axis overlay (local coords)
            if axis_abs:
                axis_local = [(E - E0, N - N0) for (E, N) in axis_abs]
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in axis_local],
                    y=[p[1] for p in axis_local],
                    mode="lines+markers",
                    line=dict(width=4),
                    marker=dict(size=8),
                    name="Telg"
                ))
                fig.add_trace(go.Scatter(
                    x=[axis_local[-1][0]],
                    y=[axis_local[-1][1]],
                    mode="markers",
                    marker=dict(size=10),
                    name="Viimane"
                ))

            fig.update_layout(
                height=650,
                title="Pealtvaade (kliki telje punktide lisamiseks)",
                margin=dict(l=10, r=10, t=40, b=10),
                dragmode="pan",
                uirevision="keep",
                legend=dict(orientation="h"),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            st.caption("👉 Kliki kuhu tahes – lisame teljele lähima TIN punkti (snap).")

            click_data = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=650,
                key="tin_clicks",
            )

            # handle click -> snap -> add point
            if click_data and (not axis_finished):
                lx = float(click_data[0]["x"])  # local
                ly = float(click_data[0]["y"])

                # convert to abs
                px = E0 + lx
                py = N0 + ly

                snapped = nearest_point_xyz(tin_index, px, py, max_rings=10)
                if snapped is not None:
                    sx, sy, _sz = snapped
                    st.session_state["axis_abs"] = axis_abs + [(float(sx), float(sy))]
                    st.rerun()
                else:
                    st.warning("Snap ebaõnnestus (ei leidnud lähedal punkte).")

            if axis_abs:
                L = polyline_length(axis_abs)
                st.write(f"**Telje pikkus:** {L:.2f} m  |  Punkte: {len(axis_abs)}")

            # ---------------- PK compute ----------------
            st.markdown("### Arvutusparameetrid")
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

            if st.button("Arvuta PK tabel (telje järgi)", use_container_width=True):
                if len(axis_abs) < 2:
                    st.warning("Joonista telg enne arvutamist.")
                else:
                    res = compute_pk_table_from_landxml(
                        xml_bytes=xml_bytes,
                        axis_xy_abs=axis_abs,
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
                    if res["axis_length_m"] > 0:
                        planned_area = float(res["total_volume_m3"] / res["axis_length_m"])

                    key = st.session_state["landxml_key"] or p.get("landxml_key") or ""
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
                        use_container_width=True
                    )

        st.divider()

        # ---------------- R2 file upload ----------------
        st.subheader("📤 Failid (Cloudflare R2)")
        uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
        if uploads:
            prefix = project_prefix(p["name"])
            for up in uploads:
                upload_file(s3, prefix, up)
            st.success(f"Üles laaditud {len(uploads)} faili.")
            st.rerun()

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
