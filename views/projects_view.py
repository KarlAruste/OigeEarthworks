# views/projects_view.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from streamlit_plotly_events import plotly_events

from db import list_projects, create_project, get_project, set_project_landxml
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
    build_tin_index,
    snap_xy_to_tin,
    polyline_length,
    compute_pk_table_from_landxml,
)


def _downsample(xy: np.ndarray, max_pts: int = 60000) -> np.ndarray:
    if xy is None or xy.size == 0:
        return xy
    if xy.shape[0] <= max_pts:
        return xy
    idx = np.random.choice(xy.shape[0], size=max_pts, replace=False)
    return xy[idx]


def _make_fig(points_local: np.ndarray, axis_local: list, uirev: str):
    fig = go.Figure()

    if points_local is None or len(points_local) == 0:
        fig.update_layout(
            height=650,
            title="TIN punkte ei leitud (kontrolli LandXML)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    xmin = float(np.min(points_local[:, 0]))
    xmax = float(np.max(points_local[:, 0]))
    ymin = float(np.min(points_local[:, 1]))
    ymax = float(np.max(points_local[:, 1]))

    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    pad_x = dx * 0.08
    pad_y = dy * 0.08

    fig.add_trace(
        go.Scattergl(
            x=points_local[:, 0].astype(float),
            y=points_local[:, 1].astype(float),
            mode="markers",
            marker=dict(size=4, opacity=0.75),
            name="TIN punktid",
            hoverinfo="skip",
        )
    )

    if axis_local and len(axis_local) >= 1:
        xs = [p[0] for p in axis_local]
        ys = [p[1] for p in axis_local]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(width=4),
                marker=dict(size=10),
                name="Telg",
            )
        )

    fig.update_layout(
        height=650,
        title="Pealtvaade (kohalikud koordinaadid) – kliki telje punktide lisamiseks",
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
        legend=dict(orientation="h"),
        uirevision=uirev,  # hoiab zoomi/pani
    )

    fig.update_xaxes(
        title="E (local, m)",
        range=[xmin - pad_x, xmax + pad_x],
        tickformat=".0f",
    )
    fig.update_yaxes(
        title="N (local, m)",
        range=[ymin - pad_y, ymax + pad_y],
        tickformat=".0f",
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def render_projects_view():
    st.title("Projects")
    s3 = get_s3()
    projects = list_projects()

    # ---------------- session state ----------------
    st.session_state.setdefault("active_project_id", None)

    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)

    st.session_state.setdefault("tin_pts", None)      # dict id->(E,N,Z)
    st.session_state.setdefault("tin_faces", None)
    st.session_state.setdefault("tin_index", None)    # TinIndex
    st.session_state.setdefault("origin_EN", None)    # (E0,N0)

    st.session_state.setdefault("axis_abs", [])       # ABS [(E,N)]
    st.session_state.setdefault("axis_finished", False)
    st.session_state.setdefault("axis_drawing", False)

    st.session_state.setdefault("plot_uirev", "init")
    st.session_state.setdefault("last_click_sig", None)  # debounce

    debug = st.checkbox("DEBUG", value=False)

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

                # reset when switching
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["tin_pts"] = None
                st.session_state["tin_faces"] = None
                st.session_state["tin_index"] = None
                st.session_state["origin_EN"] = None
                st.session_state["axis_abs"] = []
                st.session_state["axis_finished"] = False
                st.session_state["axis_drawing"] = False
                st.session_state["plot_uirev"] = f"init-{proj['id']}"
                st.session_state["last_click_sig"] = None
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
        st.markdown("### 📄 LandXML")
        landxml = st.file_uploader(
            "Laadi üles LandXML (.xml/.landxml)",
            type=["xml", "landxml"],
            key="landxml_upload",
        )

        if landxml and st.button("Salvesta LandXML (R2) & lae mudel", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)

            pts, faces = read_landxml_tin_from_bytes(xml_bytes)
            idx = build_tin_index(pts, faces)

            xyz = idx.pts_xyz  # (E,N,Z)
            E0 = float(np.median(xyz[:, 0]))
            N0 = float(np.median(xyz[:, 1]))

            st.session_state["landxml_bytes"] = xml_bytes
            st.session_state["landxml_key"] = key
            st.session_state["tin_pts"] = pts
            st.session_state["tin_faces"] = faces
            st.session_state["tin_index"] = idx
            st.session_state["origin_EN"] = (E0, N0)

            st.session_state["axis_abs"] = []
            st.session_state["axis_finished"] = False
            st.session_state["axis_drawing"] = False
            st.session_state["last_click_sig"] = None

            st.session_state["plot_uirev"] = f"model-{pid}-{key}"

            st.success(f"LandXML laetud. Punkte: {len(pts):,} | Faces: {len(faces):,}")
            st.rerun()

        if not st.session_state["tin_index"]:
            st.info("Lae LandXML, et näha punkte ja joonistada telg.")
            return

        idx = st.session_state["tin_index"]
        E0, N0 = st.session_state["origin_EN"]

        xyz = idx.pts_xyz
        xy_local = np.column_stack([xyz[:, 0] - E0, xyz[:, 1] - N0])
        xy_local_show = _downsample(xy_local, max_pts=60000)

        axis_abs = st.session_state["axis_abs"]
        axis_local = [(E - E0, N - N0) for (E, N) in axis_abs]

        st.caption(
            f"ABS E min/max: {xyz[:,0].min():.3f} / {xyz[:,0].max():.3f} | "
            f"ABS N min/max: {xyz[:,1].min():.3f} / {xyz[:,1].max():.3f}"
        )

        # ---------------- Axis UI ----------------
        st.markdown("### 🧭 Telje joonistamine (kliki pildil)")
        cA, cB, cC, cD = st.columns([1, 1, 1, 2])

        with cA:
            if st.button("Alusta / joonista telg", use_container_width=True):
                st.session_state["axis_abs"] = []
                st.session_state["axis_finished"] = False
                st.session_state["axis_drawing"] = True
                st.session_state["last_click_sig"] = None
                st.rerun()

        with cB:
            if st.button("Undo", use_container_width=True):
                if st.session_state["axis_abs"]:
                    st.session_state["axis_abs"] = st.session_state["axis_abs"][:-1]
                    st.session_state["axis_finished"] = False
                    st.rerun()

        with cC:
            if st.button("Tühjenda telg", use_container_width=True):
                st.session_state["axis_abs"] = []
                st.session_state["axis_finished"] = False
                st.session_state["axis_drawing"] = True
                st.session_state["last_click_sig"] = None
                st.rerun()

        with cD:
            if st.button("Lõpeta telg", use_container_width=True):
                if len(st.session_state["axis_abs"]) < 2:
                    st.warning("Telg peab olema vähemalt 2 punktiga.")
                else:
                    st.session_state["axis_finished"] = True
                    st.session_state["axis_drawing"] = False
                    st.success("Telg lõpetatud.")
                    st.rerun()

        drawing = st.session_state["axis_drawing"]
        finished = st.session_state["axis_finished"]

        snap_on = st.checkbox("Snap telje punktid TIN-i lähimale tipule", value=True)
        st.write(f"**Telje režiim:** {'✅ ON' if drawing else '❌ OFF'} | **Telg lõpetatud:** {finished}")

        # ---------------- Plot + click capture ----------------
        fig = _make_fig(
            points_local=xy_local_show,
            axis_local=axis_local,
            uirev=st.session_state["plot_uirev"],
        )

        click_data = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=650,
            key="tin_plot_events",
        )

        if debug:
            st.write("DEBUG click_data:", click_data)

        # add point only when drawing is ON and not finished
        if drawing and (not finished) and click_data:
            x_local = float(click_data[0]["x"])
            y_local = float(click_data[0]["y"])

            sig = f"{x_local:.4f}:{y_local:.4f}"
            if st.session_state.get("last_click_sig") != sig:
                st.session_state["last_click_sig"] = sig

                E_click = x_local + E0
                N_click = y_local + N0

                if snap_on:
                    E_snap, N_snap, _Z = snap_xy_to_tin(idx, E_click, N_click)
                    st.session_state["axis_abs"] = st.session_state["axis_abs"] + [(float(E_snap), float(N_snap))]
                else:
                    st.session_state["axis_abs"] = st.session_state["axis_abs"] + [(float(E_click), float(N_click))]

                st.rerun()

        if st.session_state["axis_abs"]:
            L = polyline_length(st.session_state["axis_abs"])
            st.write(f"**Telje pikkus:** {L:.2f} m | Punkte: {len(st.session_state['axis_abs'])}")

        # ---------------- PK parameters + compute ----------------
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
            if len(st.session_state["axis_abs"]) < 2:
                st.warning("Joonista telg enne arvutamist.")
            else:
                res = compute_pk_table_from_landxml(
                    xml_bytes=st.session_state["landxml_bytes"],
                    axis_xy_abs=st.session_state["axis_abs"],
                    pk_step=float(pk_step),
                    cross_len=float(cross_len),
                    sample_step=float(sample_step),
                    tol=float(tol),
                    min_run=float(min_run),
                    min_depth_from_bottom=float(min_depth),
                    slope_text=str(slope),
                    bottom_w=float(bottom_w),
                )

                df = pd.DataFrame(res["rows"])
                st.success(f"✅ Kokku maht: {res['total_volume_m3']:.3f} m³")
                st.write(f"Telje pikkus: **{res['axis_length_m']:.2f} m** | PK-sid: **{res['count']}**")

                planned_area = None
                if res["axis_length_m"] and res["axis_length_m"] > 0 and res["total_volume_m3"] is not None:
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
                    use_container_width=True,
                )

        st.divider()

        # ---------------- R2 files ----------------
        st.subheader("📤 Failid (Cloudflare R2)")
        uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
        if uploads:
            prefix = project_prefix(p["name"])
            for up in uploads:
                upload_file(s3, prefix, up)
            st.success(f"Üles laaditud {len(uploads)} faili.")
            st.rerun()

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
