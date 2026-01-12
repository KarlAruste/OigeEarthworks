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
    compute_local_origin_EN, to_abs_EN, to_local_EN,
    build_tin_index, nearest_point_xyz,
    polyline_length,
    compute_pk_table_from_landxml,
)


def _downsample(xyz: np.ndarray, max_pts: int = 70000) -> np.ndarray:
    if xyz.shape[0] <= max_pts:
        return xyz
    idx = np.random.choice(xyz.shape[0], size=max_pts, replace=False)
    return xyz[idx]


def _make_figure(local_pts_EN: np.ndarray, axis_local: list[tuple[float, float]]):
    """
    local_pts_EN: Nx2 array (E_local, N_local)
    axis_local: list of (E_local, N_local)
    """
    fig = go.Figure()

    if local_pts_EN is None or local_pts_EN.size == 0:
        fig.update_layout(height=650, title="Punkte ei ole", margin=dict(l=10, r=10, t=40, b=10))
        return fig

    x = local_pts_EN[:, 0]
    y = local_pts_EN[:, 1]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = (xmax - xmin) if (xmax > xmin) else 1.0
    dy = (ymax - ymin) if (ymax > ymin) else 1.0
    pad_x = dx * 0.05
    pad_y = dy * 0.05

    # 1) TIN points
    fig.add_trace(go.Scattergl(
        x=x, y=y,
        mode="markers",
        marker=dict(size=3, opacity=0.6),
        name="TIN",
        hoverinfo="skip",
    ))

    # 2) Click-catcher grid (so click works even on empty space)
    gx = np.linspace(xmin, xmax, 60)
    gy = np.linspace(ymin, ymax, 40)
    grid = np.array([(xx, yy) for xx in gx for yy in gy], dtype=float)

    fig.add_trace(go.Scatter(
        x=grid[:, 0],
        y=grid[:, 1],
        mode="markers",
        marker=dict(size=25, opacity=0.0),
        name="__click__",
        hoverinfo="skip",
        showlegend=False
    ))

    # 3) Axis
    if axis_local:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in axis_local],
            y=[p[1] for p in axis_local],
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=9),
            name="Telg",
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Pealtvaade (kohalikud koordinaadid) – kliki telje punktide lisamiseks",
        dragmode="pan",
        uirevision="keep",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title="E_local (m)", range=[xmin - pad_x, xmax + pad_x], zeroline=False)
    fig.update_yaxes(title="N_local (m)", range=[ymin - pad_y, ymax + pad_y], scaleanchor="x", scaleratio=1, zeroline=False)
    return fig


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # session
    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)

    st.session_state.setdefault("E0", None)
    st.session_state.setdefault("N0", None)

    st.session_state.setdefault("xyz_abs", None)        # Nx3 (E,N,Z) all points (may be large)
    st.session_state.setdefault("xyz_show_abs", None)   # downsample Nx3 (E,N,Z)
    st.session_state.setdefault("xy_show_local", None)  # downsample Nx2 (E_local, N_local)

    st.session_state.setdefault("tin_index", None)

    st.session_state.setdefault("axis_abs", [])         # [(E,N), ...]
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

                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["E0"] = None
                st.session_state["N0"] = None
                st.session_state["xyz_abs"] = None
                st.session_state["xyz_show_abs"] = None
                st.session_state["xy_show_local"] = None
                st.session_state["tin_index"] = None

                st.session_state["axis_abs"] = []
                st.session_state["axis_finished"] = False

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
        st.subheader("📄 LandXML → lae mudel")
        landxml = st.file_uploader("Laadi üles LandXML (.xml/.landxml)", type=["xml", "landxml"], key="landxml_upload")

        if landxml and st.button("Salvesta LandXML (R2) & lae punktid", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)

            pts, faces = read_landxml_tin_from_bytes(xml_bytes)
            if not pts:
                st.error("❌ Ei leidnud ühtegi punkti LandXML-ist.")
                st.stop()

            xyz = np.array(list(pts.values()), dtype=float)  # (E,N,Z)
            E0, N0 = compute_local_origin_EN(xyz)

            xyz_show = _downsample(xyz, max_pts=70000)
            xy_show_local = np.column_stack([
                xyz_show[:, 0] - E0,
                xyz_show[:, 1] - N0
            ])

            idx = build_tin_index(pts, faces)

            st.session_state["landxml_bytes"] = xml_bytes
            st.session_state["landxml_key"] = key
            st.session_state["xyz_abs"] = xyz
            st.session_state["xyz_show_abs"] = xyz_show
            st.session_state["xy_show_local"] = xy_show_local
            st.session_state["E0"] = E0
            st.session_state["N0"] = N0
            st.session_state["tin_index"] = idx

            st.session_state["axis_abs"] = []
            st.session_state["axis_finished"] = False

            st.success(f"✅ Laetud: punkte={xyz.shape[0]} | faces={len(faces)} | origin(E0,N0)={E0:.2f},{N0:.2f}")
            st.rerun()

        st.divider()

        # ---------------- Axis drawing ----------------
        st.subheader("🧭 Telje joonistamine (kliki pildil)")

        xml_bytes = st.session_state.get("landxml_bytes")
        E0 = st.session_state.get("E0")
        N0 = st.session_state.get("N0")
        xy_show_local = st.session_state.get("xy_show_local")
        tin_index = st.session_state.get("tin_index")

        if not xml_bytes or xy_show_local is None or tin_index is None or E0 is None or N0 is None:
            st.info("Lae LandXML üles, et näha punkte ja joonistada telge.")
        else:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1:
                if st.button("Alusta / joonista telg", use_container_width=True):
                    st.session_state["axis_abs"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with c2:
                if st.button("Undo", use_container_width=True):
                    if st.session_state["axis_abs"]:
                        st.session_state["axis_abs"] = st.session_state["axis_abs"][:-1]
                        st.session_state["axis_finished"] = False
                        st.rerun()
            with c3:
                if st.button("Tühjenda telg", use_container_width=True):
                    st.session_state["axis_abs"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with c4:
                if st.button("Lõpeta telg", use_container_width=True):
                    if len(st.session_state["axis_abs"]) < 2:
                        st.warning("Telg peab olema vähemalt 2 punktiga.")
                    else:
                        st.session_state["axis_finished"] = True
                        st.success("Telg lõpetatud.")
                        st.rerun()

            axis_abs = st.session_state["axis_abs"]
            axis_local = [to_local_EN(E, N, E0, N0) for (E, N) in axis_abs]

            fig = _make_figure(xy_show_local, axis_local)

            st.caption("👉 Kliki ükskõik kuhu (ka tühjale alale) — lisame teljele lähima TIN punkti (snap).")
            click_data = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=650,
                key="tin_plot_events",
            )

            if click_data and not st.session_state["axis_finished"]:
                x_loc = float(click_data[0]["x"])
                y_loc = float(click_data[0]["y"])
                E_click, N_click = to_abs_EN(x_loc, y_loc, E0, N0)

                snap = nearest_point_xyz(tin_index, E_click, N_click, max_rings=12)
                if snap is None:
                    st.warning("Snap ei leidnud lähimat punkti (proovi teises kohas).")
                else:
                    E_s, N_s, _Z_s = snap
                    st.session_state["axis_abs"] = axis_abs + [(float(E_s), float(N_s))]
                    st.rerun()

            if axis_abs:
                L = polyline_length(axis_abs)
                st.write(f"**Telje pikkus:** {L:.2f} m | Punkte: {len(axis_abs)}")

            st.markdown("### Arvutusparameetrid")
            a1, a2, a3 = st.columns(3)
            with a1:
                pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
                cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=25.0, step=1.0)
                sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.1, step=0.01)
            with a2:
                tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005)
                min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.2, step=0.05)
                min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.3, step=0.05)
            with a3:
                slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2")
                bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05)

            if st.button("Arvuta PK tabel (telje järgi)", use_container_width=True):
                if len(st.session_state["axis_abs"]) < 2:
                    st.warning("Joonista telg enne arvutamist.")
                else:
                    res = compute_pk_table_from_landxml(
                        xml_bytes=xml_bytes,
                        axis_xy=st.session_state["axis_abs"],  # ABS (E,N)
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

                    key = st.session_state.get("landxml_key") or p.get("landxml_key") or ""
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
                    "⬇️ Lae alla",
                    data=data,
                    file_name=f["name"],
                    key=f"dl_{f['key']}",
                    use_container_width=True,
                )
            with c:
                if st.button("🗑️ Kustuta", key=f"del_{f['key']}", use_container_width=True):
                    delete_key(s3, f["key"])
                    st.rerun()
