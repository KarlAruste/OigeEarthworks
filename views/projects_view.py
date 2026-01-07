# views/projects_view.py

import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    polyline_length,
)

from streamlit_plotly_events import plotly_events


# -------------------------
# Helpers
# -------------------------

def _downsample_points(xyz: np.ndarray, max_pts: int = 60000) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


def _ensure_xy_is_EN(xyz: np.ndarray) -> np.ndarray:
    """
    Your coords example:
      N=6_568_466 (big), E=609_910 (small), Z=...
    We want X=E (small), Y=N (big).

    If median X looks like Northing (~millions) and median Y looks like Easting (~hundreds of thousands),
    swap first two columns.
    """
    if xyz is None or xyz.size == 0:
        return xyz

    x_med = float(np.nanmedian(xyz[:, 0]))
    y_med = float(np.nanmedian(xyz[:, 1]))

    # heuristic: Northing usually > 2,000,000; Easting usually < 2,000,000 (in many projected systems incl. Estonia)
    if x_med > 2_000_000 and y_med < 2_000_000:
        # xyz currently (N,E,Z) -> convert to (E,N,Z)
        xyz2 = xyz.copy()
        xyz2[:, 0], xyz2[:, 1] = xyz[:, 1], xyz[:, 0]
        return xyz2

    return xyz


def _origin_abs(xyz_abs: np.ndarray) -> tuple[float, float]:
    # origin = mean to keep local coords centered
    x0 = float(np.nanmean(xyz_abs[:, 0]))
    y0 = float(np.nanmean(xyz_abs[:, 1]))
    return x0, y0


def _abs_to_local_xy(xyz_abs: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    x0, y0 = origin
    out = xyz_abs[:, :2].astype(float).copy()
    out[:, 0] -= x0
    out[:, 1] -= y0
    return out


def _axis_abs_to_local(axis_abs: list[tuple[float, float]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    x0, y0 = origin
    return [(float(x - x0), float(y - y0)) for (x, y) in axis_abs]


def _snap_local_click_to_nearest_abs(
    click_local_xy: tuple[float, float],
    xyz_abs: np.ndarray,
    origin: tuple[float, float],
) -> tuple[float, float]:
    """
    click_local_xy is in local coords.
    Snap to nearest xyz_abs point (but compute in local for numeric stability).
    """
    x0, y0 = origin
    cx = float(click_local_xy[0])
    cy = float(click_local_xy[1])

    X = (xyz_abs[:, 0].astype(float) - x0)
    Y = (xyz_abs[:, 1].astype(float) - y0)

    m = np.isfinite(X) & np.isfinite(Y)
    if not np.any(m):
        return (float(cx + x0), float(cy + y0))

    X = X[m]
    Y = Y[m]
    dx = X - cx
    dy = Y - cy
    j = int(np.argmin(dx * dx + dy * dy))

    # return ABS coords
    return (float(X[j] + x0), float(Y[j] + y0))


def _make_plotly_fig(local_pts_xy: np.ndarray, axis_local_xy: list[tuple[float, float]]):
    fig = go.Figure()

    if local_pts_xy is None or local_pts_xy.size == 0:
        fig.update_layout(
            height=650,
            title="TIN punkte ei leitud",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    x = local_pts_xy[:, 0]
    y = local_pts_xy[:, 1]

    # 1) visible points
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=4, opacity=0.75),
            name="TIN punktid",
            hoverinfo="skip",
        )
    )

    # 2) big almost-invisible click-catcher layer (so click works even if not pixel-perfect)
    #    NOTE: opacity must be >0 to receive click events reliably.
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=18, opacity=0.01),
            name="click_layer",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # axis
    if axis_local_xy and len(axis_local_xy) > 0:
        xs = [p[0] for p in axis_local_xy]
        ys = [p[1] for p in axis_local_xy]
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

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    pad_x = dx * 0.08
    pad_y = dy * 0.08

    fig.update_layout(
        height=650,
        title="Pealtvaade (lokaalne; kliki telje punktide lisamiseks)",
        margin=dict(l=10, r=10, t=50, b=10),
        dragmode="pan",
        uirevision="keep",
        clickmode="event+select",
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title="E (local, m)", range=[xmin - pad_x, xmax + pad_x])
    fig.update_yaxes(title="N (local, m)", range=[ymin - pad_y, ymax + pad_y], scaleanchor="x", scaleratio=1)

    return fig


# -------------------------
# Main view
# -------------------------

def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("axis_xy_abs", [])        # [(E,N)] ABS
    st.session_state.setdefault("axis_finished", False)
    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)
    st.session_state.setdefault("origin_abs", None)
    st.session_state.setdefault("xyz_show_abs", None)     # ABS for display+snap

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
                st.session_state["axis_xy_abs"] = []
                st.session_state["axis_finished"] = False
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["origin_abs"] = None
                st.session_state["xyz_show_abs"] = None
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
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📄 LandXML → salvesta ja lae mudel")

        landxml = st.file_uploader("Laadi üles LandXML (.xml/.landxml)", type=["xml", "landxml"], key="landxml_upload")

        if landxml and st.button("Salvesta LandXML (R2) & ava joonistamiseks", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)

            st.session_state["landxml_bytes"] = xml_bytes
            st.session_state["landxml_key"] = key
            st.session_state["axis_xy_abs"] = []
            st.session_state["axis_finished"] = False
            st.session_state["origin_abs"] = None
            st.session_state["xyz_show_abs"] = None

            st.success("LandXML salvestatud ja laaditud sessiooni.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Axis drawing + PK ----------------
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("🧭 Telje joonistamine (polyline) + PK tabel")

        xml_bytes = st.session_state["landxml_bytes"]
        if not xml_bytes:
            st.info("Lae LandXML üles, et saaks telge joonistada.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Read points
        try:
            pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)

            xyz_raw = np.array(list(pts_dict.values()), dtype=float)
            xyz_raw = xyz_raw[np.isfinite(xyz_raw).all(axis=1)]

            # Ensure (E,N,Z)
            xyz_abs = _ensure_xy_is_EN(xyz_raw)

            xyz_show_abs = _downsample_points(xyz_abs, max_pts=60000)
            st.session_state["xyz_show_abs"] = xyz_show_abs

            if st.session_state["origin_abs"] is None:
                st.session_state["origin_abs"] = _origin_abs(xyz_show_abs)

            x_min = float(np.nanmin(xyz_show_abs[:, 0])); x_max = float(np.nanmax(xyz_show_abs[:, 0]))
            y_min = float(np.nanmin(xyz_show_abs[:, 1])); y_max = float(np.nanmax(xyz_show_abs[:, 1]))
            st.caption(f"ABS X(E) min/max: {x_min:.3f} / {x_max:.3f} | ABS Y(N) min/max: {y_min:.3f} / {y_max:.3f}")
            st.caption(f"Punkte kokku: {xyz_raw.shape[0]} | Näitan: {xyz_show_abs.shape[0]}")

        except Exception as e:
            st.error(f"LandXML lugemine ebaõnnestus: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Buttons
        cA, cB, cC, cD = st.columns([1, 1, 1, 2])
        with cA:
            if st.button("Alusta / joonista telg", use_container_width=True):
                st.session_state["axis_xy_abs"] = []
                st.session_state["axis_finished"] = False
                st.rerun()
        with cB:
            if st.button("Undo (eemalda viimane)", use_container_width=True):
                if st.session_state["axis_xy_abs"]:
                    st.session_state["axis_xy_abs"] = st.session_state["axis_xy_abs"][:-1]
                    st.session_state["axis_finished"] = False
                    st.rerun()
        with cC:
            if st.button("Tühjenda telg", use_container_width=True):
                st.session_state["axis_xy_abs"] = []
                st.session_state["axis_finished"] = False
                st.rerun()
        with cD:
            if st.button("Lõpeta telg", use_container_width=True):
                if len(st.session_state["axis_xy_abs"]) < 2:
                    st.warning("Telg peab olema vähemalt 2 punktiga.")
                else:
                    st.session_state["axis_finished"] = True
                    st.success("Telg lõpetatud.")
                    st.rerun()

        axis_xy_abs = st.session_state["axis_xy_abs"]
        finished = st.session_state["axis_finished"]
        origin = st.session_state["origin_abs"]

        # Local view for stability
        pts_local = _abs_to_local_xy(xyz_show_abs, origin)
        axis_local = _axis_abs_to_local(axis_xy_abs, origin)

        fig = _make_plotly_fig(pts_local, axis_local)

        st.caption("👉 Kliki punktide/joone lähedale — lisab teljele LÄHIMA TIN punkti (snap). (Pan: lohista)")

        # plotly_events: click event only fires when clicking a trace.
        # We added 'click_layer' with big nearly transparent markers to catch clicks.
        click_data = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=650,
            key="tin_plot_events_local",
        )

        # Debug line (shows if click is received)
        st.write(f"Klikke saadud: {len(click_data) if click_data else 0}")

        if click_data and (not finished):
            lx = float(click_data[0]["x"])
            ly = float(click_data[0]["y"])
            ax, ay = _snap_local_click_to_nearest_abs((lx, ly), xyz_show_abs, origin)
            st.session_state["axis_xy_abs"] = axis_xy_abs + [(ax, ay)]
            st.rerun()

        if axis_xy_abs:
            L = polyline_length(axis_xy_abs)
            st.write(f"**Telje pikkus:** {L:.2f} m  |  Punkte: {len(axis_xy_abs)}")

        # Parameters
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
            if len(axis_xy_abs) < 2:
                st.warning("Joonista telg enne arvutamist (vähemalt 2 punkti).")
            else:
                res = compute_pk_table_from_landxml(
                    xml_bytes=xml_bytes,
                    axis_xy=axis_xy_abs,  # ABS (E,N)
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

                key = st.session_state.get("landxml_key") or (p.get("landxml_key") or "")
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

        p2 = get_project(pid)
        if p2 and p2.get("planned_volume_m3") is not None:
            st.markdown("### Projekti salvestatud planeeritud väärtused")
            st.write(f"**Planned length:** {float(p2['planned_length_m'] or 0):.2f} m")
            st.write(f"**Planned area:** {float(p2['planned_area_m2'] or 0):.3f} m²")
            st.write(f"**Planned volume:** {float(p2['planned_volume_m3'] or 0):.3f} m³")

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
