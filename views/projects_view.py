# views/projects_view.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from db import (
    list_projects, create_project, get_project,
    set_project_landxml
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


def _downsample_points(xyz: np.ndarray, max_pts: int = 50000) -> np.ndarray:
    if xyz.shape[0] <= max_pts:
        return xyz
    idx = np.random.choice(xyz.shape[0], size=max_pts, replace=False)
    return xyz[idx]


def _make_tin_figure(xyz_show: np.ndarray, axis_xy):
    fig = go.Figure()

    # turvalisus: kui xyz_show tühi, näita midagi
    if xyz_show is None or xyz_show.size == 0:
        fig.update_layout(
            height=620,
            title="TIN punkte ei leitud (kontrolli LandXML)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    x = xyz_show[:, 0]
    y = xyz_show[:, 1]

    # Range (et punktid oleks kohe näha)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # väike padding
    dx = (xmax - xmin) if (xmax > xmin) else 1.0
    dy = (ymax - ymin) if (ymax > ymin) else 1.0
    pad_x = dx * 0.05
    pad_y = dy * 0.05

    fig.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=3, opacity=0.6),
        name="TIN punktid",
        hoverinfo="skip"
    ))

    if axis_xy:
        xs = [p[0] for p in axis_xy]
        ys = [p[1] for p in axis_xy]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=8),
            name="Telg"
        ))

        L = polyline_length(axis_xy)
        fig.add_annotation(
            x=xs[-1], y=ys[-1],
            text=f"{L:.2f} m",
            showarrow=True,
            arrowhead=2,
            ax=30, ay=-30
        )

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Pealtvaade (kliki punkte telje joonistamiseks)",
        legend=dict(orientation="h"),
        dragmode="pan",
        uirevision="keep",
    )

    fig.update_xaxes(title="X", range=[xmin - pad_x, xmax + pad_x])
    fig.update_yaxes(title="Y", range=[ymin - pad_y, ymax + pad_y], scaleanchor="x", scaleratio=1)

    return fig



def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    if "active_project_id" not in st.session_state:
        st.session_state["active_project_id"] = None

    if "axis_xy" not in st.session_state:
        st.session_state["axis_xy"] = []
    if "axis_finished" not in st.session_state:
        st.session_state["axis_finished"] = False

    if "landxml_bytes" not in st.session_state:
        st.session_state["landxml_bytes"] = None
    if "landxml_key" not in st.session_state:
        st.session_state["landxml_key"] = None

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
                st.session_state["axis_xy"] = []
                st.session_state["axis_finished"] = False
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
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
            st.session_state["axis_xy"] = []
            st.session_state["axis_finished"] = False

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
        else:
            try:
                pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
                xyz = np.array(list(pts_dict.values()), dtype=float)
                xyz_show = _downsample_points(xyz, max_pts=60000)
            except Exception as e:
                st.error(f"LandXML lugemine ebaõnnestus: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            cA, cB, cC, cD = st.columns([1, 1, 1, 2])
            with cA:
                if st.button("Alusta / joonista telg", use_container_width=True):
                    st.session_state["axis_xy"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with cB:
                if st.button("Undo (eemalda viimane)", use_container_width=True):
                    if st.session_state["axis_xy"]:
                        st.session_state["axis_xy"] = st.session_state["axis_xy"][:-1]
                        st.session_state["axis_finished"] = False
                        st.rerun()
            with cC:
                if st.button("Tühjenda telg", use_container_width=True):
                    st.session_state["axis_xy"] = []
                    st.session_state["axis_finished"] = False
                    st.rerun()
            with cD:
                if st.button("Lõpeta telg", use_container_width=True):
                    if len(st.session_state["axis_xy"]) < 2:
                        st.warning("Telg peab olema vähemalt 2 punktiga.")
                    else:
                        st.session_state["axis_finished"] = True
                        st.success("Telg lõpetatud.")
                        st.rerun()

            axis_xy = st.session_state["axis_xy"]
            finished = st.session_state["axis_finished"]

            fig = _make_tin_figure(xyz_show, axis_xy)

            st.caption("👉 Klikk graafikul lisab telje punkti. Zoom/pan töötab Plotly toolbarist.")
            click_data = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=620,
    key="tin_plot_events",
    config={
        "scrollZoom": True,        # hiire rullik zoom
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                              },
                  )



            if click_data and not finished:
                x = float(click_data[0]["x"])
                y = float(click_data[0]["y"])
                st.session_state["axis_xy"] = axis_xy + [(x, y)]
                st.rerun()

            if axis_xy:
                L = polyline_length(axis_xy)
                st.write(f"**Telje pikkus:** {L:.2f} m  |  Punkte: {len(axis_xy)}")

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
                if len(axis_xy) < 2:
                    st.warning("Joonista telg enne arvutamist.")
                else:
                    res = compute_pk_table_from_landxml(
                        xml_bytes=xml_bytes,
                        axis_xy=axis_xy,
                        pk_step=float(pk_step),
                        cross_len=float(cross_len),
                        sample_step=float(sample_step),
                        tol=float(tol),
                        min_run=float(min_run),
                        min_depth_from_bottom=float(min_depth),
                        slope_text=slope,
                        bottom_w=float(bottom_w),
                    )

                    rows = res["rows"]
                    df = pd.DataFrame(rows)

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
                        use_container_width=True
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
