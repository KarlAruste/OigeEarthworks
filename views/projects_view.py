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
    compute_pk_table_from_landxml,   # peab sul olemas olema
    polyline_length,
)

from streamlit_plotly_events import plotly_events


def _downsample_points(xyz: np.ndarray, max_pts: int = 60000) -> np.ndarray:
    if xyz.shape[0] <= max_pts:
        return xyz
    idx = np.random.choice(xyz.shape[0], size=max_pts, replace=False)
    return xyz[idx]


def _make_tin_figure_local(en_local: np.ndarray, axis_local):
    """
    en_local: (N,2) local coords (E_local, N_local)
    axis_local: list[(E_local, N_local)]
    """
    fig = go.Figure()

    if en_local is None or en_local.size == 0:
        fig.update_layout(
            height=620,
            title="TIN punkte ei leitud (kontrolli LandXML)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    x = en_local[:, 0]
    y = en_local[:, 1]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = (xmax - xmin) if (xmax > xmin) else 1.0
    dy = (ymax - ymin) if (ymax > ymin) else 1.0
    pad_x = dx * 0.06
    pad_y = dy * 0.06

    # TIN punktid (klikitavad!)
    fig.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=6, opacity=0.55),  # suurem => lihtsam klikkida
        name="TIN punktid",
        hovertemplate="E_local=%{x:.2f}<br>N_local=%{y:.2f}<extra></extra>",
    ))

    # Telg
    if axis_local:
        xs = [p[0] for p in axis_local]
        ys = [p[1] for p in axis_local]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=10),
            name="Telg"
        ))

        L = polyline_length(axis_local)
        fig.add_annotation(
            x=xs[-1], y=ys[-1],
            text=f"Telg: {L:.2f} m",
            showarrow=True,
            arrowhead=2,
            ax=30, ay=-30
        )

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Pealtvaade (lokalne; kliki TIN punkti telje lisamiseks)",
        legend=dict(orientation="h"),
        dragmode="pan",
        uirevision="keep",
    )

    fig.update_xaxes(title="E (local, m)", range=[xmin - pad_x, xmax + pad_x], zeroline=False)
    fig.update_yaxes(title="N (local, m)", range=[ymin - pad_y, ymax + pad_y], scaleanchor="x", scaleratio=1, zeroline=False)

    return fig


def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # session state
    st.session_state.setdefault("active_project_id", None)

    # hoiame ABS telge DB/ARVUTUSTE jaoks
    st.session_state.setdefault("axis_xy_abs", [])
    st.session_state.setdefault("axis_finished", False)

    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)

    # local origin (ABS->local)
    st.session_state.setdefault("origin_E", None)
    st.session_state.setdefault("origin_N", None)

    # downsampled points local (display)
    st.session_state.setdefault("xyz_show_local", None)
    st.session_state.setdefault("xyz_show_abs", None)

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
                st.session_state["origin_E"] = None
                st.session_state["origin_N"] = None
                st.session_state["xyz_show_local"] = None
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

        if landxml and st.button("Salvesta LandXML (R2) & ava telje jaoks", use_container_width=True):
            prefix = project_prefix(p["name"]) + "landxml/"
            key = upload_file(s3, prefix, landxml)
            xml_bytes = download_bytes(s3, key)

            # parse points
            pts_dict, faces = read_landxml_tin_from_bytes(xml_bytes)
            if not pts_dict:
                st.error("Ei leidnud ühtegi TIN punkti. Kontrolli LandXML struktuuri (Definition/Pnts/P).")
                st.stop()

            xyz_abs = np.array(list(pts_dict.values()), dtype=float)  # (E,N,Z)
            # origin -> local (et poleks 6.5M telgi)
            origin_E = float(np.min(xyz_abs[:, 0]))
            origin_N = float(np.min(xyz_abs[:, 1]))

            en_local = np.column_stack([xyz_abs[:, 0] - origin_E, xyz_abs[:, 1] - origin_N])
            xyz_show_abs = _downsample_points(xyz_abs, max_pts=60000)
            en_show_local = np.column_stack([xyz_show_abs[:, 0] - origin_E, xyz_show_abs[:, 1] - origin_N])

            st.session_state["landxml_bytes"] = xml_bytes
            st.session_state["landxml_key"] = key

            st.session_state["origin_E"] = origin_E
            st.session_state["origin_N"] = origin_N

            st.session_state["xyz_show_abs"] = xyz_show_abs
            st.session_state["xyz_show_local"] = en_show_local

            st.session_state["axis_xy_abs"] = []
            st.session_state["axis_finished"] = False

            st.success(f"LandXML laetud: punkte={xyz_abs.shape[0]} | faces={len(faces)}")
            st.caption(f"Origin: E={origin_E:.3f} | N={origin_N:.3f}")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Axis drawing + PK ----------------
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("🧭 Telje joonistamine (polyline) + PK tabel")

        xml_bytes = st.session_state.get("landxml_bytes")
        origin_E = st.session_state.get("origin_E")
        origin_N = st.session_state.get("origin_N")
        xyz_show_local = st.session_state.get("xyz_show_local")
        xyz_show_abs = st.session_state.get("xyz_show_abs")

        if not xml_bytes or xyz_show_local is None or xyz_show_abs is None:
            st.info("Lae LandXML üles, et saaks telge joonistada ja punkte näha.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # nupud
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

            axis_abs = st.session_state["axis_xy_abs"]
            finished = st.session_state["axis_finished"]

            # ABS -> local telg joonistamiseks
            axis_local = [(x - origin_E, y - origin_N) for (x, y) in axis_abs]

            # fig
            fig = _make_tin_figure_local(xyz_show_local, axis_local)

            st.caption("👉 **Klikka TIN punktile** (sinine), siis lisame teljele lähima punkti (snap = see punkt).")
            st.caption("Zoom: toolbar (või rullik, kui brauser lubab). Pan: lohista.")

            # IMPORTANT:
            # streamlit-plotly-events 0.0.6 EI toeta config argumenti.
            # Seega ÄRA lisa config=... siia.
            click_data = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=620,
                key="tin_plot_events",
            )

            # klikk töötab ainult trace'i punktidel -> kasutame pointNumber'i
            if click_data and not finished:
                ev = click_data[0]
                if "pointNumber" in ev:
                    i = int(ev["pointNumber"])
                    # see indeks viitab meie xyz_show_local/abs punktile
                    e_abs = float(xyz_show_abs[i, 0])
                    n_abs = float(xyz_show_abs[i, 1])
                    st.session_state["axis_xy_abs"] = axis_abs + [(e_abs, n_abs)]
                    st.rerun()
                else:
                    st.warning("Klikk ei tabanud punkti. Proovi klikata sinisele punktile lähemalt.")

            if axis_abs:
                L = polyline_length(axis_abs)
                st.write(f"**Telje pikkus:** {L:.2f} m  |  Punkte: {len(axis_abs)}")

            # ---------------- Params + Compute ----------------
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
                        axis_xy=axis_abs,  # ABS koordinaadid!
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
