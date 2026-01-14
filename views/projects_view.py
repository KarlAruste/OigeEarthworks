# views/projects_view.py
import io
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from typing import List, Tuple, Optional

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

# -------------------------------------------------
# Helpers (DB rows: dict OR tuple)
# -------------------------------------------------
def _row_get(r, key, default=None):
    """
    Supports:
      - dict rows (psycopg2 RealDictCursor)
      - psycopg Row (._mapping)
      - tuple rows fallback (assumes projects SELECT * column order)
    """
    if r is None:
        return default

    if isinstance(r, dict):
        return r.get(key, default)

    if hasattr(r, "_mapping"):
        try:
            return r._mapping.get(key, default)
        except Exception:
            pass

    # tuple fallback (projects table order)
    idx = {
        "id": 0,
        "name": 1,
        "start_date": 2,
        "end_date": 3,
        "landxml_key": 4,
        "top_width_m": 5,
        "planned_length_m": 6,
        "planned_area_m2": 7,
        "planned_volume_m3": 8,
        "created_at": 9,
    }
    i = idx.get(key, None)
    if i is None:
        return default
    try:
        return r[i]
    except Exception:
        return default


# -------------------------------------------------
# LandXML/Alignment parsing helpers
# -------------------------------------------------
def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _autodetect_swap_en(xy: List[Tuple[float, float]]) -> bool:
    """
    Heuristik: Eestis (L-EST97) Easting ~ 6xx xxx, Northing ~ 65xx xxx.
    Mõnel ekspordil on need vahetuses.
    """
    if not xy or len(xy) < 3:
        return False

    xs = np.array([p[0] for p in xy], dtype=float)
    ys = np.array([p[1] for p in xy], dtype=float)

    # typical ranges
    # E: ~ 300k..900k, N: ~ 6.4M..6.7M  (or vice versa if swapped)
    x_med = float(np.nanmedian(xs))
    y_med = float(np.nanmedian(ys))

    x_looks_north = (6_000_000 <= x_med <= 7_000_000)
    y_looks_east = (100_000 <= y_med <= 1_200_000)

    return bool(x_looks_north and y_looks_east)


def _swap_xy(xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(y, x) for (x, y) in xy]


def _parse_alignment_landxml_bytes(b: bytes) -> List[Tuple[float, float]]:
    """
    Loeb LandXML alignmenti koordinaadid:
      - <P> x y z </P> (või x y)
      - <Start>, <End>, <Center>, <PI> jne (x y)
    Tagastab list[(x,y)] järjestuses nagu leitakse.
    """
    try:
        text = b.decode("utf-8", errors="ignore")
    except Exception:
        text = str(b)

    # Simple regex-based extraction (works well for Civil3D exports)
    pts = []

    # 1) <P>...</P>
    for m in re.finditer(r"<P[^>]*>([^<]+)</P>", text):
        raw = m.group(1).strip()
        parts = raw.split()
        if len(parts) >= 2:
            x = _safe_float(parts[0])
            y = _safe_float(parts[1])
            if x is not None and y is not None:
                pts.append((x, y))

    # 2) <Start> x y </Start> etc
    for tag in ["Start", "End", "Center", "PI"]:
        for m in re.finditer(rf"<{tag}>([^<]+)</{tag}>", text):
            raw = m.group(1).strip()
            parts = raw.split()
            if len(parts) >= 2:
                x = _safe_float(parts[0])
                y = _safe_float(parts[1])
                if x is not None and y is not None:
                    pts.append((x, y))

    # de-duplicate consecutive duplicates
    out = []
    for p in pts:
        if not out or (abs(out[-1][0] - p[0]) > 1e-9 or abs(out[-1][1] - p[1]) > 1e-9):
            out.append(p)

    return out


def _parse_axis_csv_or_txt_bytes(b: bytes) -> List[Tuple[float, float]]:
    """
    Supports:
      - CSV with headers E,N or X,Y
      - plain txt with two columns
      - separators: ; , whitespace
    """
    s = b.decode("utf-8", errors="ignore").strip()
    if not s:
        return []

    # try pandas read_csv with common separators
    for sep in [";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(s), sep=sep, engine="python")
            if df.shape[1] < 2:
                continue

            cols = [c.strip().lower() for c in df.columns]
            # try find e/n columns
            def col_find(names):
                for n in names:
                    if n in cols:
                        return df.columns[cols.index(n)]
                return None

            cE = col_find(["e", "east", "easting", "x"])
            cN = col_find(["n", "north", "northing", "y"])

            if cE is None or cN is None:
                # fallback first two columns
                cE = df.columns[0]
                cN = df.columns[1]

            xy = []
            for _, r in df.iterrows():
                x = _safe_float(r[cE])
                y = _safe_float(r[cN])
                if x is not None and y is not None:
                    xy.append((float(x), float(y)))
            if len(xy) >= 2:
                return xy
        except Exception:
            pass

    # last fallback: manual parse lines
    xy = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[;,\s]+", line)
        if len(parts) >= 2:
            x = _safe_float(parts[0])
            y = _safe_float(parts[1])
            if x is not None and y is not None:
                xy.append((x, y))
    return xy


def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


# -------------------------------------------------
# Main view
# -------------------------------------------------
def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects() or []

    # Session defaults
    st.session_state.setdefault("active_project_id", None)

    # Mahud tab state
    st.session_state.setdefault("mahud_landxml_key", None)
    st.session_state.setdefault("mahud_landxml_bytes", None)
    st.session_state.setdefault("mahud_axis_xy", None)  # list[(E,N)]
    st.session_state.setdefault("mahud_axis_name", None)
    st.session_state.setdefault("mahud_swap_en", False)

    # ---------------- Create project ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Loo projekt")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Projekti nimi", placeholder="nt Tapa_Objekt_01")
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
            proj_id = _row_get(proj, "id")
            proj_name = _row_get(proj, "name")
            if st.button(str(proj_name), use_container_width=True, key=f"projbtn_{proj_id}"):
                st.session_state["active_project_id"] = int(proj_id)

                # reset Mahud session bits when switching project
                st.session_state["mahud_landxml_key"] = None
                st.session_state["mahud_landxml_bytes"] = None
                st.session_state["mahud_axis_xy"] = None
                st.session_state["mahud_axis_name"] = None
                st.session_state["mahud_swap_en"] = False

                st.rerun()

    with right:
        pid = st.session_state.get("active_project_id")
        if not pid:
            st.info("Vali vasakult projekt.")
            return

        p = get_project(int(pid))
        if not p:
            st.session_state["active_project_id"] = None
            st.rerun()

        project_name = _row_get(p, "name", "")
        st.subheader(str(project_name))
        st.caption(f"Tähtaeg: {_row_get(p, 'end_date')}")

        tabs = st.tabs(["Üldine", "Mahud", "Failid"])

        # ==========================================================
        # TAB: ÜLDINE
        # ==========================================================
        with tabs[0]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📊 Planeeritud väärtused (DB)")

            pv = _row_get(p, "planned_volume_m3")
            pl = _row_get(p, "planned_length_m")
            pa = _row_get(p, "planned_area_m2")
            lk = _row_get(p, "landxml_key")

            st.write(f"**LandXML key:** {lk or '—'}")
            st.write(f"**Planned length:** {float(pl or 0):.2f} m")
            st.write(f"**Planned area:** {float(pa or 0):.3f} m²")
            st.write(f"**Planned volume:** {float(pv or 0):.3f} m³")

            st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================================
        # TAB: MAHUD
        # ==========================================================
        with tabs[1]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("🧮 Mahud (LandXML + telg/alignment failist)")

            # ---------- 1) LandXML: upload or select existing ----------
            st.markdown("### 1) Pinnamudel (LandXML)")

            prefix_landxml = project_prefix(project_name) + "landxml/"
            all_proj_files = list_files(s3, prefix_landxml) or []
            landxml_files = [f for f in all_proj_files if f["name"].lower().endswith((".xml", ".landxml"))]

            colA, colB = st.columns([1, 1], gap="large")

            with colA:
                up = st.file_uploader("Laadi üles LandXML (.xml/.landxml)", type=["xml", "landxml"], key="mahud_landxml_upload")
                if up and st.button("Salvesta LandXML (R2)", use_container_width=True, key="mahud_save_landxml"):
                    key = upload_file(s3, prefix_landxml, up)
                    b = download_bytes(s3, key)
                    st.session_state["mahud_landxml_key"] = key
                    st.session_state["mahud_landxml_bytes"] = b
                    st.success("LandXML salvestatud ja laaditud.")
                    st.rerun()

            with colB:
                if landxml_files:
                    options = ["— vali olemasolev —"] + [f["name"] for f in landxml_files]
                    sel = st.selectbox("Vali varem üles laaditud LandXML", options, key="mahud_landxml_select")
                    if sel and sel != "— vali olemasolev —":
                        # find key
                        fmatch = next((f for f in landxml_files if f["name"] == sel), None)
                        if fmatch and st.button("Kasuta valitud LandXML", use_container_width=True, key="mahud_use_selected_landxml"):
                            b = download_bytes(s3, fmatch["key"])
                            st.session_state["mahud_landxml_key"] = fmatch["key"]
                            st.session_state["mahud_landxml_bytes"] = b
                            st.success("LandXML laetud valikust.")
                            st.rerun()
                else:
                    st.info("Selles projektis pole veel LandXML faile (landxml/).")

            landxml_key = st.session_state.get("mahud_landxml_key") or _row_get(p, "landxml_key")
            landxml_bytes = st.session_state.get("mahud_landxml_bytes")

            if landxml_bytes is None and landxml_key:
                try:
                    landxml_bytes = download_bytes(s3, landxml_key)
                    st.session_state["mahud_landxml_bytes"] = landxml_bytes
                    st.session_state["mahud_landxml_key"] = landxml_key
                except Exception:
                    pass

            st.caption(f"Aktiivne LandXML: **{landxml_key or '—'}**")

            # quick sanity: show point count
            if landxml_bytes:
                try:
                    pts_dict, _faces = read_landxml_tin_from_bytes(landxml_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)
                    st.caption(f"Punkte (TIN): **{xyz.shape[0]}**")
                except Exception as e:
                    st.error(f"LandXML TIN lugemine ebaõnnestus: {e}")

            st.divider()

            # ---------- 2) Axis/alignment import ----------
            st.markdown("### 2) Telg / Alignment (failist)")

            axis_file = st.file_uploader(
                "Laadi teljefail (CSV/TXT punktid või Alignment LandXML)",
                type=["csv", "txt", "xml", "landxml"],
                key="mahud_axis_upload",
            )

            colS1, colS2, colS3 = st.columns([1, 1, 1], gap="large")
            with colS1:
                swap_manual = st.checkbox(
                    "Telje koordinaadid on vahetuses (N=Easting ja E=Northing) → Swap E/N",
                    value=bool(st.session_state.get("mahud_swap_en", False)),
                    key="mahud_swap_en_cb",
                )
                st.session_state["mahud_swap_en"] = bool(swap_manual)

            with colS2:
                snap_note = st.caption("Kui telg tuleb Civil3D-st ja veerud on vahetuses, kasuta Swap E/N.")

            with colS3:
                st.caption("Soovitus: Alignment LandXML (<Alignments>) või lihtne CSV (E;N).")

            if axis_file and st.button("Impordi telg", use_container_width=True, key="mahud_import_axis"):
                b = axis_file.read()
                name = axis_file.name

                xy = []
                if name.lower().endswith((".xml", ".landxml")):
                    xy = _parse_alignment_landxml_bytes(b)
                else:
                    xy = _parse_axis_csv_or_txt_bytes(b)

                if not xy or len(xy) < 2:
                    st.error("Telje import ebaõnnestus: ei leidnud piisavalt punkte.")
                else:
                    # auto-detect swap suggestion
                    auto_swap = _autodetect_swap_en(xy)
                    do_swap = bool(st.session_state["mahud_swap_en"])
                    if auto_swap and not do_swap:
                        st.warning("Auto-detect: telje E/N näivad vahetuses. Lülita 'Swap E/N' sisse ja impordi uuesti (või arvuta swapiga).")

                    if do_swap:
                        xy = _swap_xy(xy)

                    st.session_state["mahud_axis_xy"] = xy
                    st.session_state["mahud_axis_name"] = name
                    st.success(f"Telg imporditud: {len(xy)} punkti.")
                    st.rerun()

            axis_xy = st.session_state.get("mahud_axis_xy") or []
            axis_name = st.session_state.get("mahud_axis_name") or "—"

            if axis_xy:
                L = polyline_length(axis_xy)
                st.write(f"**Telg:** {axis_name}  |  Punkte: **{len(axis_xy)}**  |  Pikkus: **{L:.2f} m**")
                with st.expander("Näita telje esimesed 10 punkti"):
                    df_axis = pd.DataFrame(axis_xy[:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)
            else:
                st.info("Impordi telg/alignment, et saaks PK tabelit arvutada.")

            st.divider()

            # ---------- 3) Calc params ----------
            st.markdown("### 3) Arvutusparameetrid")
            c1, c2, c3 = st.columns(3)

            with c1:
                pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1, key="mahud_pk_step")
                cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=17.0, step=1.0, key="mahud_cross_len")
                sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.1, step=0.01, key="mahud_sample_step")

            with c2:
                tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005, key="mahud_tol")
                min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.2, step=0.05, key="mahud_min_run")
                min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.3, step=0.05, key="mahud_min_depth")

            with c3:
                slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2", key="mahud_slope")
                bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05, key="mahud_bottom_w")

            st.divider()

            # ---------- 4) Compute ----------
            st.markdown("### 4) Arvuta")
            if st.button("Arvuta PK tabel", use_container_width=True, key="mahud_calc_btn"):
                if not landxml_bytes:
                    st.error("Vali või lae LandXML enne arvutamist.")
                elif not axis_xy or len(axis_xy) < 2:
                    st.error("Impordi telg/alignment enne arvutamist.")
                else:
                    try:
                        res = compute_pk_table_from_landxml(
                            xml_bytes=landxml_bytes,
                            axis_xy=axis_xy,
                            pk_step=float(pk_step),
                            cross_len=float(cross_len),
                            sample_step=float(sample_step),
                            tol=float(tol),
                            min_run=float(min_run),
                            min_depth_from_bottom=float(min_depth),
                            slope_text=str(slope),
                            bottom_w=float(bottom_w),
                        )

                        rows = res.get("rows", [])
                        df = pd.DataFrame(rows)

                        total_v = res.get("total_volume_m3", 0.0)
                        axis_len = res.get("axis_length_m", polyline_length(axis_xy))
                        count = res.get("count", len(rows))

                        st.success(f"✅ Kokku maht: {float(total_v or 0):.3f} m³")
                        st.write(f"Telje pikkus: **{float(axis_len or 0):.2f} m** | PK-sid: **{int(count or 0)}**")

                        planned_area = None
                        if axis_len and float(axis_len) > 0:
                            planned_area = float(total_v) / float(axis_len)

                        # Save into DB planned_* + landxml_key
                        key_to_save = st.session_state.get("mahud_landxml_key") or landxml_key or ""
                        set_project_landxml(
                            int(_row_get(p, "id")),
                            landxml_key=key_to_save,
                            planned_volume_m3=float(total_v),
                            planned_length_m=float(axis_len),
                            planned_area_m2=float(planned_area) if planned_area is not None else None,
                        )

                        st.dataframe(df, use_container_width=True)

                        csv = df.to_csv(index=False, sep=";").encode("utf-8")
                        st.download_button(
                            "⬇️ Lae alla CSV",
                            data=csv,
                            file_name=f"{project_name}_pk_tabel.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="mahud_dl_csv",
                        )

                    except Exception as e:
                        st.error(f"Arvutus ebaõnnestus: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================================
        # TAB: FAILID
        # ==========================================================
        with tabs[2]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📤 Failid (Cloudflare R2)")

            uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files_uploads")
            if uploads:
                prefix = project_prefix(project_name)
                for up in uploads:
                    upload_file(s3, prefix, up)
                st.success(f"Üles laaditud {len(uploads)} faili.")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📄 Projekti failid")
            prefix = project_prefix(project_name)
            files = list_files(s3, prefix) or []

            if not files:
                st.info("Selles projektis pole veel faile.")
                return

            for f in files:
                a, bcol, c = st.columns([6, 2, 2])
                with a:
                    st.write(f"📄 {f['name']}")
                    st.caption(f"{f.get('size', 0) / 1024:.1f} KB")
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
