# views/projects_view.py
# Projects view + Mahud tab (LandXML volume). Axis imported from Civil3D (Alignment/Polyline LandXML or CSV/TXT).
# Fixes:
#  - Adds "Mahud" tab
#  - LandXML surfaces are stored in R2 and can be selected later (no re-upload needed)
#  - Axis import supports E/N swap (N=Easting, E=Northing) with heuristic + checkbox
#  - Axis length uses Alignment length attribute if present (fix “wrong length”)
#  - PK compute runs and shows why it fails (coverage check + robust edge fallback)
#  - Text readability improvements (lighter text blocks, clearer captions)

import io
import os
import re
import json
import math
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

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
    build_tin_index,
    nearest_point_xyz,
)

# -------------------------
# UI helpers
# -------------------------

def _ui_readability_css():
    st.markdown(
        """
<style>
/* Improve readability on dark background */
.block { background:#141821; border:1px solid #243042; border-radius:14px; padding:16px; }
.small { color:#cbd5e1; font-size:13px; }
.caption2 { color:#e5e7eb; opacity:0.85; font-size:13px; }
.kv { color:#e5e7eb; font-size:14px; }
.kv b { color:#ffffff; }
hr.soft { border:none; border-top:1px solid #243042; margin:14px 0; }
</style>
""",
        unsafe_allow_html=True,
    )


def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    """Downsample to keep UI fast."""
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


# -------------------------
# R2 helpers for LandXML selection
# -------------------------

def _list_landxml_keys_for_project(s3, project_name: str) -> list[dict]:
    """
    Returns list of dicts: {"key":..., "name":..., "size":...}
    Only files under prefix landxml/
    """
    prefix = project_prefix(project_name) + "landxml/"
    files = list_files(s3, prefix)
    out = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith((".xml", ".landxml")):
            out.append(f)
    return out


# -------------------------
# Axis import (Civil3D)
# -------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _iter_by_local(root, local_name: str):
    ln = local_name.lower()
    for el in root.iter():
        if _strip_ns(el.tag).lower() == ln:
            yield el


def _parse_two_floats(text: str):
    parts = (text or "").strip().replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def read_axis_from_civil3d_file(file_bytes: bytes, filename: str) -> dict:
    """
    Supports:
      - LandXML Alignment (<Alignments><Alignment ... length="..."><CoordGeom> ... <Line><Start> E N </Start> <End> E N </End>
                          <Curve> <Start>..</Start> <PI>..</PI> <End>..</End> ... </Curve> ...)
        We extract points from Start/PI/End tags. (Enough for chainage/tangent sampling)
      - CSV/TXT with columns or pairs:
          E;N   or  E,N  or  E N  per line
    Returns:
      {"xy": [(E,N),...], "length_attr": float|None, "source": str}
    """
    name = (filename or "").lower()

    # ---- CSV/TXT ----
    if name.endswith((".csv", ".txt")):
        txt = file_bytes.decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        pts = []
        for ln in lines:
            # allow header lines
            if re.search(r"[a-zA-Z]", ln) and not re.search(r"\d", ln):
                continue
            # split by ; , or whitespace
            parts = re.split(r"[;,\s]+", ln.strip())
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
                pts.append((a, b))
            except Exception:
                continue

        if len(pts) < 2:
            raise ValueError("Telje failist ei leitud piisavalt punkte (vähemalt 2).")

        return {"xy": pts, "length_attr": None, "source": "csv/txt"}

    # ---- XML / LandXML ----
    root = ET.fromstring(file_bytes)

    # Find first Alignment length attribute (prefer)
    length_attr = None
    for al in _iter_by_local(root, "Alignment"):
        la = al.get("length")
        if la:
            try:
                length_attr = float(la)
            except Exception:
                length_attr = None
        # take first alignment
        break

    # Extract points from CoordGeom tags in order:
    # Start, PI, End (and also any PIs inside Curve blocks)
    pts = []

    # Use Start/PI/End tags in document order
    for tag in ("Start", "PI", "End"):
        for el in _iter_by_local(root, tag):
            xy = _parse_two_floats(el.text)
            if xy is None:
                continue
            pts.append(xy)

    # If that was too many / messy, try more structured: within CoordGeom
    # But simplest: dedupe consecutive duplicates
    def dedupe_consecutive(xy_list):
        out = []
        for p in xy_list:
            if not out:
                out.append(p)
            else:
                if abs(out[-1][0] - p[0]) > 1e-9 or abs(out[-1][1] - p[1]) > 1e-9:
                    out.append(p)
        return out

    pts = dedupe_consecutive(pts)

    if len(pts) < 2:
        raise ValueError("Telje LandXML-ist ei leitud piisavalt koordinaate (Start/PI/End).")

    return {"xy": pts, "length_attr": length_attr, "source": "landxml_alignment"}


def maybe_swap_en(xy: list[tuple[float, float]], force_swap: bool | None = None) -> tuple[list[tuple[float, float]], bool]:
    """
    Heuristic:
      L-EST97: Northing ~ 6.x million, Easting ~ 0.5-0.8 million
      In your case, axis file sometimes comes swapped: first number ~ 6.5M (should be N but is in 'E' column)
    If force_swap is True -> swap always
    If force_swap is False -> never swap
    If None -> heuristic swap
    """
    if not xy:
        return xy, False

    arr = np.array(xy, dtype=float)
    a_med = float(np.median(arr[:, 0]))
    b_med = float(np.median(arr[:, 1]))

    # If first looks like Northing (millions) and second looks like Easting (hundreds of thousands) -> it is actually N,E, so swap to E,N
    heur_swap = (a_med > 2_000_000.0 and b_med < 2_000_000.0)

    if force_swap is True:
        do = True
    elif force_swap is False:
        do = False
    else:
        do = heur_swap

    if do:
        return [(float(b), float(a)) for (a, b) in xy], True
    return [(float(a), float(b)) for (a, b) in xy], False


# -------------------------
# Coverage check (axis vs surface)
# -------------------------

def axis_surface_overlap_median(idx, axis_xy: list[tuple[float, float]], sample_n: int = 8) -> float | None:
    if not axis_xy:
        return None
    step = max(1, len(axis_xy) // sample_n)
    test = axis_xy[::step][:sample_n]
    dists = []
    for (ex, ny) in test:
        nn = nearest_point_xyz(idx, ex, ny, max_rings=10)
        if nn is None:
            continue
        dx = nn[0] - ex
        dy = nn[1] - ny
        dists.append((dx * dx + dy * dy) ** 0.5)
    if not dists:
        return None
    return float(np.median(dists))


# -------------------------
# Main view
# -------------------------

def render_projects_view():
    _ui_readability_css()

    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # Session defaults
    st.session_state.setdefault("active_project_id", None)

    # For Mahud tab
    st.session_state.setdefault("mahud_surface_key", None)
    st.session_state.setdefault("mahud_surface_bytes", None)
    st.session_state.setdefault("mahud_axis_xy", None)
    st.session_state.setdefault("mahud_axis_len_attr", None)
    st.session_state.setdefault("mahud_axis_source", None)
    st.session_state.setdefault("mahud_axis_swap", None)

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
            if st.button(proj["name"], use_container_width=True, key=f"projbtn_{proj['id']}"):
                st.session_state["active_project_id"] = proj["id"]

                # reset Mahud state on project switch
                st.session_state["mahud_surface_key"] = None
                st.session_state["mahud_surface_bytes"] = None
                st.session_state["mahud_axis_xy"] = None
                st.session_state["mahud_axis_len_attr"] = None
                st.session_state["mahud_axis_source"] = None
                st.session_state["mahud_axis_swap"] = None

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

        tabs = st.tabs(["Üldine", "Mahud", "Failid"])

        # ============================================================
        # TAB: Üldine (keep your minimal info)
        # ============================================================
        with tabs[0]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.markdown('<div class="kv"><b>Planeeritud väärtused (DB):</b></div>', unsafe_allow_html=True)

            p2 = get_project(pid)
            st.write(f"Planned length: **{float(p2.get('planned_length_m') or 0):.2f} m**")
            st.write(f"Planned area: **{float(p2.get('planned_area_m2') or 0):.3f} m²**")
            st.write(f"Planned volume: **{float(p2.get('planned_volume_m3') or 0):.3f} m³**")
            st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # TAB: Mahud (LandXML surface + imported axis)
        # ============================================================
        with tabs[1]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📐 Mahud (LandXML + telg Civil3D-st)")

            st.markdown('<div class="caption2">'
                        '1) Vali või laadi üles <b>pinnamudel</b> (LandXML TIN) &nbsp; '
                        '2) Impordi <b>telg</b> (Alignment/Polyline LandXML või CSV/TXT) &nbsp; '
                        '3) Sea parameetrid ja arvuta PK tabel.'
                        '</div>', unsafe_allow_html=True)

            st.markdown('<hr class="soft">', unsafe_allow_html=True)

            # ---------- 1) Surface: select existing from R2 or upload new ----------
            st.markdown("### 1) Pinnamudel (LandXML)")
            existing = _list_landxml_keys_for_project(s3, p["name"])
            existing_names = ["— vali olemasolev —"] + [f["name"] for f in existing]

            cA, cB = st.columns([2, 1])
            with cA:
                sel = st.selectbox("Vali varem üles laaditud LandXML", existing_names, index=0)
            with cB:
                st.caption("Kui valikut pole, laadi uus üles.")

            up = st.file_uploader("...või laadi uus pinnamudel", type=["xml", "landxml"], key="mahud_surface_upload")

            def _load_surface_from_key(key: str):
                b = download_bytes(s3, key)
                st.session_state["mahud_surface_key"] = key
                st.session_state["mahud_surface_bytes"] = b

            def _upload_and_load_surface(uploaded):
                prefix = project_prefix(p["name"]) + "landxml/"
                key = upload_file(s3, prefix, uploaded)
                b = download_bytes(s3, key)
                st.session_state["mahud_surface_key"] = key
                st.session_state["mahud_surface_bytes"] = b

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("Kasuta valitud pinnamudelit", use_container_width=True):
                    if sel and sel != "— vali olemasolev —":
                        key = [f["key"] for f in existing if f["name"] == sel][0]
                        _load_surface_from_key(key)
                        st.success("Pinnamudel laaditud.")
                        st.rerun()
                    else:
                        st.warning("Vali olemasolev LandXML või laadi uus üles.")

            with col_btn2:
                if st.button("Lae uus pinnamudel üles & kasuta", use_container_width=True):
                    if up is None:
                        st.warning("Vali fail.")
                    else:
                        _upload_and_load_surface(up)
                        st.success("Pinnamudel salvestatud R2 ja laaditud.")
                        st.rerun()

            surface_bytes = st.session_state.get("mahud_surface_bytes")
            surface_key = st.session_state.get("mahud_surface_key")

            if surface_key:
                st.caption(f"Pinnamudel: {os.path.basename(surface_key)}")

            # Parse surface for quick preview + index
            idx = None
            if surface_bytes:
                try:
                    pts_dict, faces = read_landxml_tin_from_bytes(surface_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)  # (E,N,Z)
                    xyz2 = _downsample_points(xyz, max_pts=20000)

                    x_min = float(np.nanmin(xyz2[:, 0])); x_max = float(np.nanmax(xyz2[:, 0]))
                    y_min = float(np.nanmin(xyz2[:, 1])); y_max = float(np.nanmax(xyz2[:, 1]))
                    st.caption(f"ABS E min/max: {x_min:.3f} / {x_max:.3f} | ABS N min/max: {y_min:.3f} / {y_max:.3f}")

                    idx = build_tin_index(pts_dict, faces)
                except Exception as e:
                    st.error(f"Pinnamudeli lugemine ebaõnnestus: {e}")
                    idx = None

            st.markdown('<hr class="soft">', unsafe_allow_html=True)

            # ---------- 2) Axis import ----------
            st.markdown("### 2) Telg (Civil3D Alignment/Polyline)")
            axis_up = st.file_uploader(
                "Laadi teljefail (.xml/.landxml või .csv/.txt punktidega)",
                type=["xml", "landxml", "csv", "txt"],
                key="mahud_axis_upload",
            )

            c1, c2, c3 = st.columns([1.4, 1.2, 1.4])

            with c1:
                swap_checkbox = st.checkbox(
                    "Telje koordinaadid on vahetuses (N=Easting ja E=Northing) → Swap E/N",
                    value=bool(st.session_state.get("mahud_axis_swap") or False),
                )

            with c2:
                use_heur = st.checkbox("Kasuta automaatset E/N tuvastust", value=True)

            with c3:
                snap_axis = st.checkbox("Snap telg TIN-i lähimale tipule (soovi korral)", value=False)

            if st.button("Impordi telg", use_container_width=True):
                if axis_up is None:
                    st.warning("Vali teljefail.")
                else:
                    try:
                        parsed = read_axis_from_civil3d_file(axis_up.getvalue(), axis_up.name)
                        xy_raw = parsed["xy"]
                        length_attr = parsed["length_attr"]
                        source = parsed["source"]

                        # swap logic
                        force = True if swap_checkbox else (None if use_heur else False)
                        xy_fixed, did_swap = maybe_swap_en(xy_raw, force_swap=force)

                        st.session_state["mahud_axis_xy"] = xy_fixed
                        st.session_state["mahud_axis_len_attr"] = length_attr
                        st.session_state["mahud_axis_source"] = source
                        st.session_state["mahud_axis_swap"] = did_swap if force is None else bool(swap_checkbox)

                        st.success("Telg imporditud.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Telje import ebaõnnestus: {e}")

            axis_xy = st.session_state.get("mahud_axis_xy") or []
            axis_len_attr = st.session_state.get("mahud_axis_len_attr")
            axis_source = st.session_state.get("mahud_axis_source")

            if axis_xy:
                # choose displayed length:
                L_poly = float(polyline_length(axis_xy))
                if axis_len_attr is not None and axis_len_attr > 0:
                    L_show = float(axis_len_attr)
                    st.caption(f"Telg: {axis_source} | Punkte: {len(axis_xy)} | Pikkus (Alignment length): {L_show:.2f} m")
                else:
                    L_show = L_poly
                    st.caption(f"Telg: {axis_source} | Punkte: {len(axis_xy)} | Pikkus (polyline): {L_show:.2f} m")

                with st.expander("Näita telje esimesed 10 punkti"):
                    df_axis = pd.DataFrame(axis_xy[:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)

            st.markdown('<hr class="soft">', unsafe_allow_html=True)

            # ---------- 3) Compute parameters ----------
            st.markdown("### 3) Arvutusparameetrid")

            c1, c2, c3 = st.columns(3)
            with c1:
                pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
                cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=17.0, step=1.0)
                sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.10, step=0.01)
            with c2:
                tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005)
                min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.20, step=0.05)
                min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.30, step=0.05)
            with c3:
                slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2")
                bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05)

            # ---------- Compute ----------
            if st.button("Arvuta PK tabel", use_container_width=True):
                if not surface_bytes:
                    st.warning("Vali või laadi pinnamudel (LandXML).")
                elif not axis_xy or len(axis_xy) < 2:
                    st.warning("Impordi telg (vähemalt 2 punkti).")
                else:
                    # optional snap axis to TIN vertices
                    axis_use = axis_xy
                    if snap_axis and idx is not None:
                        snapped = []
                        for (ex, ny) in axis_xy:
                            nn = nearest_point_xyz(idx, ex, ny, max_rings=10)
                            if nn is None:
                                snapped.append((ex, ny))
                            else:
                                snapped.append((float(nn[0]), float(nn[1])))
                        axis_use = snapped

                    # coverage check
                    if idx is not None:
                        d_med = axis_surface_overlap_median(idx, axis_use, sample_n=8)
                        if d_med is None:
                            st.warning("Ei suutnud kontrollida telje kattuvust TIN-iga (nearest puudub).")
                        else:
                            st.caption(f"Telje ja pinnamudeli kattuvus (mediaan kaugus): {d_med:.2f} m")
                            if d_med > 20:
                                st.warning(
                                    "⚠️ Telg on pinnamudelist kaugel. "
                                    "Tõenäoliselt vale E/N swap või erinev koordinaatsüsteem / vale pinnamudel."
                                )

                    try:
                        res = compute_pk_table_from_landxml(
                            xml_bytes=surface_bytes,
                            axis_xy_abs=axis_use,   # IMPORTANT: matches landxml.py signature
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

                        # planned area from volume/length
                        planned_area = None
                        if res["axis_length_m"] > 0 and res["total_volume_m3"] is not None:
                            planned_area = float(res["total_volume_m3"] / res["axis_length_m"])

                        # save to DB
                        key = surface_key or (p.get("landxml_key") or "")
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

                        # quick diagnostic: how many None rows
                        none_count = int(df["Vol_m3"].isna().sum()) if "Vol_m3" in df.columns else 0
                        if none_count > 0:
                            st.info(f"Diagnoos: {none_count} PK rida ei suutnud leida serva/põhja (Vol_m3=None). "
                                    f"Proovi suurendada ristlõike pikkust või tol/min_run parameetreid.")

                    except Exception as e:
                        st.error(f"Arvutus ebaõnnestus: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # TAB: Failid (R2 file upload / list) - keep as you had
        # ============================================================
        with tabs[2]:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📤 Failid (Cloudflare R2)")

            uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
            if uploads:
                prefix = project_prefix(p["name"])
                for upf in uploads:
                    upload_file(s3, prefix, upf)
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
