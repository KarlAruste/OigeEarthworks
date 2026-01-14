# views/projects_view.py
import io
import re
import csv
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd

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
    compute_pk_table_from_landxml,
    polyline_length,
)


# -------------------------
# Small UI helpers
# -------------------------
def _inject_tabs_css():
    st.markdown(
        """
<style>
/* Tabs text readable in dark theme */
div[data-testid="stTabs"] button[role="tab"] p {
  color: #e5e7eb !important;
  font-weight: 600;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
  color: #000 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ff8a00 !important;
  border-radius: 10px !important;
}
div[data-testid="stTabs"] button[role="tab"] {
  background: #1b1f2a !important;
  border-radius: 10px !important;
  margin-right: 6px !important;
}
.small { color:#cbd5e1; font-size:13px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _downsample_points(xy: np.ndarray, max_pts: int = 60000) -> np.ndarray:
    if xy is None or xy.size == 0:
        return xy
    if xy.shape[0] <= max_pts:
        return xy
    idx = np.random.choice(xy.shape[0], size=max_pts, replace=False)
    return xy[idx]


# -------------------------
# Axis import (CSV/TXT/XML)
# -------------------------
def _heuristic_swap_en(E: np.ndarray, N: np.ndarray) -> bool:
    """Returns True if looks like inputs are swapped (N in 0.6M, E in 6.5M)."""
    if E.size == 0 or N.size == 0:
        return False
    Em = float(np.nanmedian(E))
    Nm = float(np.nanmedian(N))
    # Estonia typical: E ~ 300k..800k, N ~ 6M..7M
    # If reversed -> swap
    return (Em > 2_000_000 and Nm < 2_000_000)


def _parse_axis_csv_bytes(data: bytes) -> list[tuple[float, float]]:
    text = data.decode("utf-8", errors="ignore")
    # try detect delimiter
    # allow ; , \t whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Telje fail on tühi.")

    # sniff delimiter
    delim = ";"
    if "," in lines[0] and ";" not in lines[0]:
        delim = ","
    elif "\t" in lines[0]:
        delim = "\t"

    # read rows
    out = []
    reader = csv.reader(lines, delimiter=delim)
    for row in reader:
        if not row:
            continue
        # join if weird spacing
        row2 = []
        for c in row:
            c = c.strip()
            if not c:
                continue
            row2.append(c)
        if len(row2) < 2:
            continue

        # skip header-like
        if re.search(r"[A-Za-z]", row2[0]) or re.search(r"[A-Za-z]", row2[1]):
            continue

        try:
            a = float(row2[0].replace(",", "."))
            b = float(row2[1].replace(",", "."))
            out.append((a, b))
        except Exception:
            continue

    if len(out) < 2:
        raise ValueError("CSV/TXT-st ei leitud vähemalt 2 koordinaatpunkti (2 veergu).")
    return out


def _parse_axis_xml_bytes(data: bytes) -> list[tuple[float, float]]:
    """
    Very tolerant XML parser:
    - reads any <P> that contains "x y z" or "a b z"
    - reads any <CgPoint> that contains "x y z"
    Takes first two numbers as (a,b) and later we can swap.
    """
    root = ET.fromstring(data)

    pts = []

    def take_numbers(txt: str):
        if not txt:
            return None
        parts = txt.replace(",", " ").split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass
        return nums

    # <P>...</P>
    for p in root.iter():
        ln = p.tag.split("}")[-1].lower()
        if ln in ("p", "cgpoint"):
            nums = take_numbers((p.text or "").strip())
            if not nums or len(nums) < 2:
                continue
            pts.append((float(nums[0]), float(nums[1])))

    if len(pts) < 2:
        raise ValueError("XML-ist ei leitud telje punkte (<P> või <CgPoint>). Ekspordi telg CSV-na või tee lihtne XML punktidega.")
    return pts


# ET is used in XML parsing here
import xml.etree.ElementTree as ET


def read_axis_points_from_upload(uploaded_file, swap_en: bool = True) -> list[tuple[float, float]]:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if name.endswith(".csv") or name.endswith(".txt"):
        pts = _parse_axis_csv_bytes(data)
    elif name.endswith(".xml") or name.endswith(".landxml"):
        pts = _parse_axis_xml_bytes(data)
    else:
        raise ValueError("Telje fail peab olema CSV/TXT/XML/LANDXML.")

    arr = np.array(pts, dtype=float)
    A = arr[:, 0]
    B = arr[:, 1]

    # If swap_en enabled: treat file columns as (N,E) and swap to (E,N)
    # This matches your statement: "N is actually easting and E is northing"
    if swap_en:
        E = B
        N = A
    else:
        E = A
        N = B

    # final heuristic safety (if still swapped)
    if _heuristic_swap_en(E, N):
        E, N = N, E

    out = [(float(e), float(n)) for e, n in zip(E, N)]

    # remove consecutive duplicates
    cleaned = []
    for p in out:
        if not cleaned or (abs(cleaned[-1][0] - p[0]) > 1e-9 or abs(cleaned[-1][1] - p[1]) > 1e-9):
            cleaned.append(p)

    if len(cleaned) < 2:
        raise ValueError("Telg jäi pärast puhastust liiga lühikeseks.")
    return cleaned


# -------------------------
# LandXML list helpers
# -------------------------
def list_project_landxml_keys(s3, project_name: str) -> list[dict]:
    """Return list_files under landxml/ prefix only."""
    prefix = project_prefix(project_name) + "landxml/"
    files = list_files(s3, prefix)
    # keep only xml/landxml
    out = []
    for f in files:
        nm = f.get("name", "")
        low = nm.lower()
        if low.endswith(".xml") or low.endswith(".landxml"):
            out.append(f)
    return out


# -------------------------
# Main view
# -------------------------
def render_projects_view():
    _inject_tabs_css()

    st.title("Projects")
    s3 = get_s3()
    projects = list_projects()

    st.session_state.setdefault("active_project_id", None)

    # per-project cached selections
    st.session_state.setdefault("mahud_selected_landxml_key", None)
    st.session_state.setdefault("mahud_axis_points", None)  # list[(E,N)]
    st.session_state.setdefault("mahud_axis_name", None)

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
                # reset per-project mahud state
                st.session_state["mahud_selected_landxml_key"] = None
                st.session_state["mahud_axis_points"] = None
                st.session_state["mahud_axis_name"] = None
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

        tab_general, tab_mahud, tab_files = st.tabs(["Üldine", "Mahud", "Failid"])

        # ==============
        # ÜLDINE
        # ==============
        with tab_general:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.write("Siin võid hiljem lisada projekti üldinfo, KPI-d jne.")
            p2 = get_project(pid)
            if p2:
                st.markdown("**Salvestatud planeeritud väärtused (DB):**")
                st.write(f"Planned length: **{float(p2.get('planned_length_m') or 0):.2f} m**")
                st.write(f"Planned area: **{float(p2.get('planned_area_m2') or 0):.3f} m²**")
                st.write(f"Planned volume: **{float(p2.get('planned_volume_m3') or 0):.3f} m³**")
                st.caption(f"LandXML key: {p2.get('landxml_key') or '-'}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ==============
        # MAHUD (LandXML + telg import + arvutus)
        # ==============
        with tab_mahud:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📐 Mahud (LandXML + telg Civil3D-st)")
            st.caption("Töövoog: 1) vali/üleslae LandXML → 2) impordi telg (polyline/alignment punktid) → 3) arvuta PK tabel & maht.")

            # ---- 1) LandXML choose or upload ----
            st.markdown("### 1) Pinnamudel (LandXML)")

            landxml_files = list_project_landxml_keys(s3, p["name"])
            options = ["— vali olemasolev —"] + [f["name"] for f in landxml_files]

            # preselect saved key if present
            saved_key = p.get("landxml_key")
            pre_name = None
            if saved_key:
                for f in landxml_files:
                    if f.get("key") == saved_key:
                        pre_name = f.get("name")
                        break

            default_idx = 0
            if st.session_state.get("mahud_selected_landxml_key") and landxml_files:
                # map key to name
                for i, f in enumerate(landxml_files, start=1):
                    if f["key"] == st.session_state["mahud_selected_landxml_key"]:
                        default_idx = i
                        break
            elif pre_name:
                for i, f in enumerate(landxml_files, start=1):
                    if f["name"] == pre_name:
                        default_idx = i
                        break

            chosen_name = st.selectbox("Vali varem üleslaetud LandXML", options, index=default_idx)

            col_u1, col_u2 = st.columns([2, 1])
            with col_u1:
                landxml_up = st.file_uploader("…või laadi uus LandXML (.xml/.landxml)", type=["xml", "landxml"], key="mahud_landxml_upload")
            with col_u2:
                do_upload = st.button("Salvesta LandXML R2", use_container_width=True, disabled=(landxml_up is None))

            if do_upload and landxml_up is not None:
                prefix = project_prefix(p["name"]) + "landxml/"
                key = upload_file(s3, prefix, landxml_up)
                st.success("LandXML üles laetud.")
                st.session_state["mahud_selected_landxml_key"] = key
                # also store to DB immediately as default
                set_project_landxml(
                    p["id"],
                    landxml_key=key,
                    planned_volume_m3=p.get("planned_volume_m3"),
                    planned_length_m=p.get("planned_length_m"),
                    planned_area_m2=p.get("planned_area_m2"),
                )
                st.rerun()

            # resolve chosen landxml key
            selected_key = None
            if chosen_name and chosen_name != "— vali olemasolev —":
                for f in landxml_files:
                    if f["name"] == chosen_name:
                        selected_key = f["key"]
                        break

            # session key wins if set
            if st.session_state.get("mahud_selected_landxml_key"):
                selected_key = st.session_state["mahud_selected_landxml_key"]

            xml_bytes = None
            if selected_key:
                try:
                    xml_bytes = download_bytes(s3, selected_key)
                    st.success(f"Valitud LandXML: {selected_key.split('/')[-1]}")
                except Exception as e:
                    st.error(f"LandXML laadimine ebaõnnestus: {e}")
                    xml_bytes = None
            else:
                st.info("Vali LandXML (või laadi uus), et jätkata.")

            # ---- show point stats quick ----
            if xml_bytes:
                try:
                    pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)
                    finite = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1])
                    xyz = xyz[finite]
                    st.caption(
                        f"Punkte (finite E,N): {xyz.shape[0]:,} | "
                        f"E min/max: {float(np.min(xyz[:,0])):.3f} / {float(np.max(xyz[:,0])):.3f} | "
                        f"N min/max: {float(np.min(xyz[:,1])):.3f} / {float(np.max(xyz[:,1])):.3f}"
                    )
                except Exception as e:
                    st.error(f"LandXML lugemine ebaõnnestus: {e}")
                    xml_bytes = None

            st.divider()

            # ---- 2) Axis import ----
            st.markdown("### 2) Telg (Civil3D polyline/alignment → punktid)")
            st.caption("Lae siia telje punktid. Soovitus: ekspordi Civil3D-st punktid CSV-na (2 veergu).")

            swap_en = st.checkbox(
                "Telje koordinaadid on vahetuses (N=Easting ja E=Northing) → Swap E/N",
                value=True,
            )

            axis_up = st.file_uploader(
                "Laadi teljefail (.csv/.txt punktidega või .xml/.landxml)",
                type=["csv", "txt", "xml", "landxml"],
                key="mahud_axis_upload",
            )

            col_a1, col_a2 = st.columns([1, 2])
            with col_a1:
                import_axis = st.button("Impordi telg", use_container_width=True, disabled=(axis_up is None))
            with col_a2:
                clear_axis = st.button("Tühjenda telg", use_container_width=True)

            if clear_axis:
                st.session_state["mahud_axis_points"] = None
                st.session_state["mahud_axis_name"] = None
                st.rerun()

            if import_axis and axis_up is not None:
                try:
                    axis_pts = read_axis_points_from_upload(axis_up, swap_en=swap_en)
                    st.session_state["mahud_axis_points"] = axis_pts
                    st.session_state["mahud_axis_name"] = axis_up.name
                    st.success(f"Telg imporditud. Punkte: {len(axis_pts)} | Pikkus: {polyline_length(axis_pts):.2f} m")
                    st.rerun()
                except Exception as e:
                    st.error(f"Telje import ebaõnnestus: {e}")

            axis_pts = st.session_state.get("mahud_axis_points")
            axis_name = st.session_state.get("mahud_axis_name")

            if axis_pts:
                st.write(f"**Telg:** {axis_name or '-'} | Punkte: **{len(axis_pts)}** | Pikkus: **{polyline_length(axis_pts):.2f} m**")
                with st.expander("Näita telje esimesed 10 punkti"):
                    df_axis = pd.DataFrame(axis_pts[:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)
            else:
                st.info("Impordi telg, et arvutada mahud.")

            st.divider()

            # ---- 3) Compute ----
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

            can_compute = (xml_bytes is not None) and (axis_pts is not None) and (len(axis_pts) >= 2)
            compute_btn = st.button("Arvuta PK tabel", use_container_width=True, disabled=(not can_compute))

            if compute_btn:
                try:
                    res = compute_pk_table_from_landxml(
                        xml_bytes=xml_bytes,
                        axis_xy_abs=axis_pts,  # ABS (E,N)
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
                    if res["axis_length_m"] and res["axis_length_m"] > 0 and res["total_volume_m3"] is not None:
                        planned_area = float(res["total_volume_m3"] / res["axis_length_m"])

                    # save selected landxml as current and planned_* to DB
                    set_project_landxml(
                        p["id"],
                        landxml_key=selected_key or (p.get("landxml_key") or ""),
                        planned_volume_m3=float(res["total_volume_m3"]),
                        planned_length_m=float(res["axis_length_m"]),
                        planned_area_m2=planned_area,
                    )

                    st.dataframe(df, use_container_width=True)

                    csv_bytes = df.to_csv(index=False, sep=";").encode("utf-8")
                    st.download_button(
                        "⬇️ Lae alla CSV",
                        data=csv_bytes,
                        file_name=f"{p['name']}_pk_tabel.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.info("Tulemused salvestati projekti planned_* väljade alla (DB).")

                except Exception as e:
                    st.error(f"Arvutus ebaõnnestus: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ==============
        # FAILID (R2 upload + list)
        # ==============
        with tab_files:
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
