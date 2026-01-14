# views/projects_view.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import traceback

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


# -------------------------
# Small helpers
# -------------------------

def _row_get(r, key, default=None):
    """Works if DB returns dict rows OR tuples (safety)."""
    if isinstance(r, dict):
        return r.get(key, default)
    return default


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


def _origin_abs(xyz_abs: np.ndarray) -> tuple[float, float]:
    x0 = float(np.nanmean(xyz_abs[:, 0]))
    y0 = float(np.nanmean(xyz_abs[:, 1]))
    return (x0, y0)


def _abs_to_local(xyz_abs: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    x0, y0 = origin
    out = xyz_abs.copy()
    out[:, 0] = out[:, 0] - x0
    out[:, 1] = out[:, 1] - y0
    return out


def _nearest_point_abs_from_local(local_xy, xyz_show_abs, origin_abs_):
    """Snap click to nearest displayed point (ABS)."""
    x0, y0 = origin_abs_
    cx_abs = float(local_xy[0] + x0)
    cy_abs = float(local_xy[1] + y0)

    X = xyz_show_abs[:, 0].astype(float)
    Y = xyz_show_abs[:, 1].astype(float)
    m = np.isfinite(X) & np.isfinite(Y)
    X = X[m]
    Y = Y[m]
    if X.size == 0:
        return (cx_abs, cy_abs)

    dx = X - cx_abs
    dy = Y - cy_abs
    j = int(np.argmin(dx * dx + dy * dy))
    return (float(X[j]), float(Y[j]))


def _ne_to_en_points(points_ne):
    """
    Failides eeldame alati N,E.
    Tagastame E,N.
    """
    out = []
    for a, b in points_ne:
        n = float(a)
        e = float(b)
        out.append((e, n))
    return out


def _parse_axis_points_from_bytes(file_name: str, b: bytes):
    """
    Toetab:
      - Alignment LandXML (Alignments/CoordGeom/Line/Start/End, Curve/PI)
      - lihtne CSV/TXT: kaks veergu N;E või N E (või koma/semikoolon)
    EELDUS: alati N E järjekord failis.
    Tagastab: axis_xy_abs list[(E,N)]
    """
    name = (file_name or "").lower()

    # 1) CSV/TXT
    if name.endswith(".csv") or name.endswith(".txt"):
        text = b.decode("utf-8", errors="ignore")
        pts_ne = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            # allow header lines
            if any(k in s.lower() for k in ["north", "easting", "n;", "e;", "n,", "e,"]):
                continue
            # split by ; , space
            for sep in [";", ",", "\t"]:
                s = s.replace(sep, " ")
            parts = [p for p in s.split(" ") if p]
            if len(parts) < 2:
                continue
            try:
                n = float(parts[0])
                e = float(parts[1])
                pts_ne.append((n, e))
            except Exception:
                continue
        axis = _ne_to_en_points(pts_ne)
        return axis

    # 2) XML / LandXML Alignment
    # parse with ElementTree (no lxml dependency here)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(b)

    def _strip(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # Collect <Start>, <End>, <PI> under CoordGeom
    pts_ne = []

    for el in root.iter():
        t = _strip(el.tag).lower()
        if t in ("start", "end", "pi"):
            if el.text and el.text.strip():
                parts = el.text.strip().split()
                if len(parts) >= 2:
                    try:
                        n = float(parts[0])
                        e = float(parts[1])
                        pts_ne.append((n, e))
                    except Exception:
                        pass

    axis = _ne_to_en_points(pts_ne)

    # remove consecutive duplicates
    cleaned = []
    last = None
    for p in axis:
        if last is None or (abs(p[0]-last[0]) > 1e-9 or abs(p[1]-last[1]) > 1e-9):
            cleaned.append(p)
            last = p
    return cleaned


# -------------------------
# Canvas component (click-to-pick)
# -------------------------
def canvas_pick_point(points_local_xy: np.ndarray, axis_local_xy: list[tuple[float, float]]):
    import json
    import streamlit.components.v1 as components

    pts = points_local_xy[:, :2].astype(float).tolist()
    axis = [(float(a), float(b)) for (a, b) in axis_local_xy]

    payload = {"points": pts, "axis": axis}

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin:0; padding:0; height:100%; width:100%; overflow:hidden; background:#fff; }}
    #wrap {{ position:relative; height:100%; width:100%; }}
    #hud {{ position:absolute; left:12px; top:10px; background:rgba(255,255,255,0.92); padding:8px 10px; border:1px solid #ddd; border-radius:8px; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; font-size:13px; }}
    #hud b {{ font-weight:600; }}
    canvas {{ display:block; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="hud">
      <div><b>Canvas</b> — Pan: lohista | Zoom: rullik | Kliki: vali lähim punkt</div>
      <div id="info">—</div>
    </div>
    <canvas id="c"></canvas>
  </div>

<script>
  const data = {json.dumps(payload)};
  const pts: number[][] = data.points;
  const axis: number[][] = data.axis;

  const canvas = document.getElementById('c');
  const info = document.getElementById('info');
  const ctx = canvas.getContext('2d');

  function resize() {{
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }}

  function bounds() {{
    let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
    for (const p of pts) {{
      const x = p[0], y = p[1];
      if (x<xmin) xmin=x; if (x>xmax) xmax=x;
      if (y<ymin) ymin=y; if (y>ymax) ymax=y;
    }}
    if (!isFinite(xmin)) {{ xmin=-1; xmax=1; ymin=-1; ymax=1; }}
    return {{xmin,xmax,ymin,ymax}};
  }}

  let view = {{ scale: 1, ox: 0, oy: 0 }};

  function fit() {{
    const b = bounds();
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const dx = (b.xmax-b.xmin) || 1;
    const dy = (b.ymax-b.ymin) || 1;
    const sx = (w*0.85) / dx;
    const sy = (h*0.85) / dy;
    view.scale = Math.min(sx, sy);
    const cx = (b.xmin+b.xmax)/2;
    const cy = (b.ymin+b.ymax)/2;
    view.ox = w/2 - cx*view.scale;
    view.oy = h/2 + cy*view.scale; // y flip
  }}

  function worldToScreen(x,y) {{
    const sx = x*view.scale + view.ox;
    const sy = -y*view.scale + view.oy;
    return [sx,sy];
  }}

  function screenToWorld(sx,sy) {{
    const x = (sx - view.ox)/view.scale;
    const y = -(sy - view.oy)/view.scale;
    return [x,y];
  }}

  function draw() {{
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0,0,w,h);

    ctx.save();
    ctx.strokeStyle = '#eef2f6';
    ctx.lineWidth = 1;
    const step = 50;
    for (let x=0; x<=w; x+=step) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }}
    for (let y=0; y<=h; y+=step) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }}
    ctx.restore();

    // points
    ctx.save();
    ctx.fillStyle = 'rgba(75, 110, 255, 0.85)';
    for (let i=0; i<pts.length; i++) {{
      const x = pts[i][0], y = pts[i][1];
      const p = worldToScreen(x,y);
      const sx = p[0], sy = p[1];
      if (sx<-10 || sy<-10 || sx>w+10 || sy>h+10) continue;
      ctx.beginPath();
      ctx.arc(sx,sy,2.4,0,Math.PI*2);
      ctx.fill();
    }}
    ctx.restore();

    // axis polyline
    if (axis && axis.length>0) {{
      ctx.save();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i=0; i<axis.length; i++) {{
        const x = axis[i][0], y = axis[i][1];
        const p = worldToScreen(x,y);
        if (i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]);
      }}
      ctx.stroke();

      ctx.fillStyle = '#f59e0b';
      for (let i=0; i<axis.length; i++) {{
        const x = axis[i][0], y = axis[i][1];
        const p = worldToScreen(x,y);
        ctx.beginPath(); ctx.arc(p[0],p[1],5,0,Math.PI*2); ctx.fill();
      }}
      ctx.restore();
    }}
  }}

  let dragging=false;
  let lastX=0, lastY=0;

  canvas.addEventListener('mousedown', (e) => {{
    dragging = true;
    lastX = e.offsetX;
    lastY = e.offsetY;
  }});

  window.addEventListener('mouseup', () => {{ dragging=false; }});

  canvas.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    const dx = e.offsetX - lastX;
    const dy = e.offsetY - lastY;
    lastX = e.offsetX;
    lastY = e.offsetY;
    view.ox += dx;
    view.oy += dy;
    draw();
  }});

  canvas.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const mouseX = e.offsetX;
    const mouseY = e.offsetY;
    const wpos = screenToWorld(mouseX, mouseY);
    const wx = wpos[0], wy = wpos[1];

    const zoom = Math.exp(-e.deltaY * 0.001);
    const newScale = Math.min(2000, Math.max(0.01, view.scale * zoom));

    view.ox = mouseX - wx*newScale;
    view.oy = mouseY + wy*newScale;
    view.scale = newScale;
    draw();
  }}, {{ passive:false }});

  function sendValue(val) {{
    window.parent.postMessage({{
      isStreamlitMessage: true,
      type: 'streamlit:setComponentValue',
      value: val
    }}, '*');
  }}

  function nearestPoint(wx, wy) {{
    let best=-1; let bestD=Infinity;
    for (let i=0; i<pts.length; i++) {{
      const dx = pts[i][0]-wx;
      const dy = pts[i][1]-wy;
      const d = dx*dx + dy*dy;
      if (d < bestD) {{ bestD=d; best=i; }}
    }}
    return best;
  }}

  canvas.addEventListener('click', (e) => {{
    const wpos = screenToWorld(e.offsetX, e.offsetY);
    const wx = wpos[0], wy = wpos[1];
    const idx = nearestPoint(wx, wy);
    if (idx < 0) return;
    const px = pts[idx][0];
    const py = pts[idx][1];
    info.textContent = `Valitud (local): x=${{px.toFixed(3)}}, y=${{py.toFixed(3)}}`;
    sendValue({{picked:true, x:px, y:py}});
  }});

  function setSize() {{
    canvas.style.width = '100%';
    canvas.style.height = '650px';
    resize();
  }}

  setSize();
  fit();
  draw();
  window.addEventListener('resize', () => {{ resize(); }});
  window.parent.postMessage({{ isStreamlitMessage:true, type:'streamlit:setFrameHeight', height: 670 }}, '*');
</script>
</body>
</html>
    """
    return components.html(html, height=670, scrolling=False)


# -------------------------
# Main view
# -------------------------
def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects() or []

    # Session defaults
    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("axis_xy", [])          # ABS [(E,N)]
    st.session_state.setdefault("axis_finished", False)
    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)
    st.session_state.setdefault("plot_origin", None)    # ABS (E0,N0)
    st.session_state.setdefault("xyz_show_cache", None) # ABS points for display/snap

    # ---------------- Create project ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Loo projekt")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Projekti nimi", placeholder="nt Tapa_Objekt_01")
    with c2:
        start = st.date_input("Algus (valikuline)", value=date.today())
        use_start = st.checkbox("Kasuta alguskuupäeva", value=False)
    with c3:
        end = st.date_input("Lõpp (tähtaeg)", value=date.today())

    if st.button("Loo projekt", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta projekti nimi.")
        else:
            create_project(name.strip(), start_date=(start if use_start else None), end_date=end)
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
            pid = _row_get(proj, "id")
            pname = _row_get(proj, "name", "Project")
            if st.button(str(pname), use_container_width=True, key=f"projbtn_{pid}_{pname}"):
                st.session_state["active_project_id"] = pid
                st.session_state["axis_xy"] = []
                st.session_state["axis_finished"] = False
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["plot_origin"] = None
                st.session_state["xyz_show_cache"] = None
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

        pname = _row_get(p, "name", "Project")
        st.subheader(str(pname))
        st.caption(f"Tähtaeg: {_row_get(p, 'end_date')}")

        tab_general, tab_volumes, tab_files = st.tabs(["Üldine", "Mahud", "Failid"])

        # ---------------- Üldine ----------------
        with tab_general:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("Üldinfo")
            st.write(f"**Projekt:** {pname}")
            st.write(f"**Algus:** {_row_get(p,'start_date')}")
            st.write(f"**Lõpp:** {_row_get(p,'end_date')}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("Salvestatud planeeritud väärtused (DB)")
            pl = _row_get(p, "planned_length_m")
            pa = _row_get(p, "planned_area_m2")
            pv = _row_get(p, "planned_volume_m3")
            st.write(f"**Planned length:** {_safe_float(pl, 0.0):.2f} m")
            st.write(f"**Planned area:** {_safe_float(pa, 0.0):.3f} m²")
            st.write(f"**Planned volume:** {_safe_float(pv, 0.0):.3f} m³")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Mahud ----------------
        with tab_volumes:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("1) LandXML pind (salvesta + vali)")

            landxml_prefix = project_prefix(pname) + "landxml/"
            landxml_files = list_files(s3, landxml_prefix) or []
            landxml_files = [f for f in landxml_files if f["name"].lower().endswith((".xml", ".landxml"))]

            up = st.file_uploader("Laadi üles LandXML pind (.xml/.landxml)", type=["xml", "landxml"], key="vol_landxml_up")

            cA, cB = st.columns([1, 2])
            with cA:
                if up and st.button("Salvesta LandXML R2", use_container_width=True):
                    key = upload_file(s3, landxml_prefix, up)
                    st.success(f"Salvestatud: {key}")
                    st.rerun()

            with cB:
                options = ["— vali R2-st —"] + [f["key"] for f in landxml_files]
                labels = {"— vali R2-st —": "— vali R2-st —"}
                for f in landxml_files:
                    labels[f["key"]] = f["name"]

                sel_key = st.selectbox(
                    "Vali varem üleslaetud LandXML pind",
                    options=options,
                    format_func=lambda k: labels.get(k, k),
                    index=0 if not _row_get(p, "landxml_key") else (options.index(_row_get(p, "landxml_key")) if _row_get(p, "landxml_key") in options else 0)
                )

                if sel_key != "— vali R2-st —" and st.button("Ava valitud LandXML", use_container_width=True):
                    xml_bytes = download_bytes(s3, sel_key)
                    st.session_state["landxml_bytes"] = xml_bytes
                    st.session_state["landxml_key"] = sel_key
                    st.session_state["axis_xy"] = []
                    st.session_state["axis_finished"] = False
                    st.session_state["plot_origin"] = None
                    st.session_state["xyz_show_cache"] = None
                    st.success("LandXML laaditud sessiooni.")
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Axis import (no checkbox; ALWAYS assume N E) ----
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("2) Telg / Alignment (failist)")

            axis_up = st.file_uploader(
                "Laadi teljefail (CSV/TXT punktidega või Alignment LandXML)",
                type=["csv", "txt", "xml", "landxml"],
                key="axis_up",
            )

            st.caption("Eeldus: failides on koordinaadid alati järjekorras **N E** (Northing, Easting).")

            if axis_up and st.button("Impordi telg", use_container_width=True):
                try:
                    axis_xy = _parse_axis_points_from_bytes(axis_up.name, axis_up.getvalue())
                    if len(axis_xy) < 2:
                        st.warning("Telg peab sisaldama vähemalt 2 punkti.")
                    else:
                        st.session_state["axis_xy"] = axis_xy  # (E,N)
                        st.session_state["axis_finished"] = True
                        st.success(f"Telg imporditud: punkte {len(axis_xy)}")
                except Exception as e:
                    st.error(f"Telje import ebaõnnestus: {e}")
                    st.code(traceback.format_exc())
                st.rerun()

            if st.session_state.get("axis_xy"):
                L = polyline_length(st.session_state["axis_xy"])
                st.write(f"**Telje punktid:** {len(st.session_state['axis_xy'])} | **Pikkus:** {L:.2f} m")
                with st.expander("Näita telje esimesed 10 punkti (E,N)"):
                    df_axis = pd.DataFrame(st.session_state["axis_xy"][:10], columns=["E", "N"])
                    st.dataframe(df_axis, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Optional: show TIN points + allow click axis (kept) ----
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("3) Kontroll: LandXML punktid (vaade) + telg (kliki TIN-ist)")

            xml_bytes = st.session_state.get("landxml_bytes")
            if not xml_bytes:
                st.info("Lae LandXML (1. samm) enne, kui saab punkte kuvada.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                try:
                    pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
                    xyz = np.array(list(pts_dict.values()), dtype=float)  # expected (E,N,Z)
                    xyz_show = _downsample_points(xyz, max_pts=25000)
                    st.session_state["xyz_show_cache"] = xyz_show

                    if st.session_state["plot_origin"] is None:
                        st.session_state["plot_origin"] = _origin_abs(xyz_show)

                    x_min = float(np.nanmin(xyz_show[:, 0])); x_max = float(np.nanmax(xyz_show[:, 0]))
                    y_min = float(np.nanmin(xyz_show[:, 1])); y_max = float(np.nanmax(xyz_show[:, 1]))
                    st.caption(f"ABS E min/max: {x_min:.3f} / {x_max:.3f} | ABS N min/max: {y_min:.3f} / {y_max:.3f}")

                except Exception as e:
                    st.error(f"LandXML lugemine ebaõnnestus: {e}")
                    st.code(traceback.format_exc())
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # click-to-add axis points (optional)
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    if st.button("Alusta klikk-telge", use_container_width=True):
                        st.session_state["axis_xy"] = []
                        st.session_state["axis_finished"] = False
                        st.rerun()
                with c2:
                    if st.button("Undo", use_container_width=True):
                        if st.session_state["axis_xy"]:
                            st.session_state["axis_xy"] = st.session_state["axis_xy"][:-1]
                            st.session_state["axis_finished"] = False
                            st.rerun()
                with c3:
                    if st.button("Lõpeta klikk-telg", use_container_width=True):
                        if len(st.session_state["axis_xy"]) < 2:
                            st.warning("Telg peab olema vähemalt 2 punktiga.")
                        else:
                            st.session_state["axis_finished"] = True
                            st.success("Klikk-telg lõpetatud.")
                            st.rerun()

                origin = st.session_state["plot_origin"]
                xyz_show_abs = st.session_state["xyz_show_cache"]
                axis_xy_abs = st.session_state.get("axis_xy", [])
                finished = st.session_state.get("axis_finished", False)

                xyz_show_local = _abs_to_local(xyz_show_abs, origin)
                axis_local = [(a - origin[0], b - origin[1]) for (a, b) in axis_xy_abs]

                st.caption("Canvas: kliki — lisab teljele lähima TIN punkti (snap).")
                picked = canvas_pick_point(xyz_show_local, axis_local)

                if picked and isinstance(picked, dict) and picked.get("picked") and (not finished):
                    lx = float(picked.get("x"))
                    ly = float(picked.get("y"))
                    ax, ay = _nearest_point_abs_from_local((lx, ly), xyz_show_abs, origin)
                    st.session_state["axis_xy"] = axis_xy_abs + [(ax, ay)]
                    st.rerun()

                if axis_xy_abs:
                    L = polyline_length(axis_xy_abs)
                    st.write(f"**Klikk-telje pikkus:** {L:.2f} m | Punkte: {len(axis_xy_abs)}")

                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Compute PK table ----
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("4) Arvuta PK tabel")

            axis_xy = st.session_state.get("axis_xy", [])
            xml_bytes = st.session_state.get("landxml_bytes")

            if not xml_bytes or len(axis_xy) < 2:
                st.info("Vaja: LandXML (1) + telg (2) või klikk-telg (3).")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    pk_step = st.number_input("PK samm (m)", min_value=0.1, value=1.0, step=0.1)
                    cross_len = st.number_input("Ristlõike kogupikkus (m)", min_value=2.0, value=17.0, step=1.0)
                    sample_step = st.number_input("Proovipunkti samm (m)", min_value=0.02, value=0.1, step=0.01)
                with c2:
                    tol = st.number_input("Tasase tolerants (m)", min_value=0.001, value=0.05, step=0.005)
                    min_run = st.number_input("Min tasane lõik (m)", min_value=0.05, value=0.2, step=0.05)
                    min_depth = st.number_input("Min kõrgus põhjast (m)", min_value=0.0, value=0.3, step=0.05)
                with c3:
                    slope = st.text_input("Nõlva kalle (nt 1:2)", value="1:2")
                    bottom_w = st.number_input("Põhja laius b (m)", min_value=0.0, value=0.40, step=0.05)

                if st.button("Arvuta PK tabel", use_container_width=True):
                    try:
                        res = compute_pk_table_from_landxml(
                            xml_bytes=xml_bytes,
                            axis_xy=axis_xy,                 # (E,N)
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
                        st.success(f"✅ Kokku maht: {float(res['total_volume_m3']):.3f} m³")
                        st.write(f"Telje pikkus: **{float(res['axis_length_m']):.2f} m** | PK-sid: **{int(res['count'])}**")

                        planned_area = None
                        if float(res["axis_length_m"]) > 0 and res["total_volume_m3"] is not None:
                            planned_area = float(res["total_volume_m3"]) / float(res["axis_length_m"])

                        key = st.session_state.get("landxml_key") or (_row_get(p, "landxml_key") or None)
                        set_project_landxml(
                            int(_row_get(p, "id")),
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
                            file_name=f"{pname}_pk_tabel.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Arvutus ebaõnnestus: {e}")
                        st.code(traceback.format_exc())

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Failid ----------------
        with tab_files:
            st.markdown('<div class="block">', unsafe_allow_html=True)
            st.subheader("📤 Failid (Cloudflare R2)")

            uploads = st.file_uploader("Laadi üles failid", accept_multiple_files=True, key="proj_files")
            if uploads:
                prefix = project_prefix(pname)
                for up in uploads:
                    upload_file(s3, prefix, up)
                st.success(f"Üles laaditud {len(uploads)} faili.")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📄 Projekti failid")
            prefix = project_prefix(pname)
            files = list_files(s3, prefix) or []

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
