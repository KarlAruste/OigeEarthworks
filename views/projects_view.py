# views/projects_view.py
# Canvas-põhine telje joonistamine Streamlitis (Renderis töökindel: klikid tulevad alati tagasi)

import json
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd

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
    build_tin_index,
    snap_xy_to_tin,
    compute_pk_table_from_landxml,
    polyline_length,
)


# -------------------------
# Helpers
# -------------------------

def _downsample_points(xyz: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    """Downsample to keep the canvas fast."""
    if xyz is None or xyz.size == 0:
        return xyz
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz
    idx = np.random.choice(n, size=max_pts, replace=False)
    return xyz[idx]


def _origin_abs(xyz_show_abs: np.ndarray) -> tuple[float, float]:
    x0 = float(np.nanmean(xyz_show_abs[:, 0]))
    y0 = float(np.nanmean(xyz_show_abs[:, 1]))
    return (x0, y0)


def _abs_to_local(xyz_abs: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    x0, y0 = origin
    out = xyz_abs.copy()
    out[:, 0] = out[:, 0] - x0
    out[:, 1] = out[:, 1] - y0
    return out


# -------------------------
# Canvas component (no build step)
# -------------------------

def canvas_pick_point(points_local_xy: np.ndarray, axis_local_xy: list[tuple[float, float]]):
    """Interactive HTML5 canvas: pan+zoom with mouse, click selects nearest point.

    Returns dict like:
      {"x": local_x, "y": local_y, "picked": true}
    or None.

    NOTE: We use Streamlit's iframe postMessage protocol to send values back.
    """
    import streamlit.components.v1 as components

    pts = points_local_xy[:, :2].astype(float)
    pts_list = pts.tolist()
    axis_list = [(float(a), float(b)) for (a, b) in axis_local_xy]

    payload = {"points": pts_list, "axis": axis_list}

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
  const pts = data.points; // [[x,y],...]
  const axis = data.axis;  // [[x,y],...]

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
    for (const [x,y] of pts) {{
      if (x<xmin) xmin=x; if (x>xmax) xmax=x;
      if (y<ymin) ymin=y; if (y>ymax) ymax=y;
    }}
    if (!isFinite(xmin)) {{ xmin=-1; xmax=1; ymin=-1; ymax=1; }}
    return {{xmin,xmax,ymin,ymax}};
  }}

  let view = {{ scale: 1, ox: 0, oy: 0 }};

  function fit() {{
    const {{xmin,xmax,ymin,ymax}} = bounds();
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const dx = (xmax-xmin) || 1;
    const dy = (ymax-ymin) || 1;
    const sx = (w*0.85) / dx;
    const sy = (h*0.85) / dy;
    view.scale = Math.min(sx, sy);
    const cx = (xmin+xmax)/2;
    const cy = (ymin+ymax)/2;
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

    ctx.save();
    ctx.fillStyle = 'rgba(75, 110, 255, 0.85)';
    for (let i=0; i<pts.length; i++) {{
      const [x,y] = pts[i];
      const [sx,sy] = worldToScreen(x,y);
      if (sx<-10 || sy<-10 || sx>w+10 || sy>h+10) continue;
      ctx.beginPath();
      ctx.arc(sx,sy,2.4,0,Math.PI*2);
      ctx.fill();
    }}
    ctx.restore();

    if (axis && axis.length>0) {{
      ctx.save();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i=0; i<axis.length; i++) {{
        const [x,y] = axis[i];
        const [sx,sy] = worldToScreen(x,y);
        if (i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
      }}
      ctx.stroke();

      ctx.fillStyle = '#f59e0b';
      for (let i=0; i<axis.length; i++) {{
        const [x,y] = axis[i];
        const [sx,sy] = worldToScreen(x,y);
        ctx.beginPath(); ctx.arc(sx,sy,5,0,Math.PI*2); ctx.fill();
      }}
      ctx.restore();
    }}

    ctx.save();
    ctx.strokeStyle = '#11182733';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(w/2,0); ctx.lineTo(w/2,h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
    ctx.restore();
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
    const [wx, wy] = screenToWorld(mouseX, mouseY);

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
    const [wx, wy] = screenToWorld(e.offsetX, e.offsetY);
    const idx = nearestPoint(wx, wy);
    if (idx < 0) return;
    const px = pts[idx][0];
    const py = pts[idx][1];
    info.textContent = `Valitud (local): E=${{px.toFixed(3)}}, N=${{py.toFixed(3)}}`;
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

    value = components.html(html, height=670, scrolling=False)
    return value


# -------------------------
# Main view
# -------------------------

def render_projects_view():
    st.title("Projects")

    s3 = get_s3()
    projects = list_projects()

    # Session defaults
    st.session_state.setdefault("active_project_id", None)
    st.session_state.setdefault("axis_xy", [])          # ABS [(E,N)]
    st.session_state.setdefault("axis_finished", False)
    st.session_state.setdefault("landxml_bytes", None)
    st.session_state.setdefault("landxml_key", None)
    st.session_state.setdefault("plot_origin", None)    # ABS (E0,N0)
    st.session_state.setdefault("xyz_show_cache", None) # ABS points for display
    st.session_state.setdefault("tin_index", None)      # full index for snapping
    st.session_state.setdefault("tin_sig", None)        # invalidation signature

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
                st.session_state["axis_xy"] = []
                st.session_state["axis_finished"] = False
                st.session_state["landxml_bytes"] = None
                st.session_state["landxml_key"] = None
                st.session_state["plot_origin"] = None
                st.session_state["xyz_show_cache"] = None
                st.session_state["tin_index"] = None
                st.session_state["tin_sig"] = None
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
            st.session_state["plot_origin"] = None
            st.session_state["xyz_show_cache"] = None
            st.session_state["tin_index"] = None
            st.session_state["tin_sig"] = None

            st.success("LandXML salvestatud ja laaditud sessiooni.")
            st.rerun()

        # ---------------- Axis drawing + PK ----------------
        st.subheader("🧭 Telje joonistamine (polyline) + PK tabel")

        xml_bytes = st.session_state["landxml_bytes"]
        if not xml_bytes:
            st.info("Lae LandXML üles, et saaks telge joonistada.")
            return

        # read/cache points + build index once
        try:
            pts_dict, faces = read_landxml_tin_from_bytes(xml_bytes)
            xyz = np.array(list(pts_dict.values()), dtype=float)
            xyz_show = _downsample_points(xyz, max_pts=20000)
            st.session_state["xyz_show_cache"] = xyz_show

            if st.session_state["plot_origin"] is None:
                st.session_state["plot_origin"] = _origin_abs(xyz_show)

            sig = (st.session_state.get("landxml_key"), len(xml_bytes), len(pts_dict), len(faces))
            if st.session_state["tin_index"] is None or st.session_state["tin_sig"] != sig:
                st.session_state["tin_index"] = build_tin_index(pts_dict, faces)
                st.session_state["tin_sig"] = sig

            x_min = float(np.nanmin(xyz_show[:, 0])); x_max = float(np.nanmax(xyz_show[:, 0]))
            y_min = float(np.nanmin(xyz_show[:, 1])); y_max = float(np.nanmax(xyz_show[:, 1]))
            st.caption(f"ABS X min/max: {x_min:.3f} / {x_max:.3f} | ABS Y min/max: {y_min:.3f} / {y_max:.3f}")

        except Exception as e:
            st.error(f"LandXML lugemine ebaõnnestus: {e}")
            return

        # buttons
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

        axis_xy_abs = st.session_state["axis_xy"]
        finished = st.session_state["axis_finished"]
        origin = st.session_state["plot_origin"]
        xyz_show_abs = st.session_state["xyz_show_cache"]
        idx_full = st.session_state["tin_index"]

        xyz_show_local = _abs_to_local(xyz_show_abs, origin)
        axis_local = [(a - origin[0], b - origin[1]) for (a, b) in axis_xy_abs]

        st.caption("✅ Canvas: kliki kus tahes — lisab teljele lähima TIN punkti (snap päris TIN vastu). Zoom: rullik. Pan: lohista.")

        picked = canvas_pick_point(xyz_show_local, axis_local)

        if picked and isinstance(picked, dict) and picked.get("picked") and (not finished):
            lx = float(picked.get("x"))
            ly = float(picked.get("y"))

            cx_abs = lx + origin[0]
            cy_abs = ly + origin[1]

            sx, sy = snap_xy_to_tin(idx_full, float(cx_abs), float(cy_abs))
            st.session_state["axis_xy"] = axis_xy_abs + [(float(sx), float(sy))]
            st.rerun()

        if axis_xy_abs:
            L = polyline_length(axis_xy_abs)
            st.write(f"**Telje pikkus:** {L:.2f} m  |  Punkte: {len(axis_xy_abs)}")

        # parameters
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
                    axis_xy_abs=axis_xy_abs,  # FIXED
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
