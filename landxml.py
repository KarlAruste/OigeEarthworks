# landxml.py
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ----------------------------
# XML helpers
# ----------------------------
def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _iter_by_local(root, local_name: str):
    ln = local_name.lower()
    for el in root.iter():
        if _strip(el.tag).lower() == ln:
            yield el


def _find_first_by_local(root, local_name: str) -> Optional[ET.Element]:
    for el in _iter_by_local(root, local_name):
        return el
    return None


# ----------------------------
# Read LandXML TIN from bytes
# Supports:
#  - <Surfaces><Surface><Definition surfType="TIN"><Pnts><P ...>N E Z</P> ...</Pnts><Faces><F>...</F></Faces>
# Output ALWAYS: pts = {id: (E, N, Z)}
# ----------------------------
def read_landxml_tin_from_bytes(
    xml_bytes: bytes,
) -> Tuple[Dict[int, Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    root = ET.fromstring(xml_bytes)

    pts: Dict[int, Tuple[float, float, float]] = {}

    # points
    for p in _iter_by_local(root, "P"):
        pid = p.get("id")
        txt = (p.text or "").strip()
        if not pid or not txt:
            continue
        parts = txt.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            a = float(parts[0])
            b = float(parts[1])
            z = float(parts[2])
            pts[int(pid)] = (a, b, z)  # order fixed below
        except Exception:
            continue

    if not pts:
        raise ValueError("LandXML: ei leidnud <P> punkte (Definition/Pnts/P).")

    # faces
    faces: List[Tuple[int, int, int]] = []
    for f in _iter_by_local(root, "F"):
        txt = (f.text or "").strip()
        if not txt:
            continue
        parts = txt.split()
        if len(parts) < 3:
            continue
        try:
            faces.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except Exception:
            continue

    if not faces:
        raise ValueError("LandXML: ei leidnud <F> faces (Definition/Faces/F).")

    # Fix coordinate order heuristic: if first looks like Northing (millions) and second like Easting (hundreds of thousands) -> swap.
    arr = np.array(list(pts.values()), dtype=float)  # (a,b,z)
    a_med = float(np.median(arr[:, 0]))
    b_med = float(np.median(arr[:, 1]))

    need_swap = (a_med > 2_000_000.0 and b_med < 2_000_000.0)
    if need_swap:
        pts = {pid: (val[1], val[0], val[2]) for pid, val in pts.items()}  # E,N,Z

    return pts, faces


# ----------------------------
# Alignment / Axis import
#  - LandXML Alignment: <Alignments><Alignment length="..." ...><CoordGeom> ... <Line> <Start> <End> ... <Curve> <Start>/<PI>/<End>
#  - CSV/TXT: columns can be "E;N" or "N;E" or "x,y"
# Output ALWAYS: [(E,N), ...]
# ----------------------------
def _swap_en_if_needed_xy(xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not xy or len(xy) < 2:
        return xy
    a = np.array([p[0] for p in xy], dtype=float)
    b = np.array([p[1] for p in xy], dtype=float)
    a_med = float(np.median(a))
    b_med = float(np.median(b))
    # Typical Estonia: N ~ 6.x million, E ~ 0.3..0.8 million
    need_swap = (a_med > 2_000_000.0 and b_med < 2_000_000.0)
    if need_swap:
        return [(y, x) for (x, y) in xy]
    return xy


def read_axis_from_bytes(
    data: bytes,
    filename: str,
    force_swap_en: bool = False,
) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "axis_xy": [(E,N),...],
        "axis_length_m": float,    # computed from points
        "declared_length_m": float|None,  # from Alignment length attr if present
        "source": "landxml_alignment"|"csv_txt"
      }
    """
    name = (filename or "").lower().strip()

    # ---- try LandXML Alignment ----
    if name.endswith(".xml") or name.endswith(".landxml"):
        try:
            root = ET.fromstring(data)

            # find first Alignment
            align = None
            for a in _iter_by_local(root, "Alignment"):
                align = a
                break

            if align is not None:
                declared = align.get("length")
                declared_len = float(declared) if declared not in (None, "") else None

                # Gather points from CoordGeom
                # We prefer explicit Start/End points of each element. This gives a dense enough polyline for straight segments.
                axis: List[Tuple[float, float]] = []

                coord = None
                for cg in align.iter():
                    if _strip(cg.tag).lower() == "coordgeom":
                        coord = cg
                        break

                if coord is None:
                    # fallback: sometimes alignment points are in <CoordGeom> but namespace issues; try local scan
                    coord = _find_first_by_local(align, "CoordGeom")

                if coord is None:
                    raise ValueError("Alignment leitud, aga <CoordGeom> puudub.")

                def _parse_xy(text: str) -> Tuple[float, float]:
                    parts = (text or "").strip().replace(",", " ").split()
                    if len(parts) < 2:
                        raise ValueError("Coord text invalid")
                    return float(parts[0]), float(parts[1])

                # iterate geometry primitives
                for child in list(coord):
                    tag = _strip(child.tag).lower()
                    if tag in ("line", "curve", "spiral"):
                        start_el = _find_first_by_local(child, "Start")
                        end_el = _find_first_by_local(child, "End")
                        pi_el = _find_first_by_local(child, "PI")

                        # Add Start
                        if start_el is not None and (start_el.text or "").strip():
                            axis.append(_parse_xy(start_el.text))

                        # Some curves include PI point; add it for better shape
                        if pi_el is not None and (pi_el.text or "").strip():
                            axis.append(_parse_xy(pi_el.text))

                        # Add End
                        if end_el is not None and (end_el.text or "").strip():
                            axis.append(_parse_xy(end_el.text))

                # de-duplicate consecutive identical points
                cleaned: List[Tuple[float, float]] = []
                for p in axis:
                    if not cleaned:
                        cleaned.append(p)
                    else:
                        if (abs(cleaned[-1][0] - p[0]) > 1e-9) or (abs(cleaned[-1][1] - p[1]) > 1e-9):
                            cleaned.append(p)

                if len(cleaned) < 2:
                    raise ValueError("Alignmentist ei õnnestunud lugeda piisavalt punkte (vähemalt 2).")

                # Heuristic swap to E,N + optional force swap
                cleaned = _swap_en_if_needed_xy(cleaned)
                if force_swap_en:
                    cleaned = [(y, x) for (x, y) in cleaned]

                return {
                    "axis_xy": cleaned,
                    "axis_length_m": polyline_length(cleaned),
                    "declared_length_m": declared_len,
                    "source": "landxml_alignment",
                }
        except Exception:
            # fall through to CSV/TXT
            pass

    # ---- CSV/TXT points ----
    txt = data.decode("utf-8", errors="ignore").strip()
    if not txt:
        raise ValueError("Telje fail on tühi.")

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    axis: List[Tuple[float, float]] = []

    # detect separator
    # allow: "E;N", "E,N", "E N"
    for ln in lines:
        # skip header-ish
        low = ln.lower()
        if any(k in low for k in ("e;", "n;", "e,n", "north", "east", "x;", "y;", "x,", "y,")) and not any(ch.isdigit() for ch in ln):
            continue

        s = ln.replace("\t", " ").replace(",", " ").replace(";", " ")
        parts = [p for p in s.split() if p]
        if len(parts) < 2:
            continue
        try:
            a = float(parts[0])
            b = float(parts[1])
            axis.append((a, b))
        except Exception:
            continue

    if len(axis) < 2:
        raise ValueError("Telje failist ei saanud lugeda punkte. Ootan CSV/TXT kujul: E;N või N;E.")

    axis = _swap_en_if_needed_xy(axis)
    if force_swap_en:
        axis = [(y, x) for (x, y) in axis]

    return {
        "axis_xy": axis,
        "axis_length_m": polyline_length(axis),
        "declared_length_m": None,
        "source": "csv_txt",
    }


# ----------------------------
# Geometry / TIN index
# ----------------------------
def _point_in_tri_2d(px, py, ax, ay, bx, by, cx, cy) -> bool:
    def s(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    b1 = s(px, py, ax, ay, bx, by) < 0.0
    b2 = s(px, py, bx, by, cx, cy) < 0.0
    b3 = s(px, py, cx, cy, ax, ay) < 0.0
    return (b1 == b2) and (b2 == b3)


def _interp_z(px, py, A, B, C) -> Optional[float]:
    ax, ay, az = A
    bx, by, bz = B
    cx, cy, cz = C

    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    if abs(nz) < 1e-12:
        return None

    return float(az - (nx * (px - ax) + ny * (py - ay)) / nz)


@dataclass
class TinIndex:
    tris: np.ndarray          # (T,3,3) points (E,N,Z)
    minx: float
    miny: float
    cell: float
    tri_buckets: dict         # (ix,iy)->[tri_idx]
    pt_buckets: dict          # (ix,iy)->[(E,N,Z),...]
    pts_xyz: np.ndarray       # (N,3) full cloud (E,N,Z)


def build_tin_index(
    pts: Dict[int, Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
    target_bucket_count: int = 150_000,
) -> TinIndex:
    tris = np.empty((len(faces), 3, 3), dtype=float)
    for i, (a, b, c) in enumerate(faces):
        tris[i, 0, :] = pts[a]
        tris[i, 1, :] = pts[b]
        tris[i, 2, :] = pts[c]

    xs = tris[:, :, 0]
    ys = tris[:, :, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())

    area = max((maxx - minx) * (maxy - miny), 1e-9)
    cell = math.sqrt(area / max(target_bucket_count, 10_000))
    cell = max(cell, 0.5)

    tri_bbox = np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1)
    tri_buckets = {}
    for i, bb in enumerate(tri_bbox):
        ix0 = int((bb[0] - minx) // cell)
        iy0 = int((bb[1] - miny) // cell)
        ix1 = int((bb[2] - minx) // cell)
        iy1 = int((bb[3] - miny) // cell)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                tri_buckets.setdefault((ix, iy), []).append(i)

    pt_buckets = {}
    pts_xyz = np.array(list(pts.values()), dtype=float)
    for (x, y, z) in pts_xyz:
        ix = int((x - minx) // cell)
        iy = int((y - miny) // cell)
        pt_buckets.setdefault((ix, iy), []).append((float(x), float(y), float(z)))

    return TinIndex(
        tris=tris, minx=minx, miny=miny, cell=cell,
        tri_buckets=tri_buckets, pt_buckets=pt_buckets, pts_xyz=pts_xyz
    )


def nearest_point_xyz(idx: TinIndex, px: float, py: float, max_rings: int = 8) -> Optional[Tuple[float, float, float]]:
    ix = int((px - idx.minx) // idx.cell)
    iy = int((py - idx.miny) // idx.cell)

    best_d2 = None
    best = None

    for r in range(0, max_rings + 1):
        found_any = False
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if r > 0 and (abs(dx) != r and abs(dy) != r):
                    continue
                pts_list = idx.pt_buckets.get((ix + dx, iy + dy))
                if not pts_list:
                    continue
                found_any = True
                for x, y, z in pts_list:
                    d2 = (x - px) ** 2 + (y - py) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (x, y, z)
        if found_any and best is not None:
            return best
    return None


def z_at_xy(idx: TinIndex, px: float, py: float) -> Optional[float]:
    ix = int((px - idx.minx) // idx.cell)
    iy = int((py - idx.miny) // idx.cell)

    candidates: List[int] = []
    for dx in (0, -1, 1):
        for dy in (0, -1, 1):
            candidates += idx.tri_buckets.get((ix + dx, iy + dy), [])

    for ti in candidates:
        A, B, C = idx.tris[ti]
        if _point_in_tri_2d(px, py, A[0], A[1], B[0], B[1], C[0], C[1]):
            z = _interp_z(px, py, A, B, C)
            if z is not None:
                return z

    nn = nearest_point_xyz(idx, px, py, max_rings=8)
    if nn is None:
        return None
    return float(nn[2])


# ----------------------------
# Profiles + edges/depth
# ----------------------------
def sample_profile(idx: TinIndex, x1, y1, x2, y2, step) -> Tuple[np.ndarray, np.ndarray]:
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1e-9:
        raise ValueError("Ristlõike joon on liiga lühike.")
    n = max(2, int(L / step) + 1)
    ds = np.linspace(0.0, L, n)

    zs = np.full(n, np.nan)
    for i in range(n):
        t = ds[i] / L
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        z = z_at_xy(idx, px, py)
        if z is not None:
            zs[i] = z
    return ds, zs


def find_edges_and_depth(
    ds: np.ndarray,
    zs: np.ndarray,
    tol: float,
    min_run: float,
    sample_step: float,
    min_depth_from_bottom: float = 0.3,
    center_window_m: float = 6.0,
    fallback_mode: str = "robust",   # "none" | "robust"
):
    """
    Tagastab dict:
      width, depth, bottom_z, edge_z, left_s, right_s, left_edge_z, right_edge_z, method

    method:
      - "flat"    -> leiti tasased servad
      - "robust"  -> fallback (percentile top + min bottom)
      - None      -> ei õnnestunud
    """
    valid = np.isfinite(zs)
    if valid.sum() < 5:
        return None

    n = len(ds)
    mid = n // 2

    # --- bottom near center window (as before) ---
    halfw = max(2, int((center_window_m / sample_step) / 2))
    i0 = max(0, mid - halfw)
    i1 = min(n, mid + halfw + 1)

    if valid[i0:i1].sum() < 3:
        i0, i1 = 0, n

    sub = zs[i0:i1].copy()
    sub[~np.isfinite(sub)] = np.inf
    bottom_idx = i0 + int(np.argmin(sub))
    bottom_z = float(zs[bottom_idx])

    seg_pts = max(3, int(min_run / sample_step) + 1)

    def is_edge_flat(seg):
        segv = seg[np.isfinite(seg)]
        if len(segv) < 2:
            return False, None
        if float(segv.max() - segv.min()) > tol:
            return False, None
        m = float(segv.mean())
        if m < bottom_z + min_depth_from_bottom:
            return False, None
        return True, m

    # --- try FLAT edges (original logic) ---
    left_center = None
    left_edge_z = None
    for end in range(bottom_idx, seg_pts - 1, -1):
        seg = zs[end - seg_pts + 1:end + 1]
        ok, m = is_edge_flat(seg)
        if ok:
            left_center = end - (seg_pts // 2)
            left_edge_z = m
            break

    right_center = None
    right_edge_z = None
    for start in range(bottom_idx, n - seg_pts):
        seg = zs[start:start + seg_pts]
        ok, m = is_edge_flat(seg)
        if ok:
            right_center = start + (seg_pts // 2)
            right_edge_z = m
            break

    if left_center is not None and right_center is not None and left_edge_z is not None and right_edge_z is not None:
        width = float(ds[right_center] - ds[left_center])
        if width <= 0:
            return None
        edge_z = float((left_edge_z + right_edge_z) / 2.0)
        depth = float(edge_z - bottom_z)
        if depth <= 0:
            return None
        return {
            "width": width,
            "depth": depth,
            "bottom_z": bottom_z,
            "edge_z": edge_z,
            "left_s": float(ds[left_center]),
            "right_s": float(ds[right_center]),
            "left_edge_z": float(left_edge_z),
            "right_edge_z": float(right_edge_z),
            "method": "flat",
        }

    # --- FALLBACK ---
    if fallback_mode != "robust":
        return None

    zsv = zs[np.isfinite(zs)]
    if zsv.size < 5:
        return None

    # robust edge level = top percentile (p90)
    edge_z = float(np.percentile(zsv, 90))
    bottom_z2 = float(np.min(zsv))
    depth = float(edge_z - bottom_z2)
    if depth <= 0:
        return None

    # estimate width: span where z is near "edge_z"
    # threshold: within (tol*2) below edge_z
    thr = edge_z - max(tol * 2.0, 0.05)
    idxs = np.where((np.isfinite(zs)) & (zs >= thr))[0]
    if idxs.size >= 2:
        left_i = int(idxs[0])
        right_i = int(idxs[-1])
        width = float(ds[right_i] - ds[left_i])
        left_s = float(ds[left_i])
        right_s = float(ds[right_i])
    else:
        # if can't estimate, use full cross length
        width = float(ds[-1] - ds[0])
        left_s = float(ds[0])
        right_s = float(ds[-1])

    return {
        "width": width,
        "depth": depth,
        "bottom_z": bottom_z2,
        "edge_z": edge_z,
        "left_s": left_s,
        "right_s": right_s,
        "left_edge_z": edge_z,
        "right_edge_z": edge_z,
        "method": "robust",
    }



# ----------------------------
# Slope + area
# ----------------------------
def parse_slope_ratio(text: str) -> float:
    t = text.strip().replace(" ", "")
    if ":" in t:
        a, b = t.split(":", 1)
        v = float(a)
        h = float(b)
        if v == 0:
            raise ValueError("Nõlv 1:n: vertikaal ei tohi olla 0")
        return h / v
    return float(t)


def area_trapezoid(bottom_w: float, depth: float, slope_h_over_v: float) -> float:
    top_w = bottom_w + 2.0 * slope_h_over_v * depth
    return (bottom_w + top_w) * 0.5 * depth


# ----------------------------
# Polyline helpers (ABS coords!)
# ----------------------------
def polyline_cumlen(xy: List[Tuple[float, float]]) -> np.ndarray:
    cum = [0.0]
    for i in range(len(xy) - 1):
        dx = xy[i + 1][0] - xy[i][0]
        dy = xy[i + 1][1] - xy[i][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    return np.array(cum, dtype=float)


def polyline_length(xy: List[Tuple[float, float]]) -> float:
    if len(xy) < 2:
        return 0.0
    return float(polyline_cumlen(xy)[-1])


def point_and_tangent_at(xy: List[Tuple[float, float]], s: float):
    cum = polyline_cumlen(xy)
    total = float(cum[-1])

    if s <= 0:
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        tx, ty = x1 - x0, y1 - y0
        L = math.hypot(tx, ty) or 1.0
        return (x0, y0), (tx / L, ty / L)

    if s >= total:
        x0, y0 = xy[-2]
        x1, y1 = xy[-1]
        tx, ty = x1 - x0, y1 - y0
        L = math.hypot(tx, ty) or 1.0
        return (x1, y1), (tx / L, ty / L)

    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(xy) - 2))

    s0 = cum[i]
    s1 = cum[i + 1]
    segL = s1 - s0
    t = (s - s0) / segL if segL > 0 else 0.0

    x0, y0 = xy[i]
    x1, y1 = xy[i + 1]
    px = x0 + t * (x1 - x0)
    py = y0 + t * (y1 - y0)

    tx, ty = (x1 - x0), (y1 - y0)
    L = math.hypot(tx, ty) or 1.0
    return (px, py), (tx / L, ty / L)


def pk_fmt(meters: int) -> str:
    km = meters // 1000
    m = meters % 1000
    if meters < 100:
        return f"{km}+{m:02d}"
    return f"{km}+{m:03d}"


# ----------------------------
# PK table computation from LandXML bytes + axis ABS polyline
# ----------------------------
def compute_pk_table_from_landxml(
    xml_bytes: bytes,
    axis_xy_abs: Optional[List[Tuple[float, float]]] = None,
    axis_xy: Optional[List[Tuple[float, float]]] = None,  # backwards compat
    pk_step: float = 1.0,
    cross_len: float = 25.0,
    sample_step: float = 0.1,
    tol: float = 0.05,
    min_run: float = 0.2,
    min_depth_from_bottom: float = 0.3,
    slope_text: str = "1:2",
    bottom_w: float = 0.40,
):
    if axis_xy_abs is None:
        axis_xy_abs = axis_xy or []
    if len(axis_xy_abs) < 2:
        raise ValueError("Telg peab olema vähemalt 2 punktiga.")

    pts, faces = read_landxml_tin_from_bytes(xml_bytes)
    idx = build_tin_index(pts, faces)
    slope_hv = parse_slope_ratio(slope_text)

    total_len = polyline_length(axis_xy_abs)
    count = int(math.floor(total_len / pk_step))
    half = cross_len / 2.0

    rows = []
    total_volume = 0.0

    for k in range(1, count + 1):
        s = k * pk_step
        (px, py), (tx, ty) = point_and_tangent_at(axis_xy_abs, s)

        # normal
        nx, ny = -ty, tx

        x1 = px - nx * half
        y1 = py - ny * half
        x2 = px + nx * half
        y2 = py + ny * half

        ds, zs = sample_profile(idx, x1, y1, x2, y2, step=sample_step)
        info = find_edges_and_depth(
            ds, zs,
            tol=tol,
            min_run=min_run,
            sample_step=sample_step,
            min_depth_from_bottom=min_depth_from_bottom,
            center_window_m=min(6.0, cross_len / 3.0)
            fallback_mode="robust",
        )

        pk_label = pk_fmt(int(round(s)))

        if info is None:
            rows.append({
                "PK": pk_label,
                "Width_m": None,
                "Depth_m": None,
                "EdgeZ": None,
                "BottomZ": None,
                "Area_m2": None,
                "Vol_m3": None,
            })
            continue

        depth = float(info["depth"])
        edge_z = float(info["edge_z"])
        bottom_z = float(info["bottom_z"])

        A = float(area_trapezoid(bottom_w=bottom_w, depth=depth, slope_h_over_v=slope_hv))
        V = float(A * pk_step)
        total_volume += V

        rows.append({
            "PK": pk_label,
            "Width_m": float(info["width"]),
            "Depth_m": depth,
            "EdgeZ": edge_z,
            "BottomZ": bottom_z,
            "Area_m2": A,
            "Vol_m3": V,
        })

    return {
        "rows": rows,
        "count": count,
        "axis_length_m": float(total_len),
        "total_volume_m3": float(total_volume),
    }
