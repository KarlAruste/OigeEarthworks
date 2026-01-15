# landxml.py
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from lxml import etree


# ============================================================
# Helpers
# ============================================================

def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def _xp(root, q: str):
    ns_uri = root.nsmap.get(None)
    ns = {"lx": ns_uri} if ns_uri else {}
    return root.xpath(q, namespaces=ns) if ns_uri else root.xpath(q.replace("lx:", ""))

def _as_float_list(text: str) -> List[float]:
    return [float(x) for x in text.strip().split()]

def _ne_to_en2(n: float, e: float) -> Tuple[float, float]:
    # File is N,E -> internal X=E, Y=N
    return float(e), float(n)

def _ne_to_en3(n: float, e: float, z: float) -> Tuple[float, float, float]:
    x, y = _ne_to_en2(n, e)
    return x, y, float(z)

def pk_fmt(meters: int) -> str:
    km = meters // 1000
    m = meters % 1000
    return f"{km}+{m:03d}"


# ============================================================
# TIN reader (LandXML Surfaces)
# LandXML points are assumed as: N E Z  (your files)
# Internally we store: (E, N, Z)
# ============================================================

def read_landxml_tin_from_bytes(data: bytes):
    root = etree.fromstring(data)

    p_elems = _xp(root, ".//lx:Surfaces//lx:Surface//lx:Definition//lx:Pnts//lx:P")
    f_elems = _xp(root, ".//lx:Surfaces//lx:Surface//lx:Definition//lx:Faces//lx:F")

    if not p_elems or not f_elems:
        raise ValueError("LandXML-ist ei leitud TIN Pnts/Faces (Surfaces/Definition/Pnts/Faces).")

    pts: Dict[int, Tuple[float, float, float]] = {}
    for p in p_elems:
        pid = int(p.get("id"))
        vals = _as_float_list(p.text)
        if len(vals) < 3:
            continue
        n, e, z = vals[0], vals[1], vals[2]  # file: N E Z
        pts[pid] = _ne_to_en3(n, e, z)

    faces: List[Tuple[int, int, int]] = []
    for f in f_elems:
        a, b, c = [int(x) for x in f.text.strip().split()]
        faces.append((a, b, c))

    if len(pts) < 3 or len(faces) < 1:
        raise ValueError("TIN andmed on liiga väikesed (punkte/faces puudub).")

    return pts, faces


# ============================================================
# TIN geometry + spatial index
# ============================================================

def point_in_tri_2d(px, py, ax, ay, bx, by, cx, cy):
    def s(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    b1 = s(px, py, ax, ay, bx, by) < 0.0
    b2 = s(px, py, bx, by, cx, cy) < 0.0
    b3 = s(px, py, cx, cy, ax, ay) < 0.0
    return (b1 == b2) and (b2 == b3)

def interp_z(px, py, A, B, C):
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

    return az - (nx * (px - ax) + ny * (py - ay)) / nz

@dataclass
class TinIndex:
    tris: np.ndarray
    minx: float
    miny: float
    cell: float
    tri_buckets: dict
    pt_buckets: dict

def build_tin_index(pts, faces, target_bucket_count=150_000):
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
    for (x, y, z) in pts.values():
        ix = int((x - minx) // cell)
        iy = int((y - miny) // cell)
        pt_buckets.setdefault((ix, iy), []).append((float(x), float(y), float(z)))

    return TinIndex(tris=tris, minx=minx, miny=miny, cell=cell, tri_buckets=tri_buckets, pt_buckets=pt_buckets)

def nearest_point_z(idx: TinIndex, px: float, py: float, max_rings: int = 6):
    ix = int((px - idx.minx) // idx.cell)
    iy = int((py - idx.miny) // idx.cell)

    best_d2 = None
    best_z = None

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
                        best_z = z
        if found_any and best_z is not None:
            return float(best_z)

    return None

def z_at_xy(idx: TinIndex, px: float, py: float):
    ix = int((px - idx.minx) // idx.cell)
    iy = int((py - idx.miny) // idx.cell)

    candidates = []
    for dx in (0, -1, 1):
        for dy in (0, -1, 1):
            candidates += idx.tri_buckets.get((ix + dx, iy + dy), [])

    for ti in candidates:
        A, B, C = idx.tris[ti]
        if point_in_tri_2d(px, py, A[0], A[1], B[0], B[1], C[0], C[1]):
            z = interp_z(px, py, A, B, C)
            if z is not None:
                return float(z)

    return nearest_point_z(idx, px, py, max_rings=6)

def sample_profile(idx: TinIndex, x1, y1, x2, y2, step):
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


# ============================================================
# Edge finding (REAL ditch edges)
# 1) Try "first flat plateau" outward from bottom (left/right)
# 2) Fallback: slope-break (gradient drop) if no plateau exists
# ============================================================

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")

def find_edges_and_depth(
    ds: np.ndarray,
    zs: np.ndarray,
    tol: float,
    min_run: float,
    sample_step: float,
    min_depth_from_bottom: float = 0.3,
    center_window_m: float = 6.0,

    # plateau search limits
    max_edge_dist_m: Optional[float] = None,   # nt 12.0; None = unlimited

    # slope-break fallback parameters
    slope_threshold: float = 0.12,             # dz/ds (m/m) below this means "almost flat"
    slope_smooth_window_m: float = 0.5,        # moving average window on slope
    slope_min_run_m: float = 0.7,              # must stay "almost flat" for this length
):
    """
    Returns dict with:
      width, depth, bottom_z, edge_z, left_idx, right_idx, ... etc.

    Plateau method:
      - find bottom (min Z in center window)
      - move outward and pick FIRST flat segment (tol, min_run, above bottom+min_depth)

    Fallback slope-break:
      - compute slope magnitude |dz/ds| and smooth it
      - find FIRST region where slope < slope_threshold for slope_min_run_m
        and mean z above bottom+min_depth
    """
    valid = np.isfinite(zs)
    if valid.sum() < 10:
        return None

    n = len(ds)
    if n < 5:
        return None

    mid = n // 2

    # ---- find bottom in center window ----
    halfw = max(3, int((center_window_m / sample_step) / 2))
    i0 = max(0, mid - halfw)
    i1 = min(n, mid + halfw + 1)
    if valid[i0:i1].sum() < 5:
        i0, i1 = 0, n

    sub = zs[i0:i1].copy()
    sub[~np.isfinite(sub)] = np.inf
    bottom_idx = i0 + int(np.argmin(sub))
    bottom_z = float(zs[bottom_idx])
    if not math.isfinite(bottom_z):
        return None

    seg_pts = max(5, int(min_run / sample_step) + 1)
    slope_run_pts = max(5, int(slope_min_run_m / sample_step) + 1)
    smooth_w = max(1, int(slope_smooth_window_m / sample_step))

    def within_limit(idx_center: int) -> bool:
        if max_edge_dist_m is None:
            return True
        return abs(float(ds[idx_center] - ds[bottom_idx])) <= float(max_edge_dist_m)

    def flat_ok(seg_vals: np.ndarray):
        segv = seg_vals[np.isfinite(seg_vals)]
        if len(segv) < max(3, seg_pts // 2):
            return (False, None)
        if float(segv.max() - segv.min()) > tol:
            return (False, None)
        m = float(segv.mean())
        if m < bottom_z + min_depth_from_bottom:
            return (False, None)
        return (True, m)

    # ---- 1) Plateau method: FIRST flat segment outward from bottom ----
    def find_plateau_left():
        for end in range(bottom_idx, seg_pts - 2, -1):
            center = end - (seg_pts // 2)
            if center <= 0:
                continue
            if not within_limit(center):
                continue
            seg = zs[end - seg_pts + 1 : end + 1]
            ok, m = flat_ok(seg)
            if ok:
                return int(center), float(m)
        return None

    def find_plateau_right():
        for start in range(bottom_idx, n - seg_pts + 1):
            center = start + (seg_pts // 2)
            if center >= n:
                continue
            if not within_limit(center):
                continue
            seg = zs[start : start + seg_pts]
            ok, m = flat_ok(seg)
            if ok:
                return int(center), float(m)
        return None

    # ---- 2) Slope-break fallback ----
    # compute slope magnitude |dz/ds| for finite points
    # we need continuous-ish arrays; fill NaNs by interpolation to make slope usable
    def slope_break_find(side: str):
        z = zs.astype(float).copy()
        x = ds.astype(float).copy()

        finite = np.isfinite(z)
        if finite.sum() < 10:
            return None

        # interpolate NaNs linearly across ds (only for slope detection)
        if not finite.all():
            idxs = np.arange(n)
            z[~finite] = np.interp(idxs[~finite], idxs[finite], z[finite])

        dz = np.diff(z)
        dx = np.diff(x)
        dx[dx == 0] = 1e-9
        slope = np.abs(dz / dx)  # length n-1
        slope = _moving_average(slope, smooth_w)

        # helper to validate a run: slope below threshold and elevation above bottom+min_depth
        def run_ok(start_i: int, end_i: int):
            # run covers slope indices [start_i:end_i] inclusive in slope array
            # corresponding z points roughly start_i..end_i+1
            z_seg = z[start_i : end_i + 2]
            m = float(np.mean(z_seg))
            if m < bottom_z + min_depth_from_bottom:
                return (False, None)
            return (True, m)

        if side == "left":
            # search outward: from bottom_idx towards 0
            # slope index i corresponds between point i and i+1
            # We want first "almost flat" run while moving left:
            # check windows that END near bottom and slide outward.
            # We'll scan candidate windows by their center point distance.
            for end_pt in range(bottom_idx, slope_run_pts + 1, -1):
                # slope window ends at end_pt-1 in slope array (between end_pt-1 and end_pt)
                end_s = end_pt - 1
                start_s = end_s - slope_run_pts + 1
                if start_s < 0:
                    continue
                center_pt = (start_s + end_s) // 2
                if not within_limit(center_pt):
                    continue
                seg = slope[start_s : end_s + 1]
                if float(np.max(seg)) <= float(slope_threshold):
                    ok, m = run_ok(start_s, end_s)
                    if ok:
                        return int(center_pt), float(m)
            return None

        else:
            # right: from bottom_idx towards n-1
            for start_pt in range(bottom_idx, (n - 1) - slope_run_pts):
                start_s = start_pt
                end_s = start_s + slope_run_pts - 1
                if end_s >= len(slope):
                    break
                center_pt = (start_s + end_s) // 2
                if not within_limit(center_pt):
                    continue
                seg = slope[start_s : end_s + 1]
                if float(np.max(seg)) <= float(slope_threshold):
                    ok, m = run_ok(start_s, end_s)
                    if ok:
                        return int(center_pt), float(m)
            return None

    left = find_plateau_left()
    right = find_plateau_right()

    # Fallbacks if plateau missing
    if left is None:
        left = slope_break_find("left")
    if right is None:
        right = slope_break_find("right")

    if left is None or right is None:
        return None

    left_idx, left_edge_z = left
    right_idx, right_edge_z = right

    if right_idx <= left_idx:
        return None

    width = float(ds[right_idx] - ds[left_idx])
    edge_z = float((left_edge_z + right_edge_z) / 2.0)
    depth = float(edge_z - bottom_z)

    if width <= 0 or depth <= 0:
        return None

    return {
        "width": width,
        "depth": depth,
        "bottom_z": bottom_z,
        "edge_z": edge_z,
        "left_s": float(ds[left_idx]),
        "right_s": float(ds[right_idx]),
        "left_edge_z": float(left_edge_z),
        "right_edge_z": float(right_edge_z),
        "bottom_idx": int(bottom_idx),
        "left_idx": int(left_idx),
        "right_idx": int(right_idx),
    }


# ============================================================
# Slope + area
# ============================================================

def parse_slope_ratio(text: str) -> float:
    """
    '1:2' -> 2.0 (H/V)
    """
    t = text.strip().replace(" ", "")
    if ":" in t:
        a, b = t.split(":", 1)
        v = float(a)
        h = float(b)
        if v == 0:
            raise ValueError("Nõlva kalle: vertikaal ei tohi olla 0")
        return h / v
    return float(t)

def area_trapezoid(bottom_w: float, depth: float, slope_h_over_v: float) -> float:
    top_w = bottom_w + 2.0 * slope_h_over_v * depth
    return (bottom_w + top_w) * 0.5 * depth


# ============================================================
# Alignment parser (Civil3D LandXML)
# Uses CoordGeom Line + Curve. Uses element length attributes.
# File coords are N E, internally store E,N
# ============================================================

@dataclass
class LineSeg:
    x0: float
    y0: float
    x1: float
    y1: float
    length: float

@dataclass
class ArcSeg:
    cx: float
    cy: float
    r: float
    rot: str  # "cw" or "ccw"
    a0: float
    a1: float
    length: float

Segment = Any  # LineSeg | ArcSeg

def _angle_norm(a: float) -> float:
    tw = 2.0 * math.pi
    a = a % tw
    if a < 0:
        a += tw
    return a

def _arc_delta(a0: float, a1: float, rot: str) -> float:
    a0 = _angle_norm(a0)
    a1 = _angle_norm(a1)
    tw = 2.0 * math.pi
    if rot == "ccw":
        d = a1 - a0
        if d < 0:
            d += tw
        return d
    else:
        d = a0 - a1
        if d < 0:
            d += tw
        return d

def parse_alignment_from_bytes(data: bytes) -> dict:
    root = etree.fromstring(data)

    alns = _xp(root, ".//lx:Alignments//lx:Alignment")
    if not alns:
        raise ValueError("Alignment LandXML-ist ei leitud <Alignments><Alignment>.")
    aln = alns[0]

    aln_name = aln.get("name") or "Alignment"
    aln_len_attr = aln.get("length")
    aln_len_attr = float(aln_len_attr) if aln_len_attr else None

    segs: List[Segment] = []

    coordgeom = _xp(aln, ".//lx:CoordGeom")[0] if _xp(aln, ".//lx:CoordGeom") else None
    if coordgeom is None:
        raise ValueError("Alignment-is puudub <CoordGeom>.")

    for child in coordgeom:
        tag = _strip(child.tag).lower()

        if tag == "line":
            L = child.get("length")
            L = float(L) if L else None

            start = _xp(child, ".//lx:Start")
            end = _xp(child, ".//lx:End")
            if not start or not end:
                continue

            n0, e0 = _as_float_list(start[0].text)[:2]
            n1, e1 = _as_float_list(end[0].text)[:2]
            x0, y0 = _ne_to_en2(n0, e0)
            x1, y1 = _ne_to_en2(n1, e1)

            if L is None:
                L = math.hypot(x1 - x0, y1 - y0)

            segs.append(LineSeg(x0=x0, y0=y0, x1=x1, y1=y1, length=float(L)))

        elif tag == "curve":
            rot = (child.get("rot") or "ccw").lower()
            r = float(child.get("radius") or "0")
            L = child.get("length")
            L = float(L) if L else None

            start = _xp(child, ".//lx:Start")
            end = _xp(child, ".//lx:End")
            center = _xp(child, ".//lx:Center")
            if not start or not end or not center or r <= 0:
                continue

            n0, e0 = _as_float_list(start[0].text)[:2]
            n1, e1 = _as_float_list(end[0].text)[:2]
            nc, ec = _as_float_list(center[0].text)[:2]

            x0, y0 = _ne_to_en2(n0, e0)
            x1, y1 = _ne_to_en2(n1, e1)
            cx, cy = _ne_to_en2(nc, ec)

            a0 = math.atan2(y0 - cy, x0 - cx)
            a1 = math.atan2(y1 - cy, x1 - cx)

            d = _arc_delta(a0, a1, rot)
            if L is None:
                L = abs(r * d)

            segs.append(ArcSeg(cx=cx, cy=cy, r=r, rot=rot, a0=a0, a1=a1, length=float(L)))

        else:
            continue

    if not segs:
        raise ValueError("Alignment segmente ei õnnestunud lugeda (Line/Curve).")

    total_len = sum(float(s.length) for s in segs)
    if aln_len_attr is not None and aln_len_attr > 0:
        total_len = float(aln_len_attr)

    return {"name": aln_name, "segments": segs, "length": float(total_len)}

def point_and_tangent_at_alignment(aln: dict, s: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    segs: List[Segment] = aln["segments"]
    total = float(aln["length"])

    s = max(0.0, min(float(s), total))

    rem = s
    for seg in segs:
        L = float(seg.length)
        if rem > L:
            rem -= L
            continue

        if isinstance(seg, LineSeg):
            t = 0.0 if L <= 0 else rem / L
            x = seg.x0 + t * (seg.x1 - seg.x0)
            y = seg.y0 + t * (seg.y1 - seg.y0)
            tx = (seg.x1 - seg.x0)
            ty = (seg.y1 - seg.y0)
            nrm = math.hypot(tx, ty) or 1.0
            return (float(x), float(y)), (float(tx / nrm), float(ty / nrm))

        if isinstance(seg, ArcSeg):
            dtheta = 0.0 if seg.r <= 0 else (rem / seg.r)
            if seg.rot == "ccw":
                ang = seg.a0 + dtheta
                tan_ang = ang + math.pi / 2.0
            else:
                ang = seg.a0 - dtheta
                tan_ang = ang - math.pi / 2.0

            x = seg.cx + seg.r * math.cos(ang)
            y = seg.cy + seg.r * math.sin(ang)
            tx = math.cos(tan_ang)
            ty = math.sin(tan_ang)
            nrm = math.hypot(tx, ty) or 1.0
            return (float(x), float(y)), (float(tx / nrm), float(ty / nrm))

        break

    last = segs[-1]
    if isinstance(last, LineSeg):
        tx, ty = (last.x1 - last.x0), (last.y1 - last.y0)
        nrm = math.hypot(tx, ty) or 1.0
        return (float(last.x1), float(last.y1)), (float(tx / nrm), float(ty / nrm))
    if isinstance(last, ArcSeg):
        ang = last.a1
        tan_ang = ang + (math.pi / 2.0 if last.rot == "ccw" else -math.pi / 2.0)
        x = last.cx + last.r * math.cos(ang)
        y = last.cy + last.r * math.sin(ang)
        return (float(x), float(y)), (float(math.cos(tan_ang)), float(math.sin(tan_ang)))

    return (0.0, 0.0), (1.0, 0.0)


# ============================================================
# Main PK table computation
# - Width_m: measured REAL ditch edge width (from find_edges_and_depth)
# - Width_calc_m: theoretical top width from slope+bottom+depth (debug/compare)
# - Area/Volume: trapezoid from bottom+depth+slope (stable for Civil comparison)
# ============================================================

def compute_pk_table(
    idx: TinIndex,
    aln: dict,
    pk_step: float,
    cross_len: float,
    sample_step: float,
    tol: float,
    min_run: float,
    min_depth_from_bottom: float,
    slope_text: str,
    bottom_w: float,
):
    if pk_step <= 0:
        raise ValueError("PK samm peab olema > 0.")
    if cross_len <= 0 or sample_step <= 0:
        raise ValueError("Ristlõike pikkus ja proovipunkti samm peavad olema > 0.")

    slope_hv = parse_slope_ratio(slope_text)
    total_len = float(aln["length"])

    count = int(math.floor(total_len / pk_step))
    if count <= 0:
        raise ValueError("Telg on lühem kui PK samm.")

    half = cross_len / 2.0
    rows = []
    total_volume = 0.0

    for k in range(1, count + 1):
        s = k * pk_step
        (px, py), (tx, ty) = point_and_tangent_at_alignment(aln, s)

        # normal (left/right)
        nx, ny = -ty, tx

        x1 = px - nx * half
        y1 = py - ny * half
        x2 = px + nx * half
        y2 = py + ny * half

        ds, zs = sample_profile(idx, x1, y1, x2, y2, step=sample_step)
        pk_label = pk_fmt(int(round(s)))

        info = find_edges_and_depth(
            ds, zs,
            tol=tol,
            min_run=min_run,
            sample_step=sample_step,
            min_depth_from_bottom=min_depth_from_bottom,
            center_window_m=min(6.0, cross_len / 3.0),
            max_edge_dist_m=None,          # kui tahad piiri: nt 12.0
            slope_threshold=0.12,
            slope_smooth_window_m=0.5,
            slope_min_run_m=0.7,
        )

        if info is None:
            rows.append({
                "PK": pk_label,
                "Width_m": None,
                "Width_calc_m": None,
                "Depth_m": None,
                "EdgeZ": None,
                "BottomZ": None,
                "Area_m2": None,
                "Vol_m3": None,
            })
            continue

        width = float(info["width"])
        depth = float(info["depth"])
        edgez = float(info["edge_z"])
        bottomz = float(info["bottom_z"])

        width_calc = float(bottom_w + 2.0 * slope_hv * depth)  # compare with Civil theoretical

        A = float(area_trapezoid(bottom_w=bottom_w, depth=depth, slope_h_over_v=slope_hv))

        if k < count:
            ds_len = pk_step
        else:
            ds_len = total_len - (pk_step * (count - 1))
            ds_len = max(ds_len, 0.0)

        V = float(A * ds_len)
        total_volume += V

        rows.append({
            "PK": pk_label,
            "Width_m": width,              # REAL ditch edge width
            "Width_calc_m": width_calc,    # theoretical (debug)
            "Depth_m": depth,
            "EdgeZ": edgez,
            "BottomZ": bottomz,
            "Area_m2": A,
            "Vol_m3": V,
        })

    return rows, float(total_volume), float(total_len), int(count)
