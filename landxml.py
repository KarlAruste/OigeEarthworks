# landxml.py
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ============================================================
# XML helpers
# ============================================================
def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _as_floats(text: str) -> List[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


# ============================================================
# LandXML read: points + faces
# - Supports <Pnts><P id=".."> ... </P> and <Faces><F>...</F>
# - Also supports fallback <PntList3D> ... </PntList3D>
# - ALWAYS returns points in (E,N,Z)
# ============================================================
def read_landxml_tin_from_bytes(xml_bytes: bytes) -> Tuple[Dict[int, Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    root = ET.fromstring(xml_bytes)

    raw: List[Tuple[int, float, float, float]] = []  # (pid, a, b, z) where a/b are first two numbers as-is
    faces: List[Tuple[int, int, int]] = []

    # --- 1) Preferred: <P id="..">a b z</P>
    for el in root.iter():
        if _strip(el.tag).lower() == "p":
            pid = el.attrib.get("id")
            txt = (el.text or "").strip()
            if not pid or not txt:
                continue
            vals = _as_floats(txt)
            if len(vals) < 3:
                continue
            try:
                raw.append((int(pid), float(vals[0]), float(vals[1]), float(vals[2])))
            except Exception:
                pass

    # --- 2) Faces <F>a b c</F> (optional)
    for el in root.iter():
        if _strip(el.tag).lower() == "f":
            txt = (el.text or "").strip()
            if not txt:
                continue
            parts = txt.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    faces.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except Exception:
                    pass

    # --- 3) Fallback: <PntList3D> a b z a b z ... </PntList3D>
    if not raw:
        for el in root.iter():
            if _strip(el.tag).lower() == "pntlist3d":
                txt = (el.text or "").strip()
                if not txt:
                    continue
                vals = _as_floats(txt)
                if len(vals) >= 3:
                    pid = 1
                    for i in range(0, len(vals) - 2, 3):
                        raw.append((pid, float(vals[i]), float(vals[i + 1]), float(vals[i + 2])))
                        pid += 1
                break

    if not raw:
        return {}, []

    # --- Global decision: are coordinates (N,E) or (E,N)?
    A = np.array([r[1] for r in raw], dtype=float)
    B = np.array([r[2] for r in raw], dtype=float)
    a_med = float(np.median(A))
    b_med = float(np.median(B))

    # Typical Estonia: N ~ 6.5M, E ~ 0.6M → if first is huge and second smaller → swap
    swap_all = (a_med > 2_000_000 and b_med < 2_000_000)

    pts: Dict[int, Tuple[float, float, float]] = {}
    for pid, a, b, z in raw:
        if swap_all:
            e, n = b, a
        else:
            e, n = a, b
        pts[int(pid)] = (float(e), float(n), float(z))

    return pts, faces


# ============================================================
# Local coordinate utilities (plotting)
# ============================================================
def compute_local_origin_EN(xyz_ENZ: np.ndarray) -> Tuple[float, float]:
    """
    xyz_ENZ: Nx3 array [E,N,Z]
    Use median as stable origin.
    """
    E0 = float(np.median(xyz_ENZ[:, 0]))
    N0 = float(np.median(xyz_ENZ[:, 1]))
    return E0, N0


def to_local_EN(E: float, N: float, E0: float, N0: float) -> Tuple[float, float]:
    return float(E - E0), float(N - N0)


def to_abs_EN(x: float, y: float, E0: float, N0: float) -> Tuple[float, float]:
    return float(x + E0), float(y + N0)


# ============================================================
# Fast spatial index for nearest point and triangle lookup
# ============================================================
@dataclass
class TinIndex:
    tris: np.ndarray                 # (T,3,3) float, each vertex (E,N,Z)
    minx: float
    miny: float
    cell: float
    tri_buckets: dict                # (ix,iy)->[tri_idx,...]
    pt_buckets: dict                 # (ix,iy)->[(E,N,Z),...]


def build_tin_index(pts: Dict[int, Tuple[float, float, float]],
                    faces: List[Tuple[int, int, int]],
                    target_bucket_count: int = 150_000) -> TinIndex:
    if faces:
        tris = np.empty((len(faces), 3, 3), dtype=float)
        for i, (a, b, c) in enumerate(faces):
            tris[i, 0, :] = pts[a]
            tris[i, 1, :] = pts[b]
            tris[i, 2, :] = pts[c]
        xs = tris[:, :, 0]
        ys = tris[:, :, 1]
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
    else:
        arr = np.array(list(pts.values()), dtype=float)
        tris = np.empty((0, 3, 3), dtype=float)
        minx, maxx = float(arr[:, 0].min()), float(arr[:, 0].max())
        miny, maxy = float(arr[:, 1].min()), float(arr[:, 1].max())

    area = max((maxx - minx) * (maxy - miny), 1e-9)
    cell = math.sqrt(area / max(target_bucket_count, 10_000))
    cell = max(cell, 0.5)

    tri_buckets = {}
    if faces:
        tri_bbox = np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1)
        for i, bb in enumerate(tri_bbox):
            ix0 = int((bb[0] - minx) // cell)
            iy0 = int((bb[1] - miny) // cell)
            ix1 = int((bb[2] - minx) // cell)
            iy1 = int((bb[3] - miny) // cell)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    tri_buckets.setdefault((ix, iy), []).append(i)

    pt_buckets = {}
    for (e, n, z) in pts.values():
        ix = int((e - minx) // cell)
        iy = int((n - miny) // cell)
        pt_buckets.setdefault((ix, iy), []).append((float(e), float(n), float(z)))

    return TinIndex(tris=tris, minx=minx, miny=miny, cell=cell, tri_buckets=tri_buckets, pt_buckets=pt_buckets)


def nearest_point_xyz(idx: TinIndex, px: float, py: float, max_rings: int = 10) -> Optional[Tuple[float, float, float]]:
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


# ============================================================
# Z at XY (triangulation if available, else nearest point)
# ============================================================
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


def z_at_xy(idx: TinIndex, px: float, py: float) -> Optional[float]:
    # try triangles first
    if idx.tris.shape[0] > 0:
        ix = int((px - idx.minx) // idx.cell)
        iy = int((py - idx.miny) // idx.cell)

        candidates = []
        for dx in (0, -1, 1):
            for dy in (0, -1, 1):
                candidates += idx.tri_buckets.get((ix + dx, iy + dy), [])

        for ti in candidates:
            A, B, C = idx.tris[ti]
            if _point_in_tri_2d(px, py, A[0], A[1], B[0], B[1], C[0], C[1]):
                z = _interp_z(px, py, A, B, C)
                if z is not None:
                    return float(z)

    # fallback nearest point
    nz = nearest_point_xyz(idx, px, py, max_rings=10)
    return float(nz[2]) if nz is not None else None


# ============================================================
# Profile sampling + edge/bottom detection
# ============================================================
def sample_profile(idx: TinIndex, x1, y1, x2, y2, step: float):
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
            zs[i] = float(z)
    return ds, zs


def find_edges_and_depth(
    ds: np.ndarray,
    zs: np.ndarray,
    tol: float,
    min_run: float,
    sample_step: float,
    min_depth_from_bottom: float = 0.3,
    center_window_m: float = 6.0,
):
    valid = np.isfinite(zs)
    if valid.sum() < 5:
        return None

    n = len(ds)
    mid = n // 2

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

    if left_center is None or right_center is None:
        return None

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
        "left_edge_z": float(left_edge_z),
        "right_edge_z": float(right_edge_z),
    }


# ============================================================
# Slope + area + pk formatting
# ============================================================
def parse_slope_ratio(text: str) -> float:
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


def pk_fmt(meters: int) -> str:
    km = meters // 1000
    m = meters % 1000
    if meters < 100:
        return f"{km}+{m:02d}"
    return f"{km}+{m:03d}"


# ============================================================
# Polyline utils (ABS)
# ============================================================
def polyline_length(xy: List[Tuple[float, float]]) -> float:
    if not xy or len(xy) < 2:
        return 0.0
    L = 0.0
    for i in range(len(xy) - 1):
        dx = xy[i + 1][0] - xy[i][0]
        dy = xy[i + 1][1] - xy[i][1]
        L += math.hypot(dx, dy)
    return float(L)


def _polyline_cumlen(xy: List[Tuple[float, float]]) -> np.ndarray:
    cum = [0.0]
    for i in range(len(xy) - 1):
        dx = xy[i + 1][0] - xy[i][0]
        dy = xy[i + 1][1] - xy[i][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    return np.array(cum, dtype=float)


def _point_and_tangent_at(xy: List[Tuple[float, float]], s: float):
    cum = _polyline_cumlen(xy)
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


# ============================================================
# Main: PK table compute (ABS axis, TIN from LandXML)
# ============================================================
def compute_pk_table_from_landxml(
    xml_bytes: bytes,
    axis_xy: List[Tuple[float, float]],  # ABS (E,N)
    pk_step: float,
    cross_len: float,
    sample_step: float,
    tol: float,
    min_run: float,
    min_depth_from_bottom: float,
    slope_text: str,
    bottom_w: float,
):
    if not axis_xy or len(axis_xy) < 2:
        raise ValueError("Telg peab olema vähemalt 2 punktiga.")

    pts, faces = read_landxml_tin_from_bytes(xml_bytes)
    if not pts:
        raise ValueError("LandXML-ist ei leitud punkte.")

    idx = build_tin_index(pts, faces)
    axis_len = polyline_length(axis_xy)
    if axis_len <= 0:
        raise ValueError("Telje pikkus on 0.")

    count = int(math.floor(axis_len / pk_step))
    if count <= 0:
        raise ValueError("Telg on lühem kui PK samm.")

    slope_hv = parse_slope_ratio(slope_text)
    half = cross_len / 2.0

    rows = []
    total_v = 0.0

    for k in range(1, count + 1):
        s = k * pk_step
        (px, py), (tx, ty) = _point_and_tangent_at(axis_xy, s)

        nx, ny = -ty, tx  # normal

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
        )

        pk_label = pk_fmt(int(round(s)))

        if info is None:
            rows.append({
                "PK": pk_label,
                "width_m": None,
                "depth_m": None,
                "edge_z": None,
                "bottom_z": None,
                "area_m2": None,
                "volume_m3": None,
            })
            continue

        width = float(info["width"])
        depth = float(info["depth"])
        edge_z = float(info["edge_z"])
        bottom_z = float(info["bottom_z"])

        A = float(area_trapezoid(bottom_w=bottom_w, depth=depth, slope_h_over_v=slope_hv))
        V = float(A * pk_step)
        total_v += V

        rows.append({
            "PK": pk_label,
            "width_m": width,
            "depth_m": depth,
            "edge_z": edge_z,
            "bottom_z": bottom_z,
            "area_m2": A,
            "volume_m3": V,
        })

    return {
        "rows": rows,
        "total_volume_m3": float(total_v),
        "axis_length_m": float(axis_len),
        "count": int(count),
    }
