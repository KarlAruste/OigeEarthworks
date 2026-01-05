from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ============================================================
# LandXML parse (TIN: Pnts + Faces)
# ============================================================
def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag
    


def read_landxml_tin_from_bytes(xml_bytes: bytes):
    """
    Returns:
      pts_dict: {point_id:int -> (x,y,z)}
      faces: list[(a_id,b_id,c_id)] using point IDs
    """
    root = ET.fromstring(xml_bytes)

    pts: Dict[int, Tuple[float, float, float]] = {}
    faces: List[Tuple[int, int, int]] = []

    # Points: <P id="123"> x y z </P>
    for el in root.iter():
        if _strip(el.tag) == "P":
            pid = el.attrib.get("id")
            if not pid:
                continue
            txt = (el.text or "").strip()
            if not txt:
                continue
            parts = txt.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                pts[int(pid)] = (x, y, z)
            except Exception:
                continue

    # Faces: <F> id1 id2 id3 </F>
    for el in root.iter():
        if _strip(el.tag) == "F":
            txt = (el.text or "").strip()
            if not txt:
                continue
            parts = txt.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
                faces.append((a, b, c))
            except Exception:
                continue

    if not pts or not faces:
        raise ValueError("LandXML-ist ei leitud TIN Pnts/Faces (P ja F elemente).")

    # Heuristiline XY swap: kui X näeb välja nagu Northing (6-7M) ja Y nagu Easting (0.3-1.2M)
    xs = np.array([p[0] for p in pts.values()], dtype=float)
    ys = np.array([p[1] for p in pts.values()], dtype=float)
    x_med = float(np.median(xs))
    y_med = float(np.median(ys))
    if x_med > 2_000_000 and y_med < 2_000_000:
        pts = {pid: (y, x, z) for pid, (x, y, z) in pts.items()}

    return pts, faces


# ============================================================
# TIN geometry (point-in-triangle + z interpolation)
# ============================================================
def point_in_tri_2d(px, py, ax, ay, bx, by, cx, cy) -> bool:
    def s(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    b1 = s(px, py, ax, ay, bx, by) < 0.0
    b2 = s(px, py, bx, by, cx, cy) < 0.0
    b3 = s(px, py, cx, cy, ax, ay) < 0.0
    return (b1 == b2) and (b2 == b3)


def interp_z(px, py, A, B, C) -> Optional[float]:
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


# ============================================================
# Spatial index for triangles + points
# ============================================================
@dataclass
class TinIndex:
    tris: np.ndarray          # (T,3,3)
    minx: float
    miny: float
    cell: float
    tri_buckets: Dict[Tuple[int, int], List[int]]
    pt_buckets: Dict[Tuple[int, int], List[Tuple[float, float, float]]]


def build_tin_index(pts: Dict[int, Tuple[float, float, float]],
                    faces: List[Tuple[int, int, int]],
                    target_bucket_count: int = 150_000) -> TinIndex:
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
    tri_buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, bb in enumerate(tri_bbox):
        ix0 = int((bb[0] - minx) // cell)
        iy0 = int((bb[1] - miny) // cell)
        ix1 = int((bb[2] - minx) // cell)
        iy1 = int((bb[3] - miny) // cell)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                tri_buckets.setdefault((ix, iy), []).append(i)

    pt_buckets: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    for x, y, z in pts.values():
        ix = int((x - minx) // cell)
        iy = int((y - miny) // cell)
        pt_buckets.setdefault((ix, iy), []).append((float(x), float(y), float(z)))

    return TinIndex(tris=tris, minx=minx, miny=miny, cell=cell, tri_buckets=tri_buckets, pt_buckets=pt_buckets)


def nearest_point_z(idx: TinIndex, px: float, py: float, max_rings: int = 6) -> Optional[float]:
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


def z_at_xy(idx: TinIndex, px: float, py: float) -> Optional[float]:
    ix = int((px - idx.minx) // idx.cell)
    iy = int((py - idx.miny) // idx.cell)

    candidates: List[int] = []
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


# ============================================================
# Ristlõige + serv/põhi + sügavus
# ============================================================
def sample_profile(idx: TinIndex, x1: float, y1: float, x2: float, y2: float, step: float):
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


def find_edges_and_depth(ds: np.ndarray, zs: np.ndarray,
                         tol: float, min_run: float, sample_step: float,
                         min_depth_from_bottom: float = 0.3,
                         center_window_m: float = 6.0) -> Optional[dict]:
    """
    Tagastab:
      {width, depth, bottom_z, edge_z, left_edge_z, right_edge_z}
    """
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

    if left_center is None or right_center is None or left_edge_z is None or right_edge_z is None:
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
# Slope + area/volume
# ============================================================
def parse_slope_ratio(text: str) -> float:
    """
    '1:2' -> 2.0 (H/V)
    '2' -> 2.0
    """
    t = (text or "").strip().replace(" ", "")
    if not t:
        raise ValueError("Nõlva kalle puudu (nt 1:2).")
    if ":" in t:
        a, b = t.split(":", 1)
        v = float(a)
        h = float(b)
        if v == 0:
            raise ValueError("Nõlva kalle: vertikaal ei tohi olla 0.")
        return h / v
    return float(t)


def area_trapezoid(bottom_w: float, depth: float, slope_h_over_v: float) -> float:
    top_w = bottom_w + 2.0 * slope_h_over_v * depth
    return (bottom_w + top_w) * 0.5 * depth


# ============================================================
# Polyline helpers + PK format
# ============================================================
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


# ============================================================
# PUBLIC: compute PK table from LandXML + axis polyline
# ============================================================
def compute_pk_table_from_landxml(
    xml_bytes: bytes,
    axis_xy: List[Tuple[float, float]],
    pk_step: float = 1.0,
    cross_len: float = 25.0,
    sample_step: float = 0.1,
    tol: float = 0.05,
    min_run: float = 0.2,
    min_depth_from_bottom: float = 0.3,
    slope_text: str = "1:2",
    bottom_w: float = 0.4,
) -> dict:
    """
    Returns:
      {
        "axis_length_m": ...,
        "rows": [ {pk, station_m, width_m, depth_m, edge_z, bottom_z, area_m2, vol_m3}, ... ],
        "total_volume_m3": ...,
        "count": ...
      }
    """
    if len(axis_xy) < 2:
        raise ValueError("Telg (axis_xy) peab sisaldama vähemalt 2 punkti.")

    pts, faces = read_landxml_tin_from_bytes(xml_bytes)
    idx = build_tin_index(pts, faces)

    axis_len = polyline_length(axis_xy)
    if axis_len <= 0:
        raise ValueError("Telg on liiga lühike.")

    if pk_step <= 0 or cross_len <= 0 or sample_step <= 0:
        raise ValueError("Sammud/pikkused peavad olema > 0.")

    slope_hv = parse_slope_ratio(slope_text)
    if bottom_w < 0:
        raise ValueError("Põhja laius peab olema >= 0.")

    count = int(math.floor(axis_len / pk_step))
    half = cross_len / 2.0

    rows: List[dict] = []
    total_v = 0.0

    for k in range(1, count + 1):
        station = k * pk_step
        (px, py), (tx, ty) = point_and_tangent_at(axis_xy, station)

        nx, ny = -ty, tx  # normal (perp)

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
            center_window_m=min(6.0, cross_len / 3.0),
        )

        pk = pk_fmt(int(round(station)))

        if info is None:
            rows.append({
                "pk": pk,
                "station_m": float(station),
                "width_m": None,
                "depth_m": None,
                "edge_z": None,
                "bottom_z": None,
                "area_m2": None,
                "vol_m3": None,
            })
            continue

        width = float(info["width"])
        depth = float(info["depth"])
        edge_z = float(info["edge_z"])
        bottom_z = float(info["bottom_z"])

        area = float(area_trapezoid(bottom_w=bottom_w, depth=depth, slope_h_over_v=slope_hv))
        vol = float(area * pk_step)

        total_v += vol

        rows.append({
            "pk": pk,
            "station_m": float(station),
            "width_m": width,
            "depth_m": depth,
            "edge_z": edge_z,
            "bottom_z": bottom_z,
            "area_m2": area,
            "vol_m3": vol,
        })

    return {
        "axis_length_m": float(axis_len),
        "rows": rows,
        "total_volume_m3": float(total_v),
        "count": int(count),
    }
