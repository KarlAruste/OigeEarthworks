# landxml.py
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_float_list(text: str) -> List[float]:
    parts = (text or "").replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def _maybe_swap_xy(xyz: np.ndarray) -> np.ndarray:
    """
    Sinu näites LandXML punktid on (N, E, Z):
      N ~ 6-7 miljonit, E ~ 300k-800k
    Plotis tahame (E, N, Z) => X=E, Y=N.
    """
    if xyz is None or xyz.size == 0:
        return xyz
    x_med = float(np.median(xyz[:, 0]))
    y_med = float(np.median(xyz[:, 1]))
    if x_med > 2_000_000 and y_med < 2_000_000:
        xyz = xyz[:, [1, 0, 2]]
    return xyz


def polyline_length(axis_xy: List[Tuple[float, float]]) -> float:
    if not axis_xy or len(axis_xy) < 2:
        return 0.0
    arr = np.asarray(axis_xy, dtype=float)
    d = np.sqrt(np.sum((arr[1:] - arr[:-1]) ** 2, axis=1))
    return float(np.sum(d))


def _point_and_tangent_at_chainage(axis_xy: List[Tuple[float, float]], s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (point_xy, tangent_unit) at chainage s along polyline.
    """
    pts = np.asarray(axis_xy, dtype=float)
    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt(np.sum(seg ** 2, axis=1))
    total = float(np.sum(seg_len))
    if total <= 1e-12:
        return pts[0], np.array([1.0, 0.0], dtype=float)

    s = float(np.clip(s, 0.0, total))
    acc = 0.0
    for i, L in enumerate(seg_len):
        if L <= 1e-12:
            continue
        if acc + L >= s:
            t = (s - acc) / L
            p = pts[i] + t * seg[i]
            tan = seg[i] / L
            return p, tan
        acc += L

    # fallback end
    tan = seg[-1] / (seg_len[-1] + 1e-12)
    return pts[-1], tan


def _parse_slope_text(slope_text: str) -> float:
    """
    Accepts: "1:2", "1/2", "2" etc. Returns n in 1:n (horizontal/vertical).
    """
    s = (slope_text or "").strip()
    if not s:
        return 2.0
    m = re.match(r"^\s*1\s*[:/]\s*([0-9]+(\.[0-9]+)?)\s*$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except Exception:
        return 2.0


def _pk_label(meters: float) -> str:
    # 0+01, 0+10, 1+05 (kui üle 100m)
    m = int(round(meters))
    km = m // 1000
    rest = m % 1000
    return f"{km}+{rest:03d}" if km > 0 else f"0+{rest:02d}"


# -----------------------------
# LandXML TIN read
# -----------------------------

def read_landxml_tin_from_bytes(xml_bytes: bytes) -> Tuple[Dict[str, Tuple[float, float, float]], List[Tuple[str, str, str]]]:
    """
    Returns:
      pts_dict: id(str) -> (X, Y, Z)  (after swap => X=E, Y=N)
      faces: list of (id1,id2,id3) strings if present (optional)
    Supports common LandXML:
      - <P id="1"> x y z </P>
      - faces: <F> 1 2 3 </F>
    """
    root = ET.fromstring(xml_bytes)

    pts: Dict[str, Tuple[float, float, float]] = {}
    faces: List[Tuple[str, str, str]] = []

    # Points
    for el in root.iter():
        if _strip(el.tag).lower() == "p":
            pid = el.attrib.get("id") or el.attrib.get("name") or None
            vals = _parse_float_list(el.text or "")
            if len(vals) >= 3:
                x, y, z = vals[0], vals[1], vals[2]
                if pid is None:
                    pid = str(len(pts) + 1)
                pts[str(pid)] = (float(x), float(y), float(z))

    # Faces (optional)
    for el in root.iter():
        if _strip(el.tag).lower() == "f":
            parts = (el.text or "").replace(",", " ").split()
            if len(parts) >= 3:
                faces.append((parts[0], parts[1], parts[2]))

    if not pts:
        raise ValueError("LandXML: punkte ei leitud (<P> elemendid puudu?).")

    # swap XY if needed
    xyz = np.array(list(pts.values()), dtype=float)
    xyz2 = _maybe_swap_xy(xyz)

    if not np.allclose(xyz[:, 0], xyz2[:, 0]) or not np.allclose(xyz[:, 1], xyz2[:, 1]):
        # apply swapped back to dict preserving ids
        keys = list(pts.keys())
        for i, k in enumerate(keys):
            pts[k] = (float(xyz2[i, 0]), float(xyz2[i, 1]), float(xyz2[i, 2]))

    return pts, faces


# -----------------------------
# Height interpolation (IDW)
# -----------------------------

@dataclass
class IDWInterpolator:
    xy: np.ndarray  # (N,2)
    z: np.ndarray   # (N,)

    def query(self, qxy: np.ndarray, k: int = 12, power: float = 2.0) -> np.ndarray:
        """
        Simple IDW using k nearest points (bruteforce, ok for sampled lines).
        qxy: (M,2)
        returns z(M,)
        """
        qxy = np.asarray(qxy, dtype=float)
        out = np.empty((qxy.shape[0],), dtype=float)

        # brute force distances
        # For speed, use chunking if needed
        for i in range(qxy.shape[0]):
            dx = self.xy[:, 0] - qxy[i, 0]
            dy = self.xy[:, 1] - qxy[i, 1]
            d2 = dx * dx + dy * dy
            # take k smallest
            if k < self.xy.shape[0]:
                idx = np.argpartition(d2, k)[:k]
            else:
                idx = np.arange(self.xy.shape[0])

            dd = np.sqrt(d2[idx])
            zz = self.z[idx]

            # if exact point
            if np.any(dd < 1e-9):
                out[i] = float(zz[np.argmin(dd)])
                continue

            w = 1.0 / (dd ** power + 1e-12)
            out[i] = float(np.sum(w * zz) / (np.sum(w) + 1e-12))

        return out


# -----------------------------
# PK table from axis + LandXML
# -----------------------------

def compute_pk_table_from_landxml(
    xml_bytes: bytes,
    axis_xy: List[Tuple[float, float]],

    pk_step: float = 1.0,
    cross_len: float = 25.0,
    sample_step: float = 0.1,

    tol: float = 0.05,
    min_run: float = 0.2,  # not heavily used here (kept for compatibility)
    min_depth_from_bottom: float = 0.3,

    slope_text: str = "1:2",
    bottom_w: float = 0.4,
) -> dict:
    """
    For each PK along axis:
      - create perpendicular cross section line
      - sample heights from TIN point cloud with IDW
      - find bottom_z (low percentile)
      - find left/right edge heights (high percentile on halves)
      - find edge positions where z >= edge_z - tol
      - top_width = t_right - t_left
      - depth = avg(edge_z) - bottom_z  (>= min_depth_from_bottom)
      - area = b*d + n*d^2
      - volume_segment = area * pk_step
    """
    if not axis_xy or len(axis_xy) < 2:
        raise ValueError("Axis polyline is empty or too short.")

    pts_dict, _faces = read_landxml_tin_from_bytes(xml_bytes)
    xyz = np.array(list(pts_dict.values()), dtype=float)
    xyz = _maybe_swap_xy(xyz)  # safety (already applied in reader)

    # Build interpolator
    xy = xyz[:, :2]
    z = xyz[:, 2]
    # filter finite
    m = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & np.isfinite(z)
    xy = xy[m]
    z = z[m]
    if xy.shape[0] < 50:
        raise ValueError("Liiga vähe kehtivaid TIN punkte (finite).")

    interp = IDWInterpolator(xy=xy, z=z)

    axis_len = polyline_length(axis_xy)
    n_slope = _parse_slope_text(slope_text)

    # stations
    if pk_step <= 0:
        pk_step = 1.0
    stations = np.arange(0.0, axis_len + 1e-9, pk_step, dtype=float)

    half = cross_len / 2.0
    if sample_step <= 0:
        sample_step = 0.1
    t_vals = np.arange(-half, half + 1e-9, sample_step, dtype=float)

    rows = []
    total_vol = 0.0

    for s in stations:
        pxy, tan = _point_and_tangent_at_chainage(axis_xy, s)
        # perp
        perp = np.array([-tan[1], tan[0]], dtype=float)
        perp = perp / (np.linalg.norm(perp) + 1e-12)

        # sample points along cross section
        qxy = pxy.reshape(1, 2) + t_vals.reshape(-1, 1) * perp.reshape(1, 2)
        zz = interp.query(qxy, k=12, power=2.0)

        # bottom robust (low percentile)
        bottom_z = float(np.percentile(zz, 2.0))

        # split halves
        mid = len(t_vals) // 2
        left_zz = zz[:mid]
        left_tt = t_vals[:mid]
        right_zz = zz[mid:]
        right_tt = t_vals[mid:]

        # edge z robust high percentile on each side
        left_edge_z = float(np.percentile(left_zz, 90.0))
        right_edge_z = float(np.percentile(right_zz, 90.0))
        edge_z_avg = 0.5 * (left_edge_z + right_edge_z)

        depth = edge_z_avg - bottom_z
        if depth < min_depth_from_bottom:
            # too shallow -> ignore / set None-like values
            top_width = None
            area = None
            vol = None
        else:
            # edge positions: first/last t where z is within tol of edge height
            # left: from center to left find last point >= left_edge_z - tol
            mask_left = left_zz >= (left_edge_z - tol)
            if np.any(mask_left):
                t_left = float(left_tt[np.where(mask_left)[0][-1]])
            else:
                t_left = float(left_tt[-1])  # fallback

            # right: from center to right find first point >= right_edge_z - tol
            mask_right = right_zz >= (right_edge_z - tol)
            if np.any(mask_right):
                t_right = float(right_tt[np.where(mask_right)[0][0]])
            else:
                t_right = float(right_tt[0])  # fallback

            top_width = float(t_right - t_left)

            # area (design) from depth and user params
            area = float(bottom_w * depth + n_slope * depth * depth)
            vol = float(area * pk_step)
            total_vol += vol

        rows.append({
            "PK": _pk_label(s),
            "Chainage_m": float(s),
            "TopWidth_m": None if top_width is None else round(top_width, 3),
            "BottomZ_m": round(bottom_z, 3),
            "EdgeZ_L_m": round(left_edge_z, 3),
            "EdgeZ_R_m": round(right_edge_z, 3),
            "Depth_m": None if depth < min_depth_from_bottom else round(depth, 3),
            "Area_m2": None if area is None else round(area, 3),
            "VolumeSeg_m3": None if vol is None else round(vol, 3),
        })

    return {
        "axis_length_m": float(axis_len),
        "count": int(len(rows)),
        "total_volume_m3": float(total_vol),
        "rows": rows,
    }
