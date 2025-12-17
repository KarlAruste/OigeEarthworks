import xml.etree.ElementTree as ET
import numpy as np

NS = {"lx": "http://www.landxml.org/schema/LandXML-1.2"}

def _read_points_xyz(root):
    pts = []
    for p in root.findall(".//lx:Pnts/lx:P", NS):
        txt = (p.text or "").strip()
        if not txt:
            continue
        parts = txt.split()
        if len(parts) < 3:
            continue
        try:
            a, b, z = float(parts[0]), float(parts[1]), float(parts[2])
            pts.append((a, b, z))
        except:
            pass
    return np.array(pts, dtype=float) if pts else None

def _solve_from_points(
    pts: np.ndarray,
    n_bins: int,
    slice_thickness_ratio: float,
    edge_tail_pct: float,
    trim_inner_pct: float = 1.0,
    min_points_per_slice: int = 120,
):
    XY = pts[:, :2]
    Z = pts[:, 2]

    XYc = XY - XY.mean(axis=0)
    cov = np.cov(XYc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]

    s = XYc @ main_dir
    perp = np.array([-main_dir[1], main_dir[0]])
    t = XYc @ perp

    s_min, s_max = float(s.min()), float(s.max())
    length = s_max - s_min
    if length <= 0:
        return None

    thick = max(length * slice_thickness_ratio, 0.5)
    centers = np.linspace(s_min, s_max, n_bins)

    areas = []
    valid_slices = 0

    for c in centers:
        mask = np.abs(s - c) <= thick
        if mask.sum() < min_points_per_slice:
            continue

        tt = t[mask]
        zz = Z[mask]

        # trim outliers in t
        t_lo = np.percentile(tt, trim_inner_pct)
        t_hi = np.percentile(tt, 100.0 - trim_inner_pct)
        m2 = (tt >= t_lo) & (tt <= t_hi)
        tt = tt[m2]
        zz = zz[m2]
        if len(tt) < min_points_per_slice:
            continue

        left_thr = np.percentile(tt, edge_tail_pct)
        right_thr = np.percentile(tt, 100.0 - edge_tail_pct)

        left = (tt <= left_thr)
        right = (tt >= right_thr)
        if left.sum() < 10 or right.sum() < 10:
            continue

        tL = float(np.median(tt[left]))
        zL = float(np.median(zz[left]))
        tR = float(np.median(tt[right]))
        zR = float(np.median(zz[right]))
        if abs(tR - tL) < 0.5:
            continue

        order = np.argsort(tt)
        tt_s = tt[order]
        zz_s = zz[order]

        top_line = zL + (zR - zL) * ((tt_s - tL) / (tR - tL))
        depth = np.maximum(0.0, top_line - zz_s)

        width = float(tt_s.max() - tt_s.min())
        if width < 0.8:
            continue

        area = float(np.trapz(depth, tt_s))
        if area > 0:
            areas.append(area)
            valid_slices += 1

    if not areas:
        return {"length": float(length), "mean_area": None, "volume": None, "valid_slices": valid_slices}

    mean_area = float(np.mean(areas))
    volume = float(float(length) * mean_area)

    return {
        "length": float(length),
        "mean_area": mean_area,
        "volume": volume,
        "valid_slices": valid_slices
    }

def estimate_length_area_volume_from_tin(
    xml_bytes: bytes,
    n_bins: int = 30,
    slice_thickness_ratio: float = 0.03,
    edge_tail_pct: float = 10.0,
):
    """
    Auto-detect:
    - Try points as (X,Y,Z)
    - Try swapped as (Y,X,Z)
    Choose result with more valid slices, then with non-null volume.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, None, None

    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 500:
        return None, None, None

    res_xy = _solve_from_points(pts, n_bins, slice_thickness_ratio, edge_tail_pct)
    pts_swapped = pts.copy()
    pts_swapped[:, 0], pts_swapped[:, 1] = pts[:, 1], pts[:, 0]
    res_yx = _solve_from_points(pts_swapped, n_bins, slice_thickness_ratio, edge_tail_pct)

    candidates = [r for r in [res_xy, res_yx] if r is not None]
    if not candidates:
        return None, None, None

    # choose best: more valid slices; prefer with volume
    candidates.sort(key=lambda r: (r["valid_slices"], r["volume"] is not None), reverse=True)
    best = candidates[0]

    return best["length"], best["mean_area"], best["volume"]
