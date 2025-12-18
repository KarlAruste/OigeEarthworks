# landxml.py
import xml.etree.ElementTree as ET
import numpy as np

NS = {"lx": "http://www.landxml.org/schema/LandXML-1.2"}


def _read_points_xyz(root):
    """
    Reads LandXML points from <Pnts><P>... expecting 3 numbers per point.
    Some exports use (E N Z) or (N E Z). We'll handle swap later.
    Returns Nx3 float array.
    """
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
        except Exception:
            pass
    return np.array(pts, dtype=float) if pts else None


def _compute_profile_area_edge_topline(tt, zz, edge_tail_pct=10.0, edge_z_pct=97.0, min_width=0.8):
    """
    Edge-based topline (kraavi servad = EG):
    - left edge points = lowest tail of t
    - right edge points = highest tail of t
    - zL/zR = high percentile of edge points (captures 'crest')
    - topline = line between (tL,zL) and (tR,zR)
    area = ∫ max(0, topline(t) - z(t)) dt
    """
    order = np.argsort(tt)
    tt_s = tt[order]
    zz_s = zz[order]

    width = float(tt_s.max() - tt_s.min())
    if width < min_width:
        return None

    left_thr = np.percentile(tt_s, edge_tail_pct)
    right_thr = np.percentile(tt_s, 100.0 - edge_tail_pct)

    left = tt_s <= left_thr
    right = tt_s >= right_thr
    if left.sum() < 8 or right.sum() < 8:
        return None

    tL = float(np.median(tt_s[left]))
    tR = float(np.median(tt_s[right]))
    if abs(tR - tL) < 0.5:
        return None

    zL = float(np.percentile(zz_s[left], edge_z_pct))
    zR = float(np.percentile(zz_s[right], edge_z_pct))

    topline = zL + (zR - zL) * ((tt_s - tL) / (tR - tL + 1e-12))
    depth = np.maximum(0.0, topline - zz_s)
    area = float(np.trapz(depth, tt_s))
    return area if area > 0 else None


def _compute_profile_area_top_percentile(tt, zz, top_percentile=95.0, min_width=0.8):
    """
    Fallback:
    assume "top plane" is high percentile of z in slice.
    """
    order = np.argsort(tt)
    tt_s = tt[order]
    zz_s = zz[order]

    width = float(tt_s.max() - tt_s.min())
    if width < min_width:
        return None

    z_top = float(np.percentile(zz_s, top_percentile))
    depth = np.maximum(0.0, z_top - zz_s)
    area = float(np.trapz(depth, tt_s))
    return area if area > 0 else None


def _solve_from_points(
    pts: np.ndarray,
    n_bins: int,
    slice_thickness_ratio: float,
    edge_tail_pct: float,
    edge_z_pct: float,
    top_percentile_fallback: float,
    trim_inner_pct: float = 1.0,
    min_points_per_slice: int = 60,
):
    """
    Core solver returning:
      length, mean_area, volume, valid_slices
    """
    XY = pts[:, :2]
    Z = pts[:, 2]

    XYc = XY - XY.mean(axis=0)
    cov = np.cov(XYc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, np.argmax(eigvals)]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    s = XYc @ axis_dir
    perp = np.array([-axis_dir[1], axis_dir[0]])
    t = XYc @ perp

    s_min, s_max = float(s.min()), float(s.max())
    length = float(s_max - s_min)
    if length <= 0:
        return {"length": None, "mean_area": None, "volume": None, "valid_slices": 0}

    thick = max(length * slice_thickness_ratio, 0.5)
    centers = np.linspace(s_min, s_max, n_bins)

    areas = []
    valid = 0

    for c in centers:
        mask = np.abs(s - c) <= thick
        if mask.sum() < min_points_per_slice:
            continue

        tt = t[mask]
        zz = Z[mask]

        # trim t outliers (stabilize edges)
        t_lo = np.percentile(tt, trim_inner_pct)
        t_hi = np.percentile(tt, 100.0 - trim_inner_pct)
        m2 = (tt >= t_lo) & (tt <= t_hi)
        tt = tt[m2]
        zz = zz[m2]

        if len(tt) < min_points_per_slice:
            continue

        area = _compute_profile_area_edge_topline(
            tt, zz,
            edge_tail_pct=edge_tail_pct,
            edge_z_pct=edge_z_pct,
        )
        if area is None:
            area = _compute_profile_area_top_percentile(
                tt, zz,
                top_percentile=top_percentile_fallback,
            )

        if area is not None:
            areas.append(area)
            valid += 1

    if not areas:
        return {"length": length, "mean_area": None, "volume": None, "valid_slices": valid}

    mean_area = float(np.mean(areas))
    volume = float(length * mean_area)
    return {"length": length, "mean_area": mean_area, "volume": volume, "valid_slices": valid}


def estimate_length_area_volume_from_tin(
    xml_bytes: bytes,
    n_bins: int = 40,
    slice_thickness_ratio: float = 0.04,
    edge_tail_pct: float = 10.0,
    edge_z_pct: float = 97.0,
    top_percentile_fallback: float = 95.0,
):
    """
    Main API used by app.
    Auto-detect XY vs swapped YX.
    Returns: (length_m, mean_area_m2, volume_m3)
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, None, None

    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 300:
        return None, None, None

    res_xy = _solve_from_points(
        pts, n_bins, slice_thickness_ratio,
        edge_tail_pct, edge_z_pct, top_percentile_fallback
    )

    pts_swapped = pts.copy()
    pts_swapped[:, 0], pts_swapped[:, 1] = pts[:, 1], pts[:, 0]
    res_yx = _solve_from_points(
        pts_swapped, n_bins, slice_thickness_ratio,
        edge_tail_pct, edge_z_pct, top_percentile_fallback
    )

    candidates = [("xy", res_xy), ("yx", res_yx)]
    candidates.sort(key=lambda x: (x[1]["valid_slices"], x[1]["volume"] is not None), reverse=True)
    best_name, best = candidates[0]

    return best["length"], best["mean_area"], best["volume"]


def landxml_debug_views(
    xml_bytes: bytes,
    n_bins: int = 40,
    slice_thickness_ratio: float = 0.04,
    edge_tail_pct: float = 10.0,
    edge_z_pct: float = 97.0,
    top_percentile_fallback: float = 95.0,
    sample_slices: int = 3,
):
    """
    For visualization in Streamlit (map + few cross-sections).
    Returns dict:
      {
        pts_xy: Nx2,
        axis_dir: (2,),
        length_m: float,
        samples: [{t,z,top,area}, ...]
      }
    """
    root = ET.fromstring(xml_bytes)
    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 300:
        return None

    # pick best orientation same as estimate()
    res_xy = _solve_from_points(pts, n_bins, slice_thickness_ratio, edge_tail_pct, edge_z_pct, top_percentile_fallback)
    pts_swapped = pts.copy()
    pts_swapped[:, 0], pts_swapped[:, 1] = pts[:, 1], pts[:, 0]
    res_yx = _solve_from_points(pts_swapped, n_bins, slice_thickness_ratio, edge_tail_pct, edge_z_pct, top_percentile_fallback)

    use_swapped = False
    if (res_yx["valid_slices"] > res_xy["valid_slices"]) or (
        res_yx["valid_slices"] == res_xy["valid_slices"] and res_yx["volume"] is not None and res_xy["volume"] is None
    ):
        use_swapped = True

    best_pts = pts_swapped if use_swapped else pts
    XY = best_pts[:, :2]
    Z = best_pts[:, 2]

    # PCA axis
    XYc = XY - XY.mean(axis=0)
    cov = np.cov(XYc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, np.argmax(eigvals)]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    s = XYc @ axis_dir
    perp = np.array([-axis_dir[1], axis_dir[0]])
    t = XYc @ perp

    s_min, s_max = float(s.min()), float(s.max())
    length = float(s_max - s_min)
    if length <= 0:
        return None

    thick = max(length * slice_thickness_ratio, 0.5)
    centers = np.linspace(s_min, s_max, n_bins)
    pick_idxs = np.linspace(0, len(centers) - 1, sample_slices).astype(int)

    samples = []
    for idx in pick_idxs:
        c = centers[idx]
        mask = np.abs(s - c) <= thick
        if mask.sum() < 60:
            continue

        tt = t[mask]
        zz = Z[mask]

        # trim
        t_lo = np.percentile(tt, 1.0)
        t_hi = np.percentile(tt, 99.0)
        m2 = (tt >= t_lo) & (tt <= t_hi)
        tt = tt[m2]
        zz = zz[m2]
        if len(tt) < 60:
            continue

        order = np.argsort(tt)
        tt_s = tt[order]
        zz_s = zz[order]

        # build topline (edge-topline -> else fallback flat)
        left_thr = np.percentile(tt_s, edge_tail_pct)
        right_thr = np.percentile(tt_s, 100.0 - edge_tail_pct)
        left = tt_s <= left_thr
        right = tt_s >= right_thr

        if left.sum() < 8 or right.sum() < 8:
            z_top = float(np.percentile(zz_s, top_percentile_fallback))
            top = np.full_like(tt_s, z_top)
        else:
            tL = float(np.median(tt_s[left]))
            tR = float(np.median(tt_s[right]))
            zL = float(np.percentile(zz_s[left], edge_z_pct))
            zR = float(np.percentile(zz_s[right], edge_z_pct))
            top = zL + (zR - zL) * ((tt_s - tL) / (tR - tL + 1e-12))

        depth = np.maximum(0.0, top - zz_s)
        area = float(np.trapz(depth, tt_s))

        samples.append({"t": tt_s, "z": zz_s, "top": top, "area": area})

    return {
        "pts_xy": XY,
        "axis_dir": axis_dir,
        "length_m": length,
        "samples": samples,
    }
