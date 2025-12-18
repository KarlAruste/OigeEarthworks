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
        except Exception:
            pass
    return np.array(pts, dtype=float) if pts else None


def estimate_top_width_from_tin(
    xml_bytes: bytes,
    n_bins: int = 40,
    slice_thickness_ratio: float = 0.04,
    edge_tail_pct: float = 10.0,
):
    """
    Hinnang kraavi pealmisele laiusele (EG–EG) TIN-ist.
    Tagastab: (top_width_m, valid_slices)

    Meetod:
    - PCA -> peatelg (s)
    - risti telg (t)
    - slice'ides võtame t-jaotuse vasaku/parema tail -> servad
    - laius = median(right) - median(left)
    - tagastame mediaani üle slice'ide
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, 0

    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 200:
        return None, 0

    XY = pts[:, :2]

    # PCA main direction
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
        return None, 0

    thick = max(length * slice_thickness_ratio, 0.5)
    centers = np.linspace(s_min, s_max, n_bins)

    widths = []
    valid = 0

    for c in centers:
        mask = np.abs(s - c) <= thick
        if mask.sum() < 60:
            continue

        tt = t[mask]
        left_thr = np.percentile(tt, edge_tail_pct)
        right_thr = np.percentile(tt, 100.0 - edge_tail_pct)

        left = tt <= left_thr
        right = tt >= right_thr
        if left.sum() < 8 or right.sum() < 8:
            continue

        tL = float(np.median(tt[left]))
        tR = float(np.median(tt[right]))
        w = float(tR - tL)
        if w > 0:
            widths.append(w)
            valid += 1

    if not widths:
        return None, 0

    return float(np.median(widths)), int(valid)
