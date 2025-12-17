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
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            pts.append((x, y, z))
        except:
            pass
    return np.array(pts, dtype=float) if pts else None


def estimate_length_area_volume_from_tin(
    xml_bytes: bytes,
    n_bins: int = 30,
    slice_thickness_ratio: float = 0.03,
    edge_tail_pct: float = 10.0,      # "serva" jaoks kui suur tail (10% vasakult + 10% paremalt)
    trim_inner_pct: float = 1.0,      # lõika äärmised outlier t-d ära
    min_points_per_slice: int = 120,
):
    """
    TIN-ist kraavi ristlõike hindamine:
    - PCA -> põhisuund s ja ristsuund t
    - igas slice'is leiame VASAKU ja PAREMA serva (t tail-id)
    - top line = sirge (vasak serv -> parem serv)
    - ristlõike pindala = ∫ max(0, top_line(t) - z(t)) dt
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, None, None

    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 500:
        return None, None, None

    XY = pts[:, :2]
    Z = pts[:, 2]

    # PCA to find main axis
    XYc = XY - XY.mean(axis=0)
    cov = np.cov(XYc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]  # 2-vector

    s = XYc @ main_dir
    perp = np.array([-main_dir[1], main_dir[0]])
    t = XYc @ perp

    s_min, s_max = float(s.min()), float(s.max())
    length = s_max - s_min
    if length <= 0:
        return None, None, None

    thick = max(length * slice_thickness_ratio, 0.5)
    centers = np.linspace(s_min, s_max, n_bins)

    areas = []
    for c in centers:
        mask = np.abs(s - c) <= thick
        if mask.sum() < min_points_per_slice:
            continue

        tt = t[mask]
        zz = Z[mask]

        # trim extreme t outliers to make edges stable
        t_lo = np.percentile(tt, trim_inner_pct)
        t_hi = np.percentile(tt, 100.0 - trim_inner_pct)
        m2 = (tt >= t_lo) & (tt <= t_hi)
        tt = tt[m2]
        zz = zz[m2]
        if len(tt) < min_points_per_slice:
            continue

        # define left/right edge sets by tails of t distribution
        left_thr = np.percentile(tt, edge_tail_pct)
        right_thr = np.percentile(tt, 100.0 - edge_tail_pct)

        left = (tt <= left_thr)
        right = (tt >= right_thr)

        if left.sum() < 10 or right.sum() < 10:
            continue

        # edge elevations as robust median
        tL = float(np.median(tt[left]))
        zL = float(np.median(zz[left]))
        tR = float(np.median(tt[right]))
        zR = float(np.median(zz[right]))

        if abs(tR - tL) < 0.5:
            continue

        # sort by t
        order = np.argsort(tt)
        tt_s = tt[order]
        zz_s = zz[order]

        # top line at each tt: linear interpolation between (tL,zL) and (tR,zR)
        top_line = zL + (zR - zL) * ((tt_s - tL) / (tR - tL))

        depth = np.maximum(0.0, top_line - zz_s)

        width = float(tt_s.max() - tt_s.min())
        if width < 0.8:
            continue

        area = float(np.trapz(depth, tt_s))
        if area > 0:
            areas.append(area)

    if not areas:
        # at least we can return length
        return float(length), None, None

    mean_area = float(np.mean(areas))
    volume = float(length * mean_area)
    return float(length), mean_area, volume
