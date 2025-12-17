import xml.etree.ElementTree as ET
import numpy as np

NS = {"lx": "http://www.landxml.org/schema/LandXML-1.2"}

def _read_points_xyz(root):
    # LandXML: <Pnts><P id="...">x y z</P>...
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

def estimate_length_area_volume_from_tin(xml_bytes: bytes,
                                        n_bins: int = 25,
                                        slice_thickness_ratio: float = 0.02,
                                        top_percentile: float = 95.0):
    """
    Heuristic:
    - Find main direction of surface in XY using PCA -> axis 's'
    - For many bins along 's', take points near bin center (thin slice)
    - In each slice, compute "excavation" area vs top plane (z_top = percentile)
      area ≈ integral over t of max(0, z_top - z(t))
    Returns: (length_m, mean_area_m2, volume_m3) or (None, None, None)
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, None, None

    pts = _read_points_xyz(root)
    if pts is None or len(pts) < 200:
        return None, None, None

    XY = pts[:, :2]
    Z  = pts[:, 2]

    # PCA on XY to get main axis
    XYc = XY - XY.mean(axis=0)
    cov = np.cov(XYc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]  # 2-vector

    # project to (s,t) coordinates
    s = XYc @ main_dir
    # perpendicular direction:
    perp = np.array([-main_dir[1], main_dir[0]])
    t = XYc @ perp

    s_min, s_max = float(s.min()), float(s.max())
    length = s_max - s_min
    if length <= 0:
        return None, None, None

    # slice thickness based on overall length
    thick = max(length * slice_thickness_ratio, 0.25)

    # bin centers
    centers = np.linspace(s_min, s_max, n_bins)

    areas = []
    for c in centers:
        mask = np.abs(s - c) <= thick
        if mask.sum() < 50:
            continue

        tt = t[mask]
        zz = Z[mask]

        # top reference (assume near edges/ground)
        z_top = np.percentile(zz, top_percentile)

        # Sort by t and integrate positive depth
        order = np.argsort(tt)
        tt_s = tt[order]
        depth = np.maximum(0.0, z_top - zz[order])

        # If too narrow / degenerate, skip
        if (tt_s.max() - tt_s.min()) < 0.5:
            continue

        area = np.trapz(depth, tt_s)  # m²
        if area > 0:
            areas.append(float(area))

    if not areas:
        return float(length), None, None

    mean_area = float(np.mean(areas))
    volume = float(length * mean_area)
    return float(length), mean_area, volume

