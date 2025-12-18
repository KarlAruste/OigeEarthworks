import xml.etree.ElementTree as ET
import numpy as np


def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def extract_tin_points_xyz(xml_bytes: bytes):
    """
    Extracts (N,3) points from LandXML.
    Attempts to detect and fix swapped XY (Northing/Easting).
    Returns numpy array or None.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None

    pts = []

    # Most common: <P> x y z </P> (sometimes y x z)
    for el in root.iter():
        if _strip(el.tag).lower() == "p":
            txt = (el.text or "").strip()
            if not txt:
                continue
            parts = txt.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                a, b, z = float(parts[0]), float(parts[1]), float(parts[2])
                pts.append([a, b, z])
            except Exception:
                pass

    if not pts:
        return None

    P = np.asarray(pts, dtype=float)

    # Heuristic swap:
    # If median X is ~6-7 million and median Y is ~0.4-1.2 million,
    # it's likely swapped (X=N, Y=E). We want X~E, Y~N.
    x_med = float(np.median(P[:, 0]))
    y_med = float(np.median(P[:, 1]))
    if x_med > 2_000_000 and y_med < 2_000_000:
        P = P[:, [1, 0, 2]]

    return P


def _pca_axis_xy(XY: np.ndarray) -> np.ndarray:
    """Unit principal axis direction via SVD."""
    C = XY - XY.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(C, full_matrices=False)
    axis = vh[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis


def _poly_area_2d(xy: np.ndarray) -> float:
    """Shoelace area for polygon points (N,2)."""
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_cross_sections_every_5m(
    xml_bytes: bytes,
    step_m: float = 5.0,
    slice_thickness_m: float = 1.0,
    edge_band_pct: float = 12.0,
    edge_z_pct: float = 92.0,
    bottom_z_pct: float = 3.0,
    min_points_in_slice: int = 60,
):
    """
    Computes cross-section geometry from a TIN where:
      - edges are existing ground (top-of-slope)
      - bottom is the ditch invert

    For each station every `step_m` along the principal axis:
      - take points with |s - s0| <= slice_thickness_m/2
      - bottom_z = percentile(bottom_z_pct) of z in slice
      - left edge band = lowest edge_band_pct% of t; left_edge_z = percentile(edge_z_pct) of z within band
      - right edge band = highest edge_band_pct% of t; right_edge_z = percentile(edge_z_pct) of z within band
      - top width = mean(t_right_band) - mean(t_left_band)
      - area = polygon area in (t,z) plane using [left_edge, right_edge, bottom]

    Returns dict or None:
      {
        "axis_length_m": float,
        "sections": [
            {"station_m":..,"area_m2":..,"top_width_m":..,"bottom_z":..,
             "left_edge_z":..,"right_edge_z":..}
        ],
        "area_min_m2":..,"area_avg_m2":..,"area_max_m2":..,
        "top_width_avg_m":..
      }
    """
    P = extract_tin_points_xyz(xml_bytes)
    if P is None or len(P) < 200:
        return None

    XY = P[:, :2]
    Z = P[:, 2]

    axis = _pca_axis_xy(XY)
    perp = np.array([-axis[1], axis[0]])

    centered = XY - XY.mean(axis=0, keepdims=True)
    s = centered @ axis     # along
    t = centered @ perp     # cross

    s_min, s_max = float(np.min(s)), float(np.max(s))
    axis_len = s_max - s_min
    if axis_len <= 1e-6:
        return None

    n = int(np.floor(axis_len / step_m)) + 1
    stations_s = s_min + np.arange(n) * step_m

    half_th = slice_thickness_m / 2.0
    sections = []

    for s0 in stations_s:
        mask = np.abs(s - s0) <= half_th
        idx = np.where(mask)[0]
        if idx.size < min_points_in_slice:
            continue

        tt = t[idx]
        zz = Z[idx]

        # bottom: robust low percentile (not absolute min)
        bottom_z = float(np.percentile(zz, bottom_z_pct))

        # edge bands by t extremes
        order = np.argsort(tt)
        k = max(8, int(np.ceil(idx.size * (edge_band_pct / 100.0))))

        left_ids = order[:k]
        right_ids = order[-k:]

        left_t = float(np.mean(tt[left_ids]))
        right_t = float(np.mean(tt[right_ids]))

        # edge z: robust high percentile within edge band
        left_edge_z = float(np.percentile(zz[left_ids], edge_z_pct))
        right_edge_z = float(np.percentile(zz[right_ids], edge_z_pct))

        top_width = float(abs(right_t - left_t))

        # polygon in (t,z)
        poly = np.array(
            [
                [left_t, left_edge_z],
                [right_t, right_edge_z],
                [right_t, bottom_z],
                [left_t, bottom_z],
            ],
            dtype=float,
        )

        area = _poly_area_2d(poly)
        if not np.isfinite(area) or area <= 0:
            continue

        sections.append(
            {
                "station_m": float(s0 - s_min),
                "area_m2": float(area),
                "top_width_m": float(top_width),
                "bottom_z": float(bottom_z),
                "left_edge_z": float(left_edge_z),
                "right_edge_z": float(right_edge_z),
            }
        )

    if not sections:
        return None

    areas = np.array([x["area_m2"] for x in sections], dtype=float)
    widths = np.array([x["top_width_m"] for x in sections], dtype=float)

    return {
        "axis_length_m": float(axis_len),
        "sections": sections,
        "area_min_m2": float(np.min(areas)),
        "area_avg_m2": float(np.mean(areas)),
        "area_max_m2": float(np.max(areas)),
        "top_width_avg_m": float(np.mean(widths)),
    }


def estimate_volume_from_sections(result: dict, user_length_m: float | None = None, step_m: float = 5.0):
    """
    If user_length_m provided:
      volume = avg_area * user_length_m
    Else:
      volume ≈ sum(area_i) * step_m  (Riemann sum)
    """
    if not result or not result.get("sections"):
        return None

    areas = np.array([s["area_m2"] for s in result["sections"]], dtype=float)
    avg_area = float(np.mean(areas))

    if user_length_m is not None and user_length_m > 0:
        return {
            "min_area_m2": float(np.min(areas)),
            "avg_area_m2": avg_area,
            "max_area_m2": float(np.max(areas)),
            "volume_m3": float(avg_area * float(user_length_m)),
        }

    return {
        "min_area_m2": float(np.min(areas)),
        "avg_area_m2": avg_area,
        "max_area_m2": float(np.max(areas)),
        "volume_m3": float(np.sum(areas) * float(step_m)),
    }
