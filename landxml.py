# landxml.py
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import numpy as np


def _strip(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _as_floats(text: str):
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def read_landxml_tin_from_bytes(xml_bytes: bytes):
    """
    Toetab:
      A) <Definition><Pnts><P id=".."> ... </P> ja <Faces><F>...</F>
      B) (fallback) <PntList3D> ... </PntList3D> (ainult punktid, faces puuduvad)

    Tagastab:
      pts: dict[int] -> (E, N, Z)
      faces: list[(a,b,c)]  (võib olla tühi kui faces ei leita)
    """
    root = ET.fromstring(xml_bytes)

    pts = {}
    faces = []

    # ---- 1) Leia <P id=".."> ----
    for el in root.iter():
        if _strip(el.tag).lower() == "p":
            pid = el.attrib.get("id")
            txt = (el.text or "").strip()
            if not pid or not txt:
                continue
            arr = _as_floats(txt)
            if len(arr) < 3:
                continue
            a, b, z = float(arr[0]), float(arr[1]), float(arr[2])
            # LandXML-s tihti (N, E, Z). Me tahame (E, N, Z).
            # Heuristika: kui esimene ~miljonid ja teine ~sajad tuhanded → esimene on N.
            if a > 2_000_000 and b < 2_000_000:
                e, n = b, a
            else:
                e, n = a, b
            pts[int(pid)] = (float(e), float(n), float(z))

    # ---- 2) Leia <F> faces ----
    for el in root.iter():
        if _strip(el.tag).lower() == "f":
            txt = (el.text or "").strip()
            if not txt:
                continue
            arr = txt.replace(",", " ").split()
            if len(arr) >= 3:
                try:
                    a, b, c = int(arr[0]), int(arr[1]), int(arr[2])
                    faces.append((a, b, c))
                except Exception:
                    pass

    # ---- 3) Fallback: <PntList3D> (kui P id ei leitud) ----
    if not pts:
        for el in root.iter():
            if _strip(el.tag).lower() == "pntlist3d":
                txt = (el.text or "").strip()
                if not txt:
                    continue
                arr = _as_floats(txt)
                # eeldame N E Z tripletid
                if len(arr) >= 3:
                    pid = 1
                    for i in range(0, len(arr) - 2, 3):
                        a, b, z = float(arr[i]), float(arr[i + 1]), float(arr[i + 2])
                        if a > 2_000_000 and b < 2_000_000:
                            e, n = b, a
                        else:
                            e, n = a, b
                        pts[pid] = (float(e), float(n), float(z))
                        pid += 1
                break

    return pts, faces


# -----------------------------
# Util: polyline length
# -----------------------------
def polyline_length(xy):
    if not xy or len(xy) < 2:
        return 0.0
    L = 0.0
    for i in range(len(xy) - 1):
        dx = xy[i + 1][0] - xy[i][0]
        dy = xy[i + 1][1] - xy[i][1]
        L += math.hypot(dx, dy)
    return float(L)


# =============================
# (Sinu olemasolevad arvutused)
# =============================
# Siin eeldan, et sul on juba compute_pk_table_from_landxml(...) valmis
# ja see töötab ABS koordinaatidega (E,N).
# ÄRA muuda selle sisemist loogikat, ainult kasuta axis_xy_abs.

def compute_pk_table_from_landxml(
    xml_bytes: bytes,
    axis_xy,
    pk_step: float,
    cross_len: float,
    sample_step: float,
    tol: float,
    min_run: float,
    min_depth_from_bottom: float,
    slope_text: str,
    bottom_w: float,
):
    """
    SIIN PEAB OLEMA SUL SEE FUNKTSIOON, MIS JUBA TÖÖTAB.
    Kui sul see juba on, jäta see samaks ja ära dubleeri.
    """
    raise NotImplementedError("Kasuta oma olemasolevat compute_pk_table_from_landxml implementatsiooni.")
