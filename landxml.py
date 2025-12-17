import xml.etree.ElementTree as ET

def parse_landxml_total_volume(xml_bytes: bytes):
    """
    Tries to extract a reasonable total volume (m3) from LandXML.
    Returns float or None.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None

    def strip(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    candidates = []
    for el in root.iter():
        name = strip(el.tag).lower()

        if name in ("volume", "totalvolume", "earthworkvolume", "cutvolume", "fillvolume"):
            txt = (el.text or "").strip()
            try:
                candidates.append(float(txt))
            except Exception:
                pass

        for k, v in el.attrib.items():
            kk = k.lower()
            if "volume" in kk:
                try:
                    candidates.append(float(v))
                except Exception:
                    pass

    positives = [c for c in candidates if c and c > 0]
    return max(positives) if positives else None

