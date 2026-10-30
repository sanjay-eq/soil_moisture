"""
Bhuvan-Bhujal ground water prospects helpers (fast version).

Speed-ups over the original:
  - one reusable HTTPS connection (requests.Session)
  - GetCapabilities downloaded once per session (cached)
  - geometry field for the polygon filter remembered per layer
  - area legend fetched directly with the filter (no separate test request)
  - sample points queried in parallel, no sleep
  - 30 s timeout instead of 90 s
"""

import functools
import re
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# --- Settings ----------------------------------------------------------------
WMS_URL = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
CAPS_URLS = ["https://bhuvan-vec1.nrsc.gov.in/bhuvan/gw/wms", WMS_URL]
FALLBACK_STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "CT", "GA", "GJ", "HR", "HP", "JH", "JK", "KA",
    "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OR", "OD", "PB", "RJ", "SK", "TN",
    "TS", "TG", "TR", "UP", "UK", "UT", "WB", "DL", "LD", "PY", "AN", "CH", "DN",
]
DEFAULT_LAYER = "auto"
DEFAULT_STYLE = "GWLITHGEOM"
GEOM_FIELDS = ["the_geom", "geom", "wkb_geometry", "shape"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research script)",
    "Referer": "https://bhuvan-app1.nrsc.gov.in/gwis/gwis.php",
}
TIMEOUT = 30
WORKERS = 4                 # parallel requests; keep low to avoid being throttled
LEGEND_SIZE = (512, 512)    # only scopes the legend; nothing is drawn
BOLD, END = "\033[1m", "\033[0m"

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_GEOM_FIELD = {}            # layer -> geometry field that the filter accepted


# --- Geometry helpers --------------------------------------------------------
def _polygons_only(g):
    """Flatten any geometry (incl. GeometryCollection) to a list of Polygons."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in _polygons_only(sub)]
    return []


def _read_kml_polygons(path):
    """Fallback reader: all polygons in a KML/KMZ, across every folder."""
    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
            data = z.read(name)
    else:
        with open(path, "rb") as f:
            data = f.read()
    root = ET.fromstring(data)

    def coords(elem):
        text = elem.text.strip() if elem is not None and elem.text else ""
        pts = []
        for tok in text.split():
            parts = tok.split(",")
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        return pts

    polys = []
    for poly in root.iter():
        if not poly.tag.endswith("Polygon"):
            continue
        outer, holes = None, []
        for child in poly.iter():
            if child.tag.endswith("outerBoundaryIs"):
                c = next((e for e in child.iter() if e.tag.endswith("coordinates")), None)
                outer = coords(c)
            elif child.tag.endswith("innerBoundaryIs"):
                c = next((e for e in child.iter() if e.tag.endswith("coordinates")), None)
                holes.append(coords(c))
        if outer and len(outer) >= 3:
            polys.append(Polygon(outer, [h for h in holes if len(h) >= 3]))
    if not polys:
        raise ValueError("No polygons found in the KML (points/lines are not supported).")
    return unary_union([p.buffer(0) for p in polys])


def _to_wkt_for_url(geom, max_vertices=150):
    """Simplify the geometry so the WKT fits in a URL."""
    tol, g = 0.0, geom
    while True:
        n = sum(len(p.exterior.coords) for p in getattr(g, "geoms", [g]))
        if n <= max_vertices:
            break
        tol = tol * 2 if tol else 0.0005
        g = geom.simplify(tol, preserve_topology=True)
    return g.wkt.replace(", ", ",")


def _padded_bbox(geom, pad_frac=0.05):
    minx, miny, maxx, maxy = geom.bounds
    px, py = (maxx - minx) * pad_frac, (maxy - miny) * pad_frac
    return (minx - px, miny - py, maxx + px, maxy + py)


# --- WMS requests ------------------------------------------------------------
def _wms_get(params):
    r = SESSION.get(WMS_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def _clean_title(title):
    """Strip the '(n)' feature count that countMatched appends."""
    return re.sub(r"\s*\(\d+\)\s*$", "", title or "").strip()


def _legend_titles(layer, style, bbox, cql=None):
    """Legend class titles (JSON GetLegendGraphic). Raises if the server errors."""
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetLegendGraphic",
        "LAYER": layer, "STYLE": style, "FORMAT": "application/json",
        "BBOX": ",".join(map(str, bbox)), "SRS": "EPSG:4326",
        "WIDTH": LEGEND_SIZE[0], "HEIGHT": LEGEND_SIZE[1],
        "LEGEND_OPTIONS": "hideEmptyRules:true;countMatched:true",
    }
    if cql:
        params["CQL_FILTER"] = cql
    data = _wms_get(params).json()          # an XML error page fails here
    titles = [_clean_title(rule.get("title") or "")
              for lyr in data.get("Legend", []) for rule in lyr.get("rules", [])]
    return [t for t in titles if t]


def _area_legend(layer, style, geom, bbox):
    """
    Legend classes inside the polygon, in as few requests as possible.
    Returns (titles, cql); cql is None if only the bounding box could be used.
    """
    wkt = _to_wkt_for_url(geom)
    fields = GEOM_FIELDS
    if layer in _GEOM_FIELD:                # known field: try it first
        fields = [_GEOM_FIELD[layer]] + [f for f in GEOM_FIELDS if f != _GEOM_FIELD[layer]]
    for field in fields:
        cql = f"INTERSECTS({field},SRID=4326;{wkt})"
        try:
            titles = _legend_titles(layer, style, bbox, cql)
            _GEOM_FIELD[layer] = field
            return titles, cql
        except Exception:
            continue
    try:
        return _legend_titles(layer, style, bbox), None
    except Exception as e:
        print(f"  Legend request failed: {e}")
        return [], None


# --- Parsing legend text -----------------------------------------------------
def _ci_get(props, name):
    return next((v for k, v in props.items() if k.lower() == name.lower()), None)


def _depth_from_title(title):
    """'30 to 80 m Deep well - ...' -> '30 to 80 m';  '> 80 m Deep well' -> '> 80 m'."""
    if not title:
        return None
    m = re.match(r"^(.*?\d+\s*m)\b\s*deep", title, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"([<>]?\s*\d+(?:\s*(?:to|-)\s*\d+)?\s*m)\b", title)
    return m.group(1).strip() if m else None


def _yield_from_title(title):
    """'... - 30 to 50 LPM Yield' -> '30 to 50 LPM';  '> 400 LPM' -> '> 400 LPM'."""
    if not title:
        return None
    m = re.search(r"([<>]?\s*\d+(?:\s*(?:to|-)\s*\d+)?\s*LPM)", title, re.I)
    return m.group(1).strip() if m else None


def _find_depth(props, depth_field=None):
    if depth_field:
        return _ci_get(props, depth_field)
    for k, v in props.items():
        if "depth" in k.lower() and v not in (None, ""):
            return v
    return None


def _legend_class_for_unit(layer, style, bbox, code_field, code):
    code_sql = str(code).replace("'", "''")
    try:
        titles = _legend_titles(layer, style, bbox, f"{code_field}='{code_sql}'")
        with_depth = [t for t in titles if _depth_from_title(t)]
        return (with_depth or titles or [None])[0]
    except Exception as e:
        print(f"  legend lookup for {code} failed: {e}")
        return None


# --- Point sampling ----------------------------------------------------------
def _feature_at(layer, x, y):
    d = 0.0005
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetFeatureInfo",
        "LAYERS": layer, "QUERY_LAYERS": layer, "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": f"{x - d},{y - d},{x + d},{y + d}",
        "WIDTH": 11, "HEIGHT": 11, "X": 5, "Y": 5,
        "INFO_FORMAT": "application/json", "FEATURE_COUNT": 1,
    }
    feats = _wms_get(params).json().get("features", [])
    return feats[0]["properties"] if feats else None


def _sample_points(geom, n):
    minx, miny, maxx, maxy = geom.bounds
    k = int(np.ceil(np.sqrt(n * 2)))
    pts = [Point(x, y) for x in np.linspace(minx, maxx, k)
           for y in np.linspace(miny, maxy, k) if geom.contains(Point(x, y))][:n]
    return pts or [geom.representative_point()]


def _sample_units(layer, geom, n=4, depth_field=None):
    """Query n points inside the polygon in parallel; group them by unit."""
    pts = _sample_points(geom, n)

    def query(p):
        try:
            return _feature_at(layer, p.x, p.y)
        except Exception as e:
            print(f"  point {p.y:.4f},{p.x:.4f}: {e}")
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(query, pts))

    units, fields = {}, set()
    for a in results:
        if not a:
            continue
        fields.update(a.keys())
        code_field = next((k for k in a if k.lower() == "alnum_code"), None)
        key = a[code_field] if code_field else str(a)
        units.setdefault(key, {"code_field": code_field,
                               "lithology": _ci_get(a, "lg_1"),
                               "landform": _ci_get(a, "lg_2"),
                               "depth": _find_depth(a, depth_field),
                               "class": None, "hits": 0})
        units[key]["hits"] += 1
    return units, fields, len(pts)


def _classify_units(units, layer, style, bbox):
    """Legend class for every unit that has no depth yet, in parallel."""
    todo = [(c, u) for c, u in units.items() if u["depth"] is None and u["code_field"]]

    def lookup(item):
        code, u = item
        return _legend_class_for_unit(layer, style, bbox, u["code_field"], code)

    with ThreadPoolExecutor(WORKERS) as ex:
        for (code, u), cls in zip(todo, ex.map(lookup, todo)):
            u["class"] = cls
            u["depth"] = _depth_from_title(cls)


# --- State layer detection ---------------------------------------------------
@functools.lru_cache(maxsize=1)
def _list_gw_layers():
    """All *_LGEOM layers with extent and styles (downloaded once per session)."""
    for url in CAPS_URLS:
        try:
            r = SESSION.get(url, params={"SERVICE": "WMS", "VERSION": "1.1.1",
                                         "REQUEST": "GetCapabilities"},
                            timeout=TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"  GetCapabilities from {url} failed: {e}")
            continue
        layers = []
        for lyr in root.iter():
            if not lyr.tag.endswith("Layer"):
                continue
            name = next((c.text for c in lyr if c.tag.endswith("Name")), None)
            if not name or not name.upper().endswith("_LGEOM"):
                continue
            if ":" not in name:
                name = "gw:" + name
            box = None
            for c in lyr:
                if c.tag.endswith("LatLonBoundingBox"):
                    box = tuple(float(c.get(k)) for k in ("minx", "miny", "maxx", "maxy"))
            styles = [n.text for st in lyr if st.tag.endswith("Style")
                      for n in st if n.tag.endswith("Name")]
            layers.append({"name": name, "bbox": box, "styles": styles})
        if layers:
            return tuple(layers)
    return tuple({"name": f"gw:{c}_LGEOM", "bbox": None, "styles": []}
                 for c in FALLBACK_STATE_CODES)


def _detect_layer(geom):
    """State *_LGEOM layer with data inside the polygon (candidates checked in parallel)."""
    layers = _list_gw_layers()
    p = geom.representative_point()
    cands = [l for l in layers if l["bbox"] is None or
             (l["bbox"][0] <= p.x <= l["bbox"][2] and l["bbox"][1] <= p.y <= l["bbox"][3])]
    cands.sort(key=lambda l: (l["bbox"][2] - l["bbox"][0]) * (l["bbox"][3] - l["bbox"][1])
               if l["bbox"] else 1e9)

    def has_data(l):
        try:
            return bool(_feature_at(l["name"], p.x, p.y))
        except Exception:
            return False

    with ThreadPoolExecutor(WORKERS) as ex:
        for l, ok in zip(cands, ex.map(has_data, cands)):
            if ok:              # first hit in smallest-extent order
                return l
    raise RuntimeError("No Bhuvan ground-water (*_LGEOM) layer has data for this area. "
                       "Pass layer='gw:<STATE>_LGEOM' explicitly.")
