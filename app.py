"""
Land & Water Explorer — Streamlit app
Upload a KML boundary and explore ERA5-Land soil moisture, Bhuvan/Bhujal
groundwater prospects, CGWB groundwater resource figures and SoilGrids soil type.

Run:  streamlit run app.py
"""

import os

# PostgreSQL/PostGIS sets PROJ_LIB / PROJ_DATA to its own (older) proj.db, which
# breaks pyproj and rasterio. Remove them so each library uses its bundled PROJ data.
for _var in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_var, None)

import io
import json
import hashlib
import tempfile
import contextlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import streamlit as st

import ee
import geemap
import rasterio
import rasterio.windows
from rasterio.mask import mask

try:  # optional: only used to read soil class names; falls back to the WRB list
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    gdal = None
from shapely import force_2d
from shapely.validation import make_valid
from shapely.ops import unary_union

import bhujal  # your local helper module (bhujal.py next to app.py)

# Let fiona-backed geopandas read KML (pyogrio reads it natively)
try:
    import fiona
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
except Exception:
    pass

st.set_page_config(page_title="Land & Water Explorer", page_icon="💧", layout="wide")

# Fixed local groundwater resources layer (CGWB assessment units).
# Change this to the actual file on your machine (.shp, .geojson, .gpkg, .kml or a .zip).
GW_DATA_PATH = r"C:\Users\Lenovo\EQ\Sanjay\soil moisture\gw_data.shp"

# Your Google Cloud project registered for Earth Engine (e.g. "ee-sanjay").
# Find it at https://console.cloud.google.com (project picker at the top).
EE_PROJECT = "eq-project-511406"

APP_VERSION = "2026-11-03b"  # shown in the sidebar so you can confirm which code is deployed

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "land_water_explorer")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def read_vector(path):
    """Read KML / GeoJSON / GPKG / zipped shapefile."""
    if path.lower().endswith(".zip"):
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def save_upload(uploaded):
    """Write an uploaded file to disk with a content hash so the path is stable across reruns."""
    data = uploaded.getvalue()
    digest = hashlib.md5(data).hexdigest()[:12]
    path = os.path.join(UPLOAD_DIR, f"{digest}_{uploaded.name}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


@contextlib.contextmanager
def capture_output():
    """Capture anything printed (e.g. by bhujal) so it can be shown in the UI."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


SA_SECTION_NAMES = ("gee_service_account", "gcp_service_account", "earthengine", "gee",
                    "service_account")


def _load_service_account():
    """Find service-account credentials in st.secrets.

    Accepts a [gee_service_account] TOML section (or a similar name), the JSON key pasted
    as a string under one of those names, or the key's fields pasted at the top level.
    Returns (info_dict or None, list of top-level secret names found).
    """
    try:
        names = list(st.secrets.keys())
    except Exception:  # no secrets file at all (e.g. running locally)
        return None, []

    info = None
    for name in SA_SECTION_NAMES:
        if name in st.secrets:
            raw = st.secrets[name]
            info = json.loads(raw) if isinstance(raw, str) else dict(raw)
            break

    if info is None and "client_email" in st.secrets and "private_key" in st.secrets:
        info = {k: st.secrets[k] for k in names if isinstance(st.secrets[k], (str, int, float))}

    if info:
        # Keys pasted with literal "\\n" sequences need real newlines
        info["private_key"] = str(info.get("private_key", "")).replace("\\n", "\n")
    return info, names


@st.cache_resource(show_spinner="Connecting to Google Earth Engine…")
def init_ee(project):
    """Initialise Earth Engine with a service account from st.secrets, or local credentials."""
    sa, secret_names = _load_service_account()

    if sa:
        missing = [k for k in ("client_email", "private_key") if not sa.get(k)]
        if missing:
            raise RuntimeError(f"Service account secret is missing: {', '.join(missing)}.")
        creds = ee.ServiceAccountCredentials(sa["client_email"], key_data=json.dumps(sa))
        ee.Initialize(creds, project=project or sa.get("project_id"))
        return f"service account {sa['client_email']}"

    try:
        ee.Initialize(project=project or None)
        return "local credentials"
    except Exception as e:
        found = ", ".join(secret_names) if secret_names else "none"
        raise RuntimeError(
            "No Earth Engine service account found in Streamlit secrets "
            f"(top-level secret names found: {found}). Add a [gee_service_account] "
            "section in the app's Settings → Secrets, or run `earthengine authenticate` "
            f"when running locally. Original error: {e}"
        ) from e


# ----------------------------------------------------------------------------
# Core analysis class (same logic as the original, returning data instead of printing)
# ----------------------------------------------------------------------------
class SoilMoisture:

    def __init__(self, kml_path, gw_path, start_date, end_date):
        self.kml_path = kml_path
        self.gdf = read_vector(kml_path)
        self.gw_data = read_vector(gw_path) if gw_path else None
        self.start_date = start_date
        self.end_date = end_date
        self._gw_cache = {}

    # ---------- ERA5-Land soil moisture ----------
    def era5_soil_moisture(self):
        gdf = self.gdf[self.gdf.geometry.notna() & ~self.gdf.geometry.is_empty]
        gdf = gdf.to_crs(epsg=4326)
        gdf["geometry"] = gdf.geometry.apply(make_valid).apply(force_2d)
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon", "Point", "LineString"])]

        gdf_ee = geemap.gdf_to_ee(gdf)

        dataset_id = "ECMWF/ERA5_LAND/DAILY_AGGR"
        bands = [
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
        ]
        rename_bands = [
            "soil_water_0_7cm",
            "soil_water_7_28cm",
            "soil_water_28_100cm",
            "soil_water_100_289cm",
        ]

        data = (
            ee.ImageCollection(dataset_id)
            .filterDate(self.start_date, self.end_date)
            .filterBounds(gdf_ee)
            .select(bands, rename_bands)
        )
        return data, gdf_ee

    def sm_data_extraction(self):
        soil_data, region = self.era5_soil_moisture()

        def extract_mean(img):
            values = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region.geometry(),
                scale=1000,
                maxPixels=1e13,
            )
            return ee.Feature(None, values).set("date", img.date().format("YYYY-MM-dd"))

        extracted = ee.FeatureCollection(soil_data.map(extract_mean))
        return geemap.ee_to_df(extracted)

    # ---------- AOI map ----------
    def visualize(self, zoom=17):
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf = self.gdf.to_crs(4326) if self.gdf.crs else self.gdf.set_crs(4326)
        gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

        minx, miny, maxx, maxy = gdf.total_bounds
        pad = 0.001  # ~100 m
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.Esri.WorldImagery, zoom=zoom)
        ax.ticklabel_format(useOffset=False)
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        return fig

    # ---------- Groundwater prospects (Bhujal) ----------
    def _aoi_geometry(self):
        gdf = self.gdf
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
        polys = [
            p
            for g in gdf.geometry
            if g is not None and not g.is_empty
            for p in bhujal._polygons_only(make_valid(force_2d(g)))
        ]
        if polys:
            return unary_union(polys)
        return bhujal._read_kml_polygons(self.kml_path)

    def groundwater_prospects(self, layer=bhujal.DEFAULT_LAYER, style=bhujal.DEFAULT_STYLE,
                              samples=4, depth_field=None):
        key = (layer, style, samples, depth_field)
        if key in self._gw_cache:
            return self._gw_cache[key]

        geom = self._aoi_geometry()
        bbox = bhujal._padded_bbox(geom)

        if layer.lower() == "auto":
            info = bhujal._detect_layer(geom)
            layer = info["name"]
            if info["styles"] and style not in info["styles"]:
                style = info["styles"][0]

        prospects, cql = bhujal._area_legend(layer, style, geom, bbox)

        units = {}
        if samples > 0:
            units, _, _ = bhujal._sample_units(layer, geom, samples, depth_field)
            bhujal._classify_units(units, layer, style, bbox)
        ordered = sorted(units.items(), key=lambda kv: -kv[1]["hits"])

        depths = list(dict.fromkeys(u["depth"] for _, u in ordered if u["depth"]))
        if not depths:
            depths = [d for d in dict.fromkeys(bhujal._depth_from_title(t) for t in prospects) if d]
        yields = list(dict.fromkeys(
            y for y in (bhujal._yield_from_title(u["class"]) for _, u in ordered) if y))
        if not yields:
            yields = [y for y in dict.fromkeys(bhujal._yield_from_title(t) for t in prospects) if y]

        res = {
            "layer": layer,
            "scope": "this area" if cql else "the area's bounding box",
            "prospects": prospects,
            "units": units,
            "depths": depths,
            "depth_text": ("Well depth: " + ", ".join(depths)) if depths
                          else "Well depth: not found in this layer",
            "yields": yields,
            "yield_text": ("Yield: " + ", ".join(yields)) if yields
                          else "Yield: not found in this layer",
        }
        self._gw_cache[key] = res
        return res

    # ---------- CGWB groundwater resource assessment ----------
    def comprehensive_gw(self):
        if self.gw_data is None:
            raise ValueError(f"Groundwater layer not found at {GW_DATA_PATH}.")

        aoi = self.gdf.to_crs(self.gw_data.crs) if self.gdf.crs != self.gw_data.crs else self.gdf
        hits = gpd.sjoin(self.gw_data, aoi, how="inner", predicate="intersects")

        req_cols = [
            "STATE", "Name of District",
            "Recharge from rainfall - Monsoon (ham)",
            "Recharge from other sources - Monsoon (ham)",
            "Recharge from rainfall - Non-monsoon (ham)",
            "Recharge from other sources - Non-monsoon (ham)",
            "Total Annual Ground Water Recharge (ham)",
            "Total Natural Discharges (ham)",
            "Annual Extractable Ground Water Resource (ham)",
            "Extraction - Irrigation (ham)", "Extraction - Industrial (ham)",
            "Extraction - Domestic (ham)", "Extraction - Total (ham)",
            "Annual GW Allocation for Domestic Use as on 2035 (ham)",
            "Net Ground Water Availability for future use (ham)",
            "Stage of Ground Water Extraction (%)",
        ]
        present = [c for c in req_cols if c in hits.columns]
        missing = [c for c in req_cols if c not in hits.columns]
        out = pd.DataFrame(hits[present]).drop_duplicates().reset_index(drop=True)
        return out, missing

    # ---------- SoilGrids WRB soil type ----------
    def get_soil_type(self):
        url = "/vsicurl/https://files.isric.org/soilgrids/latest/data/wrb/MostProbable.vrt"

        WRB = ["Acrisols", "Albeluvisols", "Alisols", "Andosols", "Arenosols", "Calcisols",
               "Cambisols", "Chernozems", "Cryosols", "Durisols", "Ferralsols", "Fluvisols",
               "Gleysols", "Gypsisols", "Histosols", "Kastanozems", "Leptosols", "Lixisols",
               "Luvisols", "Nitisols", "Phaeozems", "Planosols", "Plinthosols", "Podzols",
               "Regosols", "Solonchaks", "Solonetz", "Stagnosols", "Umbrisols", "Vertisols"]

        names = WRB
        if gdal is not None:
            try:
                names = gdal.Open(url).GetRasterBand(1).GetCategoryNames() or WRB
            except Exception:
                names = WRB

        def name_of(code):
            code = int(code)
            return names[code] if 0 <= code < len(names) and names[code] else f"class {code}"

        def summarise(arr):
            v = arr.ravel()
            v = v[v != 255]
            codes, counts = np.unique(v, return_counts=True)
            df = pd.DataFrame({
                "Code": codes.astype(int),
                "Soil group (WRB)": [name_of(c) for c in codes],
                "Pixels": counts,
            })
            if len(df):
                df["Share (%)"] = (100 * df["Pixels"] / df["Pixels"].sum()).round(1)
            return df.sort_values("Pixels", ascending=False).reset_index(drop=True)

        with rasterio.open(url) as src:
            aoi = self.gdf.to_crs(src.crs)
            soil, _ = mask(src, aoi.geometry, crop=True, all_touched=True)
            aoi_df = summarise(soil[0])

            c = aoi.union_all().centroid
            v = int(next(src.sample([(c.x, c.y)]))[0])
            centroid = {"code": v, "name": name_of(v) if v != 255 else "no data",
                        "x": c.x, "y": c.y}

            win = rasterio.windows.from_bounds(c.x - 0.005, c.y - 0.005,
                                               c.x + 0.005, c.y + 0.005, src.transform)
            window_df = summarise(src.read(1, window=win))

        return {"aoi": aoi_df, "centroid": centroid, "window": window_df}


# ----------------------------------------------------------------------------
# Cached runners (keyed on file paths, which include a content hash)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_analyzer(kml_path, gw_path, start, end):
    return SoilMoisture(kml_path, gw_path, start, end)


@st.cache_data(show_spinner=False)
def run_soil_moisture(kml_path, gw_path, start, end):
    df = get_analyzer(kml_path, gw_path, start, end).sm_data_extraction()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    return df


@st.cache_data(show_spinner=False)
def run_prospects(kml_path, gw_path, start, end, layer, style, samples, depth_field):
    with capture_output() as buf:
        res = get_analyzer(kml_path, gw_path, start, end).groundwater_prospects(
            layer=layer, style=style, samples=samples, depth_field=depth_field or None)
    return res, buf.getvalue()


@st.cache_data(show_spinner=False)
def run_comprehensive_gw(kml_path, gw_path, start, end):
    return get_analyzer(kml_path, gw_path, start, end).comprehensive_gw()


@st.cache_data(show_spinner=False)
def run_soil_type(kml_path, gw_path, start, end):
    return get_analyzer(kml_path, gw_path, start, end).get_soil_type()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Land & Water Explorer")
st.caption("Upload a field or village boundary to see soil moisture, groundwater and soil type for it.")

with st.sidebar:
    st.header("Area of interest")
    kml_file = st.file_uploader("Boundary (KML)", type=["kml"])

    st.header("Date range")
    default_end = date.today() - timedelta(days=7)  # ERA5-Land daily lags a few days
    start_date = st.date_input("Start", default_end - timedelta(days=90))
    end_date = st.date_input("End", default_end)

    st.header("Earth Engine")
    ee_project = st.text_input("Cloud project ID", EE_PROJECT,
                               help="Your Google Cloud project registered for Earth Engine.").strip()

    _sa, _secret_names = _load_service_account()
    if _sa:
        st.success(f"Service account found: {_sa.get('client_email', '?')}")
    elif _secret_names:
        st.warning("Secrets found, but no service account in them. "
                   f"Top-level names: {', '.join(_secret_names)}")
    else:
        st.info("No Streamlit secrets found — using local Earth Engine login.")
    st.caption(f"App version {APP_VERSION}")

if not kml_file:
    st.info("Upload a KML boundary in the sidebar to begin.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()

kml_path = save_upload(kml_file)
gw_path = GW_DATA_PATH if os.path.exists(GW_DATA_PATH) else None
start_str, end_str = start_date.isoformat(), end_date.isoformat()
args = (kml_path, gw_path, start_str, end_str)

try:
    analyzer = get_analyzer(*args)
except Exception as e:
    st.error(f"Could not read the uploaded files: {e}")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results = {}
results = st.session_state.results
run_id = hashlib.md5(str(args).encode()).hexdigest()


def result_key(name):
    return f"{run_id}:{name}"


tab_map, tab_sm, tab_gwp, tab_gwr, tab_soil = st.tabs(
    ["Map", "Soil moisture", "Groundwater prospects", "Groundwater resources", "Soil type"])

# ---------- Map ----------
with tab_map:
    zoom = st.slider("Basemap zoom", 10, 19, 17)
    with st.spinner("Drawing map…"):
        try:
            fig = analyzer.visualize(zoom=zoom)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not draw the map: {e}")
    st.caption(f"{len(analyzer.gdf)} feature(s) · CRS {analyzer.gdf.crs}")

# ---------- Soil moisture ----------
with tab_sm:
    st.write("Daily mean volumetric soil water (m³/m³) from ERA5-Land, averaged over the boundary.")
    if st.button("Get soil moisture", type="primary"):
        if not ee_project:
            st.error("Enter your Earth Engine Cloud project ID in the sidebar, "
                     "or set EE_PROJECT at the top of app.py.")
            st.stop()
        try:
            ee_user = init_ee(ee_project)
            st.caption(f"Earth Engine: connected with {ee_user}")
            with st.spinner("Querying Earth Engine…"):
                results[result_key("sm")] = run_soil_moisture(*args)
        except Exception as e:
            st.error(f"Soil moisture request failed: {e}")
            with st.expander("Error details"):
                st.exception(e)

    df = results.get(result_key("sm"))
    if df is not None:
        if df.empty:
            st.warning("No ERA5-Land images found for this date range.")
        else:
            value_cols = [c for c in df.columns if c.startswith("soil_water")]
            latest = df[value_cols].iloc[-1]
            cols = st.columns(len(value_cols))
            for col, name in zip(cols, value_cols):
                label = name.replace("soil_water_", "").replace("_", "–", 1).replace("_", "") \
                    .replace("cm", " cm")
                col.metric(label, f"{latest[name]:.3f}")
            st.line_chart(df[value_cols])
            st.dataframe(df[value_cols], use_container_width=True)
            st.download_button("Download CSV", df.to_csv().encode(),
                               file_name="soil_moisture.csv", mime="text/csv")

# ---------- Groundwater prospects ----------
with tab_gwp:
    st.write("Groundwater prospect classes, well depth and yield from Bhuvan hydrogeomorphology maps.")
    c1, c2, c3, c4 = st.columns(4)
    layer = c1.text_input("Layer", bhujal.DEFAULT_LAYER, help='Use "auto" to detect the state layer.')
    style = c2.text_input("Style", bhujal.DEFAULT_STYLE)
    samples = c3.number_input("Sample points", 0, 50, 4)
    depth_field = c4.text_input("Depth field (optional)")

    if st.button("Get groundwater prospects", type="primary"):
        try:
            with st.spinner("Querying Bhuvan…"):
                results[result_key("gwp")] = run_prospects(
                    *args, layer, style, int(samples), depth_field)
        except Exception as e:
            st.error(f"Groundwater prospects request failed: {e}")

    out = results.get(result_key("gwp"))
    if out is not None:
        res, log = out
        st.markdown(f"**Layer:** `{res['layer']}`")
        m1, m2 = st.columns(2)
        m1.markdown(f"#### {res['depth_text']}")
        m2.markdown(f"#### {res['yield_text']}")

        st.subheader(f"Prospects in {res['scope']}")
        if res["prospects"]:
            for t in res["prospects"]:
                st.markdown(f"- {t}")
        else:
            st.write("No legend classes returned.")

        units = sorted(res["units"].values(), key=lambda u: -u["hits"])
        if units:
            st.subheader("By hydrogeomorphic unit")
            st.dataframe(pd.DataFrame([{
                "Landform": u.get("landform"),
                "Lithology": u.get("lithology"),
                "Sample points": u["hits"],
                "Class": u.get("class") or u.get("depth") or "no class found",
            } for u in units]), use_container_width=True, hide_index=True)
        elif samples > 0:
            st.info("No features found at the sampled points.")

        if log.strip():
            with st.expander("Log"):
                st.code(log)

# ---------- Groundwater resources ----------
with tab_gwr:
    st.write("Recharge, extraction and stage of extraction for the assessment units that overlap the boundary.")
    if gw_path is None:
        st.error(f"Groundwater layer not found at {GW_DATA_PATH}. Update GW_DATA_PATH at the top of app.py.")
    else:
        if st.button("Get groundwater resources", type="primary"):
            try:
                with st.spinner("Intersecting layers…"):
                    results[result_key("gwr")] = run_comprehensive_gw(*args)
            except Exception as e:
                st.error(f"Groundwater resources lookup failed: {e}")

        out = results.get(result_key("gwr"))
        if out is not None:
            table, missing = out
            if table.empty:
                st.warning("No assessment units overlap this boundary.")
            else:
                stage_col = "Stage of Ground Water Extraction (%)"
                if stage_col in table.columns:
                    stage = pd.to_numeric(table[stage_col], errors="coerce").dropna()
                    if not stage.empty:
                        st.metric("Stage of groundwater extraction", f"{stage.mean():.1f} %")
                st.dataframe(table, use_container_width=True)
                st.download_button("Download CSV", table.to_csv(index=False).encode(),
                                   file_name="groundwater_resources.csv", mime="text/csv")
            if missing:
                st.caption("Columns not in the uploaded layer: " + ", ".join(missing))

# ---------- Soil type ----------
with tab_soil:
    st.write("Most probable WRB soil group from ISRIC SoilGrids (250 m).")
    if st.button("Get soil type", type="primary"):
        try:
            with st.spinner("Reading SoilGrids…"):
                results[result_key("soil")] = run_soil_type(*args)
        except Exception as e:
            st.error(f"Soil type lookup failed: {e}")

    soil = results.get(result_key("soil"))
    if soil is not None:
        cen = soil["centroid"]
        st.metric("Soil group at centroid", cen["name"])
        st.caption(f"Centroid x={cen['x']:.1f}, y={cen['y']:.1f} (SoilGrids Homolosine projection)")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Inside the boundary")
            if soil["aoi"].empty:
                st.write("No soil data inside the boundary.")
            else:
                st.dataframe(soil["aoi"], use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Around the centroid (~1 km)")
            if soil["window"].empty:
                st.write("No soil data in this window.")
            else:
                st.dataframe(soil["window"], use_container_width=True, hide_index=True)