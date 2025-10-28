#%%

#!/usr/bin/env python3

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]

# ------------------------------------------------------------------
#  Sky‑coverage (zenith ≤ 45 °) of a telescope network
#  ────────────────────────────────────────────────────────────────
#  • Generates true geodesic caps on the WGS‑84 ellipsoid.
#  • Robust against     – meridian (±180 °) crossings
#                        – clockwise ring orientation
#                        – self‑intersection artefacts
#  • Projection‑agnostic geometry → can be reused on any map / globe.
#  • Renders on a Mercator map with Cartopy; change `projection = …`
#    to visualise in Orthographic, Mollweide, etc.
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.geometry.polygon import orient            # Shapely < 2.0
from shapely.ops import split

# ----------------------------  USER SECTION  ----------------------------
STATIONS = [                         # name, latitude [°N], longitude [°E]
    ("Madrid",          40.4168,  -3.7038),
    ("Warsaw",          52.2297,  21.0122),
    ("Puebla",          19.0414, -98.2063),
    ("Monterrey",       25.6866, -100.3161),
]

ZENITH_RADIUS_DEG = 89             # angular radius of the cap
N_VERTICES        = 1440              # polygon resolution
# ------------------------------------------------------------------------

# CONSTANTS
EARTH_RADIUS_M = 6371008.8           # mean Earth radius (IUGG 2019)
GEOD           = Geod(ellps="WGS84") # geodesic solver

# ------------------------  LOW‑LEVEL UTILITIES --------------------------
def wrap180(lon):
    """Confine longitude to [−180°, +180°)."""
    return ((lon + 180.0) % 360.0) - 180.0

def geodesic_circle(lat0, lon0, radius_deg, n_vertices=N_VERTICES):
    """
    Great‑circle (lon, lat) points of a constant‑zenith‑angle cap.
    """
    az     = np.linspace(0.0, 360.0, n_vertices, endpoint=False, dtype=np.float64)
    #dist   = np.full_like(az, np.deg2rad(radius_deg) * EARTH_RADIUS_M)
    
    height_m = 15000
    theta = np.deg2rad(radius_deg)
    ground_dist = height_m * np.tan(theta)
    dist = np.full_like(az, ground_dist)
    
    lon_v  = np.full_like(az, lon0, dtype=np.float64)
    lat_v  = np.full_like(az, lat0, dtype=np.float64)
    lon_c, lat_c, _ = GEOD.fwd(lon_v, lat_v, az, dist)
    return wrap180(lon_c), lat_c

def split_dateline(poly):
    """
    Split *poly* at the ±180 ° meridian and shift longitudes so that
    every resulting part lies entirely within [−180°, +180°).
    """
    dateline = LineString([(180.0, -90.0), (180.0, 90.0)])
    if not poly.crosses(dateline):
        return poly

    parts = split(poly, dateline)
    fixed = []
    for part in parts:
        coords = [((x - 360.0) if x > 180.0 else x, y)
                  for x, y in part.exterior.coords]
        fixed.append(orient(Polygon(coords).buffer(0), sign=1.0))
    return MultiPolygon(fixed)

def build_cap(lat0, lon0, radius_deg=ZENITH_RADIUS_DEG):
    """
    Return a Shapely (Multi)Polygon representing the 45°‑zenith cap
    centred at (lat0, lon0).  Output is projection‑agnostic.
    """
    lon_ring, lat_ring = geodesic_circle(lat0, lon0, radius_deg)
    raw_poly = orient(Polygon(zip(lon_ring, lat_ring)).buffer(0), sign=1.0)
    return split_dateline(raw_poly)
# ------------------------------------------------------------------------

# Construct one (Multi)Polygon per station
footprints = {name: build_cap(lat, lon)
              for name, lat, lon in STATIONS}

# -----------------------------  PLOTTING  -------------------------------
projection = ccrs.Orthographic(central_longitude=-45, central_latitude=40)        # e.g. ccrs.Mercator()
fig = plt.figure(figsize=(12, 6))
ax  = plt.axes(projection=projection)
ax.set_global()

# Minimal base map
ax.add_feature(cfeature.LAND.with_scale("50m"),
               facecolor="white",  edgecolor="gray")
ax.add_feature(cfeature.OCEAN.with_scale("50m"),
               facecolor="lightblue")
ax.coastlines(resolution="50m", linewidth=0.5)

colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for idx, (name, lat, lon) in enumerate(STATIONS):
    geom = footprints[name]
    ax.add_geometries(
        geom if isinstance(geom, (Polygon, MultiPolygon)) else [geom],
        crs=ccrs.PlateCarree(),
        facecolor=colour_cycle[idx % len(colour_cycle)],
        edgecolor="none",
        alpha=0.25,
        label=f"{name} ({ZENITH_RADIUS_DEG}° cap)",
    )
    ax.plot(lon, lat, "o", markersize=4,
            color=colour_cycle[idx % len(colour_cycle)],
            transform=ccrs.PlateCarree())

# Legend – one entry per station (skip duplicates from add_geometries)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::2], labels[::2],
          loc="lower left", frameon=False, fontsize="small")

ax.set_title(f"Sky‑coverage (zenith ≤ {ZENITH_RADIUS_DEG}°) of ground stations", fontsize="medium")
#plt.show()
# Save figure
plt.savefig(f"{home_path}/DATAFLOW_v3/TESTS/network_map.png", bbox_inches="tight", dpi=300)

# %%
