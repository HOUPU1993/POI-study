import math
import time
import requests
import json
import warnings
from typing import Optional, Iterable, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def read_token(token_path: str) -> str:
    """
    Read and return the token from a local JSON file.

    The function supports two valid JSON formats:
    1) An object with a "token" key: {"token": "xxxxx"}
    2) A plain string containing the token: "xxxxx"

    Parameters
    ----------
    token_path : str
        Local file path to the JSON file containing the token.

    Returns
    -------
    str
        The token string extracted from the file.

    Raises
    ------
    ValueError
        If the JSON file format is invalid or does not contain a valid token.
    """
    with open(token_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "token" in data:
        return data["token"]
    if isinstance(data, str):
        return data
    raise ValueError("Token JSON must be either a string or an object with a key 'token'.")


def circle_center(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    R: float,
    base: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Generate a grid of circle polygons inside a geographic bounding box.
    Returns circles in EPSG:4326 (lon/lat).

    If `base` is provided, return only circles that intersect `base`
    (via spatial join, predicate='intersects').

    Parameters
    ----------
    min_lon, max_lon, min_lat, max_lat : float
        Bounding box in EPSG:4326.
    R : float
        Grid half-spacing in meters (in EPSG:3857). Spacing is 2R.
        Each circle radius is R * sqrt(2) in meters (in EPSG:3857).
    base : geopandas.GeoDataFrame | None
        A GeoDataFrame used to filter circles by intersection.
        Will be reprojected to EPSG:4326 if needed.

    Returns
    -------
    geopandas.GeoDataFrame
        Circles (filtered if base is provided), in EPSG:4326.
    """
    # -----------------------------
    # 0) Basic input validation
    # -----------------------------
    if not (isinstance(min_lon, (int, float)) and isinstance(max_lon, (int, float))
            and isinstance(min_lat, (int, float)) and isinstance(max_lat, (int, float))):
        raise TypeError("min_lon, max_lon, min_lat, max_lat must be numeric (float or int).")
    if not (max_lon > min_lon and max_lat > min_lat):
        raise ValueError("max_lon must be > min_lon and max_lat must be > min_lat.")
    if not (isinstance(R, (int, float)) and R > 0):
        raise ValueError("R must be a positive number (meters).")

    if base is not None:
        if not isinstance(base, gpd.GeoDataFrame):
            raise TypeError("base must be a GeoDataFrame or None.")
        if base.geometry is None:
            raise ValueError("base must have a valid geometry column.")
        if base.crs is None:
            raise ValueError("base.crs is None. Please set a CRS on base before using it.")

    # -----------------------------
    # 1) Build bbox in EPSG:4326
    # -----------------------------
    poly_4326 = box(min_lon, min_lat, max_lon, max_lat)
    gdf_bbox_4326 = gpd.GeoDataFrame({"name": ["bbox_4326"]}, geometry=[poly_4326], crs="EPSG:4326")

    # -----------------------------
    # 2) Reproject bbox to EPSG:3857 (meters)
    # -----------------------------
    gdf_bbox_3857 = gdf_bbox_4326.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf_bbox_3857.total_bounds

    # -----------------------------
    # 3) Grid centers inside bbox
    # -----------------------------
    xs = np.arange(minx + R, maxx + 1e-9, 2 * R)
    ys = np.arange(miny + R, maxy + 1e-9, 2 * R)

    if xs.size == 0 or ys.size == 0:
        warnings.warn("⚠️ No grid centers generated — check if R is too large or bbox too small.")
        centers_xy = np.empty((0, 2))
    else:
        xx, yy = np.meshgrid(xs, ys)
        centers_xy = np.column_stack([xx.ravel(), yy.ravel()])

    # -----------------------------
    # 4) Build circles in 3857 -> back to 4326
    # -----------------------------
    radius_m = R
    circles_3857 = [Point(cx, cy).buffer(radius_m, resolution=64) for (cx, cy) in centers_xy]
    circles = gpd.GeoDataFrame(geometry=circles_3857, crs="EPSG:3857").to_crs(epsg=4326)

    # -----------------------------
    # 5) Optional: filter circles by intersection with base
    # -----------------------------
    base_4326 = base.to_crs("EPSG:4326")
    circles_4326 = (
        gpd.sjoin(circles, base_4326, how="inner", predicate="intersects")
            .reset_index(drop=True)
    )
    circles_4326 = circles_4326[circles.columns]
    circles_m = circles_4326.to_crs(epsg=3857)
    centroids_m = circles_m.geometry.centroid
    centroids_ll = centroids_m.to_crs(epsg=4326)
    circles_4326["center_lon"] = centroids_ll.x
    circles_4326["center_lat"] = centroids_ll.y
    
    return circles_4326

def places_nearby_grid(
    circles: gpd.GeoDataFrame,
    *,
    token: str,
    R: float,
    place_types: List[str],
    field_mask: Iterable[str],
    max_result_count: int = 20,
    sleep_sec: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Query Google Places Nearby Search for each circle center.

    Parameters
    ----------
    circles : GeoDataFrame
        Must contain columns ['center_lon', 'center_lat'] in EPSG:4326.
    token : str
        Google Places API key.
    R : float
        Search radius in meters (should match circle_center R).
    place_types : list[str]
        includedPrimaryTypes sent to Google Places.
    field_mask : iterable[str]
        Fields requested via X-Goog-FieldMask.
    max_result_count : int, default 20
        Max results per query (Google cap).
    sleep_sec : float, default 0.1
        Sleep time between requests to avoid rate limiting.

    Returns
    -------
    GeoDataFrame
        POIs returned by Google Places Nearby Search (EPSG:4326).
    """

    url = "https://places.googleapis.com/v1/places:searchNearby"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": token,
        "X-Goog-FieldMask": ",".join(field_mask),
    }

    rows: List[Dict[str, Any]] = []

    for circle_id, row in circles.iterrows():
        lat = row["center_lat"]
        lon = row["center_lon"]

        payload = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": float(lat),
                        "longitude": float(lon),
                    },
                    "radius": float(R),
                }
            },
            "includedPrimaryTypes": place_types,
            "maxResultCount": int(max_result_count),
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()

        places = data.get("places", [])

        for p in places:
            rows.append({
                "circle_id": circle_id,
                "id": p.get("id"),
                "name": p.get("displayName", {}).get("text"),
                "address": p.get("formattedAddress"),
                "primary_type": p.get("primaryType"),
                "lat": p.get("location", {}).get("latitude"),
                "lon": p.get("location", {}).get("longitude"),
                "business_status": p.get("businessStatus"),
            })

        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    return gdf