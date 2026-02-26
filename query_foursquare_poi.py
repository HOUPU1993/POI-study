import os
import json
import base64
import numpy as np
import geopandas as gpd
from typing import Optional
from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual
from pyiceberg.catalog import load_catalog


def read_token(token_path: str) -> str:
    """
    Read and return the Foursquare API token from a local JSON file.

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


def json_str_tr(x):
    """
    Convert non-JSON-serializable Python objects (e.g., lists, dicts, arrays)
    into string representations that can be safely exported or visualized.

    Parameters
    ----------
    x : Any
        The input value to be converted.

    Returns
    -------
    str or None
        The JSON-compatible string representation of the input value.
    """
    if x is None:
        return None
    elif isinstance(x, (str, int, float, bool)):
        return str(x)
    elif isinstance(x, (list, tuple, set, np.ndarray)):
        return str(list(x))
    elif isinstance(x, dict):
        return str({k: json_str_tr(v) for k, v in x.items()})
    elif isinstance(x, (bytes, bytearray)):
        return base64.b64encode(x).decode("ascii")
    else:
        return str(x)


def query_foursquare(
    token_path: str,
    minLon: float,
    maxLon: float,
    minLat: float,
    maxLat: float,
    limit_size: Optional[int] = None,
    table_name: str = "datasets.places_os",
    uri: str = "https://catalog.h3-hub.foursquare.com/iceberg",
    warehouse: str = "places",
):
    """
    Query Points of Interest (POIs) from the Foursquare Iceberg Catalog
    within a given geographic bounding box and return results as a GeoDataFrame.

    This function connects to the Foursquare H3-Hub Iceberg dataset, applies a spatial filter
    (latitude/longitude bounds), retrieves the data as a Pandas DataFrame, converts it into
    a GeoDataFrame (EPSG:4326), and ensures all complex fields are JSON-serializable.

    Parameters
    ----------
    token_path : str
        Path to the local JSON file containing the Foursquare API token.
        The file should follow one of these formats:
        - {"token": "xxxxx"}
        - "xxxxx"
    minLon, maxLon, minLat, maxLat : float
        Geographic bounding box coordinates (in WGS84) for the query.
    limit_size : int, optional
        Maximum number of rows to return. If None (default), returns all available rows.
    table_name : str, default "datasets.places_os"
        Name of the Iceberg table to query.
    uri : str, default "https://catalog.h3-hub.foursquare.com/iceberg"
        Base URI of the Foursquare Iceberg REST catalog.
    warehouse : str, default "places"
        Name of the warehouse used by the Iceberg catalog.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing POI records with WGS84 geometry.

    Examples
    --------
    Limit to 5000 rows:
    >>> gdf = query_foursquare(
    ...     token_path="~/foursquare_token.json",
    ...     minLon=-119.8694, maxLon=-119.85346,
    ...     minLat=34.40887, maxLat=34.41727,
    ...     limit_size=5000
    ... )

    Retrieve all available records (no limit):
    >>> gdf = query_foursquare(
    ...     token_path="~/foursquare_token.json",
    ...     minLon=-119.8694, maxLon=-119.85346,
    ...     minLat=34.40887, maxLat=34.41727
    ... )
    """
    # --- Load token ---
    token = read_token(os.path.expanduser(token_path))

    # --- Connect to the Iceberg catalog ---
    catalog = load_catalog(
        "default",
        **{
            "warehouse": warehouse,
            "uri": uri,
            "token": token,
            "header.content-type": "application/vnd.api+json",
            "rest-metrics-reporting-enabled": "false",
        },
    )

    # --- Load target table ---
    table = catalog.load_table(table_name)

    # --- Define spatial filter ---
    expr = And(
        And(GreaterThanOrEqual("longitude", minLon), LessThanOrEqual("longitude", maxLon)),
        And(GreaterThanOrEqual("latitude", minLat), LessThanOrEqual("latitude", maxLat)),
    )

    # --- Run query ---
    df = table.scan(row_filter=expr, limit=limit_size).to_pandas()

    # --- Convert to GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # --- Clean up non-serializable columns ---
    for col in gdf.columns:
        if col != "geometry":
            if gdf[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray, bytes, bytearray))).any():
                gdf[col] = gdf[col].apply(json_str_tr)

    return gdf
