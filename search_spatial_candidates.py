import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree


def search_spatial_candidates(
    reference_gdf: gpd.GeoDataFrame,
    compared_gdf: gpd.GeoDataFrame,
    k: int = 100,
    id_col: str = "id",
    crs_for_distance: int = 3857,
):
    """
    Attach k nearest compared POI ids & distances to reference_gdf.

    Returns
    -------
    GeoDataFrame with two new columns:
    - cand_ids   : list of compared ids
    - cand_dist_m: list of distances (meters)
    """

    ref_proj = reference_gdf.to_crs(crs_for_distance)
    cmp_proj = compared_gdf.to_crs(crs_for_distance)

    ref_xy = np.column_stack([ref_proj.geometry.x, ref_proj.geometry.y])
    cmp_xy = np.column_stack([cmp_proj.geometry.x, cmp_proj.geometry.y])

    tree = cKDTree(cmp_xy)
    k_eff = min(k, len(compared_gdf))

    dist, idx = tree.query(ref_xy, k=k_eff)

    if k_eff == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    cmp_ids = compared_gdf[id_col].to_numpy()

    result = reference_gdf.copy()
    result["cand_ids"] = [[cmp_ids[j] for j in row] for row in idx]
    result["cand_dist_m"] = dist.tolist()

    return result