def add_distance_bins_to_point(
    gdf,
    ref,
    n_bins=20,
    src_crs="EPSG:4326",
    proj_crs="EPSG:3857",
    dist_col="dist_to_point",
    bin_col="dist_bin"
):

    gdf = gdf.copy()

    # reference point
    pt = ref
    pt_gdf = gpd.GeoDataFrame(geometry=[pt], crs=src_crs)

    # project
    gdf_proj = gdf.to_crs(proj_crs)
    pt_proj = pt_gdf.to_crs(proj_crs)

    # distance (meters)
    gdf[dist_col] = gdf_proj.geometry.distance(pt_proj.geometry.iloc[0]) 

    # bins
    dmin, dmax = gdf[dist_col].min(), gdf[dist_col].max()
    bins = np.linspace(dmin, dmax, n_bins + 1)

    gdf[bin_col] = pd.cut(gdf[dist_col],bins=bins,include_lowest=True)

    return gdf