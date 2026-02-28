def summarize_match_miss_by_dist_bin(
    gdf: pd.DataFrame,
    cat_col: str = "primary_cat",
    dist_bin_col: str = "dist_bin",
    true_col: str = "is_true_match",
    match_value= '1',
    name_score_col: str = "name_score", 
    n_mistake_col: str = "n_mistake",
    n_miss_col: str = "n_miss",
    n_nonmatch_col: str = "n_nonmatch",
    n_match_col: str = "n_match",
    nonmtach_c_col: str = "non_match_c",
    match_c_col: str = "match_c",
    non_match_density_col: str = "nonmatch_den",
    match_density_col: str = "match_den",
    total_density_col: str = "total_den", 
    median_name_score_col: str = "median_name_score", 
    ring_area_col: str = "ring_area",
    bin_id_col: str = "bin_id",
    area_unit: str = "m2",   # "m2" or "km2"
) -> pd.DataFrame:

    df = gdf.copy()

    # group stats
    df_out = (
        df.groupby([cat_col, dist_bin_col], observed=True)
          .apply(lambda x: pd.Series({
              "n_total": len(x),
              n_match_col: x[true_col].eq(match_value).sum(),
              n_nonmatch_col:  (~x[true_col].eq(match_value)).sum(),  # NaN -> miss
              n_mistake_col: x[true_col].eq("0").sum(), # n_mistake_match: true_col == "0"
              n_miss_col: x[true_col].isna().sum(), # n_first_miss: true_col 是 NaN
              median_name_score_col: x[name_score_col].median(),
          }))
          .reset_index()
    )

    # proportions within each bin
    df_out[match_c_col] = df_out[n_match_col] / (df_out["n_total"])
    df_out[nonmtach_c_col]  = df_out[n_nonmatch_col]  / (df_out["n_total"])

    # bin order within each category
    df_out[bin_id_col] = df_out.groupby(cat_col, observed=True).cumcount() + 1

    # ring area
    area = df_out[dist_bin_col].apply(lambda x: np.pi * (x.right**2 - x.left**2)).astype(float)
    if area_unit == "km2":
        area = area / 1e6
    df_out[ring_area_col] = area

    # densities (per m² or per km²)
    df_out[non_match_density_col]  = df_out[n_nonmatch_col]  / (df_out[ring_area_col])
    df_out[match_density_col] = df_out[n_match_col] / (df_out[ring_area_col])
    df_out[total_density_col] = df_out["n_total"] / (df_out[ring_area_col])
    
    return df_out