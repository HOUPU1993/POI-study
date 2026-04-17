"""
Location Difference Analysis Pipeline
======================================
Mirrors run_mr_analysis structure.

KEY DESIGN DECISION:
    location_distance is only computed for TRUE matches (is_true_match = 1).
    Unmatched POIs (NaN) and false matches (is_true_match = 0) are excluded.

Four modules:
    1. Overall Location Difference  (median, mean, pct within thresholds)
    2. Location Difference by Category (primary_cat)
    3. Location Difference by Distance  + Methods A / C (each OLS + WLS)
    4. Location Difference by Pop Density + Methods A / C (each OLS + WLS)

Returns:
    overall_loc, by_category_loc
    distance_bins_loc
    distance_loc_reg_A_ols, distance_loc_reg_A_wls
    distance_loc_reg_C_ols, distance_loc_reg_C_wls
    pop_bins_loc
    pop_loc_reg_A_ols, pop_loc_reg_A_wls
    pop_loc_reg_C_ols, pop_loc_reg_C_wls
    tract_gdf
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
from statsmodels.formula.api import ols, wls


# ==============================================================================
# HELPERS
# ==============================================================================

def _get_true_match_location_distance(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Return location_distance only for true matches (is_true_match = 1)."""
    true_match_mask = gdf["is_true_match"].map(
        lambda x: str(x).strip() in ("1", "1.0", "True", "true")
    )
    return gdf["location_distance"].where(true_match_mask)


def _fit_both_loc(formula: str, data: pd.DataFrame,
                  weight_col: str, x_col: str) -> tuple:
    """Fit OLS and WLS for the same formula. Returns (ols_dict, wls_dict)."""
    results = {}
    for method_tag, model_fn in [
        ("OLS", lambda: ols(formula, data=data).fit()),
        ("WLS", lambda: wls(formula, data=data, weights=data[weight_col]).fit()),
    ]:
        try:
            m = model_fn()
            results[method_tag] = dict(
                slope     = m.params[x_col],
                intercept = m.params["Intercept"],
                r_squared = m.rsquared,
                p_value   = m.pvalues[x_col],
                std_err   = m.bse[x_col],
            )
        except Exception as e:
            print(f"  [warn] {method_tag} failed: {e}")
            results[method_tag] = dict(
                slope=np.nan, intercept=np.nan,
                r_squared=np.nan, p_value=np.nan, std_err=np.nan,
            )
    return results["OLS"], results["WLS"]


# ==============================================================================
# CENSUS TRACT HELPERS
# ==============================================================================

def _get_msa_counties(cbsa_code: str, state_fips_list: list) -> dict:
    url = (
        "https://www2.census.gov/programs-surveys/metro-micro/"
        "geographies/reference-files/2020/delineation-files/list1_2020.xls"
    )
    df = pd.read_excel(url, header=2)
    df.columns = [str(c).strip() for c in df.columns]
    df["CBSA Code"]        = df["CBSA Code"].astype(str).str.split(".").str[0].str.zfill(5)
    df["FIPS State Code"]  = df["FIPS State Code"].astype(str).str.split(".").str[0].str.zfill(2)
    df["FIPS County Code"] = df["FIPS County Code"].astype(str).str.split(".").str[0].str.zfill(3)

    counties_by_state = {}
    for state_fips in state_fips_list:
        filtered = df[
            (df["CBSA Code"]       == str(cbsa_code).zfill(5)) &
            (df["FIPS State Code"] == state_fips)
        ]
        counties = (
            filtered["FIPS State Code"] + filtered["FIPS County Code"]
        ).tolist()
        if counties:
            counties_by_state[state_fips] = counties
    return counties_by_state


def fetch_msa_tracts(
    cbsa_code: str, state_fips_list: list, api_key: str, year: int = 2020,
) -> gpd.GeoDataFrame:
    """Fetch Census tract geometries + ACS population. Returns EPSG:4326 GDF."""
    print(f"  [tracts] CBSA {cbsa_code} ...")
    counties_by_state = _get_msa_counties(cbsa_code, state_fips_list)
    if not counties_by_state:
        raise RuntimeError(f"No counties found for CBSA {cbsa_code}.")

    tract_records = []
    for state_fips, counties in counties_by_state.items():
        for cfull in counties:
            url = (
                f"https://api.census.gov/data/{year}/acs/acs5"
                f"?get=NAME,B01003_001E&for=tract:*"
                f"&in=state:{state_fips}+county:{cfull[2:]}&key={api_key}"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
            tract_records.append(pd.DataFrame(data[1:], columns=data[0]))

    tract_df = pd.concat(tract_records, ignore_index=True)
    tract_df["GEOID"]      = tract_df["state"] + tract_df["county"] + tract_df["tract"]
    tract_df["population"] = pd.to_numeric(tract_df["B01003_001E"], errors="coerce")

    geom_frames = []
    for state_fips in state_fips_list:
        gdf_s = gpd.read_file(
            f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
            f"tl_{year}_{state_fips}_tract.zip"
        )
        geom_frames.append(gdf_s[["GEOID", "geometry"]])

    all_county_fips = [c for cs in counties_by_state.values() for c in cs]
    merged = pd.concat(geom_frames, ignore_index=True).merge(
        tract_df[["GEOID", "population"]], on="GEOID", how="inner"
    )
    merged = merged[merged["GEOID"].str[:5].isin(all_county_fips)].copy()

    merged_proj           = merged.to_crs("EPSG:3857")
    merged["area_km2"]    = merged_proj.geometry.area / 1e6
    merged["pop_density"] = (
        merged["population"] / merged["area_km2"]
    ).replace([np.inf, -np.inf], np.nan)
    return merged[["GEOID", "population", "area_km2", "pop_density", "geometry"]]


def _join_pop_to_poi(
    gdf: gpd.GeoDataFrame, tract_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        gdf.to_crs("EPSG:3857"),
        tract_gdf[["GEOID", "population", "area_km2", "pop_density", "geometry"]
                  ].to_crs("EPSG:3857"),
        how="left", predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return joined.to_crs("EPSG:4326")


# ==============================================================================
# MODULE 1 — Overall Location Difference
# ==============================================================================

def compute_overall_location_diff(datasets: dict) -> pd.DataFrame:
    """
    Compute overall location difference for TRUE matched POIs per dataset.

    Returns
    -------
    pd.DataFrame — dataset, n_true_matched, median_dist_m, mean_dist_m,
                   pct_within_10m, pct_within_50m
    """
    records = []
    for ds_name, gdf in datasets.items():
        dist = _get_true_match_location_distance(gdf).dropna()
        n    = len(dist)
        records.append({
            "dataset":        ds_name,
            "n_true_matched": n,
            "median_dist_m":  dist.median() if n > 0 else np.nan,
            "mean_dist_m":    dist.mean()   if n > 0 else np.nan,
            "pct_within_10m": (dist <= 10).mean() if n > 0 else np.nan,
            "pct_within_50m": (dist <= 50).mean() if n > 0 else np.nan,
        })
    return pd.DataFrame(records)


# ==============================================================================
# MODULE 2 — Location Difference by Category
# ==============================================================================

def compute_location_diff_by_category(
    datasets: dict, min_count: int = 10,
) -> pd.DataFrame:
    """
    Compute median location distance grouped by primary_cat.
    Only true matches contribute to distance metrics.
    min_count filters on total_poi (all POIs).

    Returns
    -------
    pd.DataFrame — dataset, primary_cat, total_poi, n_true_matched,
                   median_dist_m, mean_dist_m, pct_within_10m, pct_within_50m
    """
    records = []
    for ds_name, gdf in datasets.items():
        gdf = gdf.copy()
        gdf["_loc_dist"] = _get_true_match_location_distance(gdf)

        grp = gdf.groupby("primary_cat").agg(
            total_poi      = ("_loc_dist", "size"),
            n_true_matched = ("_loc_dist", "count"),
            median_dist_m  = ("_loc_dist", "median"),
            mean_dist_m    = ("_loc_dist", "mean"),
        ).reset_index()

        grp["pct_within_10m"] = gdf.groupby("primary_cat")["_loc_dist"].apply(
            lambda x: (x <= 10).sum() / x.count() if x.count() > 0 else np.nan
        ).values
        grp["pct_within_50m"] = gdf.groupby("primary_cat")["_loc_dist"].apply(
            lambda x: (x <= 50).sum() / x.count() if x.count() > 0 else np.nan
        ).values

        grp = grp[grp["total_poi"] >= min_count].copy()
        grp["dataset"] = ds_name
        records.append(grp[[
            "dataset", "primary_cat", "total_poi", "n_true_matched",
            "median_dist_m", "mean_dist_m", "pct_within_10m", "pct_within_50m",
        ]])
    return pd.concat(records, ignore_index=True)


# ==============================================================================
# MODULE 3 — Location Difference by Distance (Methods A / C, each OLS + WLS)
# ==============================================================================

def compute_location_dist_regression(
    datasets: dict, center: Point, tract_gdf: gpd.GeoDataFrame,
    n_bins: int = 50, min_count: int = 20,
) -> tuple:
    """
    Method A: bin aggregation  — bin_median_dist ~ bin_mid_km
    Method C: tract level      — tract_median_dist ~ dist_to_center_km
    Each method runs both OLS and WLS (weight = total_poi).

    Returns
    -------
    bin_df, reg_A_ols, reg_A_wls, reg_C_ols, reg_C_wls
    """
    bin_records          = []
    rec_A_ols, rec_A_wls = [], []
    rec_C_ols, rec_C_wls = [], []

    tract_proj = tract_gdf.to_crs("EPSG:3857").copy()
    pt_gdf     = gpd.GeoDataFrame(
        geometry=[center], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    tract_proj["dist_to_center_km"] = (
        tract_proj.geometry.centroid.distance(pt_gdf.geometry.iloc[0]) / 1000
    )
    tract_meta = tract_proj[["GEOID", "dist_to_center_km"]].copy()

    for ds_name, gdf in datasets.items():

        gdf_proj = gdf.to_crs("EPSG:3857").copy()
        gdf_proj["dist_to_center_km"] = (
            gdf_proj.geometry.distance(pt_gdf.geometry.iloc[0]) / 1000
        )
        gdf_work              = gdf_proj.to_crs("EPSG:4326").copy()
        gdf_work["_loc_dist"] = _get_true_match_location_distance(gdf)

        # ── Method A ─────────────────────────────────────────────────────
        dmin = gdf_work["dist_to_center_km"].min()
        dmax = gdf_work["dist_to_center_km"].max()
        gdf_work["dist_bin"] = pd.cut(
            gdf_work["dist_to_center_km"],
            bins=np.linspace(dmin, dmax, n_bins + 1),
            include_lowest=True,
        )

        bs = gdf_work.groupby("dist_bin", observed=True).agg(
            total_poi      = ("_loc_dist", "size"),
            n_true_matched = ("_loc_dist", "count"),
            median_dist_m  = ("_loc_dist", "median"),
        ).reset_index()
        bs["bin_mid_km"] = bs["dist_bin"].apply(
            lambda iv: (iv.left + iv.right) / 2
        ).astype(float)
        bs["dataset"] = ds_name
        bin_records.append(bs)

        valid_A = bs[bs["total_poi"] >= min_count].dropna(
            subset=["bin_mid_km", "median_dist_m"]
        ).reset_index(drop=True)

        base = dict(dataset=ds_name, n_used=len(valid_A))
        if len(valid_A) >= 3:
            r_ols, r_wls = _fit_both_loc(
                "median_dist_m ~ bin_mid_km", valid_A, "total_poi", "bin_mid_km"
            )
        else:
            r_ols = r_wls = dict(
                slope=np.nan, intercept=np.nan,
                r_squared=np.nan, p_value=np.nan, std_err=np.nan,
            )
        rec_A_ols.append({**base, "method": "A_OLS_bin",  **r_ols})
        rec_A_wls.append({**base, "method": "A_WLS_bin",  **r_wls})

        # ── Method C ─────────────────────────────────────────────────────
        gdf_c              = _join_pop_to_poi(gdf, tract_gdf).copy()
        gdf_c["_loc_dist"] = _get_true_match_location_distance(gdf).values

        tract_c = gdf_c.groupby("GEOID").agg(
            total_poi      = ("_loc_dist", "size"),
            n_true_matched = ("_loc_dist", "count"),
            median_dist_m  = ("_loc_dist", "median"),
        ).reset_index()
        tract_c = tract_c.merge(tract_meta, on="GEOID", how="left")
        tract_c = tract_c[tract_c["total_poi"] >= min_count].dropna(
            subset=["dist_to_center_km", "median_dist_m"]
        ).reset_index(drop=True)

        base_c = dict(dataset=ds_name, n_used=len(tract_c))
        if len(tract_c) >= 3:
            r_ols, r_wls = _fit_both_loc(
                "median_dist_m ~ dist_to_center_km",
                tract_c, "total_poi", "dist_to_center_km",
            )
        else:
            r_ols = r_wls = dict(
                slope=np.nan, intercept=np.nan,
                r_squared=np.nan, p_value=np.nan, std_err=np.nan,
            )
        rec_C_ols.append({**base_c, "method": "C_OLS_tract", **r_ols})
        rec_C_wls.append({**base_c, "method": "C_WLS_tract", **r_wls})

    bin_df = pd.concat(bin_records, ignore_index=True)
    return (
        bin_df,
        pd.DataFrame(rec_A_ols), pd.DataFrame(rec_A_wls),
        pd.DataFrame(rec_C_ols), pd.DataFrame(rec_C_wls),
    )


# ==============================================================================
# MODULE 4 — Location Difference by Population Density (Methods A / C, each OLS + WLS)
# ==============================================================================

def compute_location_pop_regression(
    datasets: dict, tract_gdf: gpd.GeoDataFrame,
    n_bins: int = 50, min_count: int = 20,
) -> tuple:
    """
    Method A: bin aggregation  — bin_median_dist ~ log(bin_mid_pop)
    Method C: tract level      — tract_median_dist ~ log(pop_density)
    Each method runs both OLS and WLS (weight = total_poi).

    Returns
    -------
    bin_df, reg_A_ols, reg_A_wls, reg_C_ols, reg_C_wls
    """
    bin_records          = []
    rec_A_ols, rec_A_wls = [], []
    rec_C_ols, rec_C_wls = [], []

    for ds_name, gdf in datasets.items():

        gdf_pop              = _join_pop_to_poi(gdf, tract_gdf).copy()
        gdf_pop["_loc_dist"] = _get_true_match_location_distance(gdf).values
        valid_mask           = gdf_pop["pop_density"].notna()

        # ── Method A ─────────────────────────────────────────────────────
        bins = np.unique(np.percentile(
            gdf_pop.loc[valid_mask, "pop_density"],
            np.linspace(0, 100, n_bins + 1),
        ))
        gdf_pop["pop_bin"] = pd.cut(
            gdf_pop["pop_density"], bins=bins, include_lowest=True
        )

        bs = gdf_pop.groupby("pop_bin", observed=True).agg(
            total_poi      = ("_loc_dist", "size"),
            n_true_matched = ("_loc_dist", "count"),
            median_dist_m  = ("_loc_dist", "median"),
        ).reset_index()
        bs["bin_mid_pop"]     = bs["pop_bin"].apply(
            lambda iv: (iv.left + iv.right) / 2
        ).astype(float)
        bs["log_bin_mid_pop"] = np.log(bs["bin_mid_pop"])
        bs["dataset"]         = ds_name
        bin_records.append(bs)

        valid_A = bs[bs["total_poi"] >= min_count].dropna(
            subset=["log_bin_mid_pop", "median_dist_m"]
        ).reset_index(drop=True)

        base = dict(dataset=ds_name, n_used=len(valid_A))
        if len(valid_A) >= 3:
            r_ols, r_wls = _fit_both_loc(
                "median_dist_m ~ log_bin_mid_pop",
                valid_A, "total_poi", "log_bin_mid_pop",
            )
        else:
            r_ols = r_wls = dict(
                slope=np.nan, intercept=np.nan,
                r_squared=np.nan, p_value=np.nan, std_err=np.nan,
            )
        rec_A_ols.append({**base, "method": "A_OLS_bin", **r_ols})
        rec_A_wls.append({**base, "method": "A_WLS_bin", **r_wls})

        # ── Method C ─────────────────────────────────────────────────────
        tract_c = gdf_pop.groupby("GEOID").agg(
            total_poi      = ("_loc_dist", "size"),
            n_true_matched = ("_loc_dist", "count"),
            median_dist_m  = ("_loc_dist", "median"),
        ).reset_index()
        tract_c = tract_c.merge(
            tract_gdf[["GEOID", "pop_density"]], on="GEOID", how="left"
        )
        tract_c = tract_c[
            (tract_c["total_poi"] >= min_count) & (tract_c["pop_density"] > 0)
        ].dropna(subset=["pop_density", "median_dist_m"]).reset_index(drop=True)
        tract_c["log_pop_density"] = np.log(tract_c["pop_density"])

        base_c = dict(dataset=ds_name, n_used=len(tract_c))
        if len(tract_c) >= 3:
            r_ols, r_wls = _fit_both_loc(
                "median_dist_m ~ log_pop_density",
                tract_c, "total_poi", "log_pop_density",
            )
        else:
            r_ols = r_wls = dict(
                slope=np.nan, intercept=np.nan,
                r_squared=np.nan, p_value=np.nan, std_err=np.nan,
            )
        rec_C_ols.append({**base_c, "method": "C_OLS_tract", **r_ols})
        rec_C_wls.append({**base_c, "method": "C_WLS_tract", **r_wls})

    bin_df = pd.concat(bin_records, ignore_index=True) if bin_records else pd.DataFrame()
    return (
        bin_df,
        pd.DataFrame(rec_A_ols), pd.DataFrame(rec_A_wls),
        pd.DataFrame(rec_C_ols), pd.DataFrame(rec_C_wls),
    )


# ==============================================================================
# MASTER PIPELINE
# ==============================================================================

def run_lc_analysis(
    datasets:        dict,
    center:          Point,
    cbsa_code:       str,
    state_fips_list: list,
    api_key:         str,
    n_bins:          int = 50,
    min_count:       int = 20,
    census_year:     int = 2020,
) -> dict:
    """
    Run all four location difference modules for a given MSA.

    Parameters
    ----------
    datasets        : {'ove': gdf, 'sf': gdf, 'fsq': gdf, 'osm': gdf}
                      Each GDF must already have a msa_name column.
    center          : shapely Point(lon, lat) in EPSG:4326
    cbsa_code       : OMB CBSA code, e.g. '16980'
    state_fips_list : state FIPS codes, e.g. ['17'] or ['36','34','09']
    api_key         : Census API key
    n_bins          : number of bins for regressions
    min_count       : minimum total_poi per bin/tract
    census_year     : ACS vintage year

    Returns
    -------
    dict with keys:
        overall_loc, by_category_loc,
        distance_bins_loc,
        distance_loc_reg_A_ols, distance_loc_reg_A_wls,
        distance_loc_reg_C_ols, distance_loc_reg_C_wls,
        pop_bins_loc,
        pop_loc_reg_A_ols, pop_loc_reg_A_wls,
        pop_loc_reg_C_ols, pop_loc_reg_C_wls,
        tract_gdf
    """
    print("[Module 1] Overall location difference")
    overall_loc = compute_overall_location_diff(datasets)
    print(overall_loc.to_string(index=False))

    print("\n[Module 2] Location difference by category")
    by_cat_loc = compute_location_diff_by_category(datasets, min_count=min_count)
    print(f"  -> {len(by_cat_loc)} rows")

    print("\n[Fetching Census tracts]")
    tract_gdf = fetch_msa_tracts(
        cbsa_code, state_fips_list, api_key, year=census_year
    )

    print("\n[Module 3] Location difference ~ distance")
    dist_bins, dist_A_ols, dist_A_wls, dist_C_ols, dist_C_wls = \
        compute_location_dist_regression(
            datasets, center, tract_gdf, n_bins, min_count
        )
    for tag, df in [("A OLS", dist_A_ols), ("A WLS", dist_A_wls),
                    ("C OLS", dist_C_ols), ("C WLS", dist_C_wls)]:
        print(f"  Method {tag}:")
        print(df[["dataset", "slope", "r_squared", "p_value", "n_used"]
                 ].to_string(index=False))

    print("\n[Module 4] Location difference ~ population density")
    pop_bins, pop_A_ols, pop_A_wls, pop_C_ols, pop_C_wls = \
        compute_location_pop_regression(datasets, tract_gdf, n_bins, min_count)
    for tag, df in [("A OLS", pop_A_ols), ("A WLS", pop_A_wls),
                    ("C OLS", pop_C_ols), ("C WLS", pop_C_wls)]:
        print(f"  Method {tag}:")
        print(df[["dataset", "slope", "r_squared", "p_value", "n_used"]
                 ].to_string(index=False))

    return {
        "overall_loc":             overall_loc,
        "by_category_loc":         by_cat_loc,
        "distance_bins_loc":       dist_bins,
        "distance_loc_reg_A_ols":  dist_A_ols,
        "distance_loc_reg_A_wls":  dist_A_wls,
        "distance_loc_reg_C_ols":  dist_C_ols,
        "distance_loc_reg_C_wls":  dist_C_wls,
        "pop_bins_loc":            pop_bins,
        "pop_loc_reg_A_ols":       pop_A_ols,
        "pop_loc_reg_A_wls":       pop_A_wls,
        "pop_loc_reg_C_ols":       pop_C_ols,
        "pop_loc_reg_C_wls":       pop_C_wls,
        "tract_gdf":               tract_gdf,
    }
