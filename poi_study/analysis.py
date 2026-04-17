"""
POI Matching Analysis Pipeline — Multi-MSA Merged Version
==========================================================
datasets dict: {'ove': gplc_ove, 'sf': gplc_sf, 'fsq': gplc_fsq, 'osm': gplc_osm}
Each GDF already contains a msa_name column — no msa_name parameter needed.

Four modules:
    1. Overall Match Rate
    2. Match Rate by Category (primary_cat)
    3. Match Rate by Distance  + Methods A / C  (each OLS + WLS)
    4. Match Rate by Pop Density + Methods A / C (each OLS + WLS)

Returns:
    overall, by_category
    distance_bins
    distance_reg_A_ols, distance_reg_A_wls
    distance_reg_C_ols, distance_reg_C_wls
    pop_bins
    pop_reg_A_ols, pop_reg_A_wls
    pop_reg_C_ols, pop_reg_C_wls
    tract_gdf
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
from statsmodels.formula.api import ols, wls


# ==============================================================================
# HELPER
# ==============================================================================

def _parse_true_match(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x: pd.NA  if pd.isna(x) else
                  True   if str(x).strip() in ("1", "1.0", "True",  "true")  else
                  False  if str(x).strip() in ("0", "0.0", "False", "false") else
                  pd.NA
    ).astype("boolean")


def _fit_both(formula: str, data: pd.DataFrame, weight_col: str,
              x_col: str) -> tuple[dict, dict]:
    """Fit OLS and WLS for the same formula, return (ols_rec, wls_rec)."""
    results = {}
    for method_tag, model_fn in [("OLS", lambda: ols(formula, data=data).fit()),
                                   ("WLS", lambda: wls(formula, data=data,
                                                         weights=data[weight_col]).fit())]:
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
            results[method_tag] = dict(slope=np.nan, intercept=np.nan,
                                        r_squared=np.nan, p_value=np.nan, std_err=np.nan)
    return results["OLS"], results["WLS"]


# ==============================================================================
# MODULE 1 — Overall Match Rate
# ==============================================================================

def compute_overall_match_rate(datasets: dict) -> pd.DataFrame:
    records = []
    for ds_name, gdf in datasets.items():
        parsed  = _parse_true_match(gdf["is_true_match"])
        total   = len(parsed)
        matched = int(parsed.sum(skipna=True))
        records.append({
            "dataset":       ds_name,
            "total_poi":     total,
            "matched_count": matched,
            "match_rate":    matched / total if total > 0 else np.nan,
        })
    return pd.DataFrame(records)


# ==============================================================================
# MODULE 2 — Match Rate by Category
# ==============================================================================

def compute_match_rate_by_category(
    datasets: dict, min_count: int = 10
) -> pd.DataFrame:
    records = []
    for ds_name, gdf in datasets.items():
        gdf = gdf.copy()
        gdf["_match"] = _parse_true_match(gdf["is_true_match"])
        grp = (
            gdf.groupby("primary_cat")["_match"]
            .agg(total_poi="size",
                 matched_count=lambda x: x.eq(True).sum())
            .reset_index()
        )
        grp = grp[grp["total_poi"] >= min_count].copy()
        grp["match_rate"] = grp["matched_count"] / grp["total_poi"]
        grp["dataset"]    = ds_name
        records.append(grp[["dataset", "primary_cat",
                             "total_poi", "matched_count", "match_rate"]])
    return pd.concat(records, ignore_index=True)


# ==============================================================================
# MODULE 3 — Distance Regression (Methods A / C, each OLS + WLS)
# ==============================================================================

def _add_distance_bins(
    gdf: gpd.GeoDataFrame, center: Point,
    n_bins: int = 20, src_crs: str = "EPSG:4326", proj_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    gdf      = gdf.copy()
    pt_gdf   = gpd.GeoDataFrame(geometry=[center], crs=src_crs).to_crs(proj_crs)
    gdf_proj = gdf.to_crs(proj_crs)
    gdf["dist_to_center"] = gdf_proj.geometry.distance(pt_gdf.geometry.iloc[0])
    dmin, dmax = gdf["dist_to_center"].min(), gdf["dist_to_center"].max()
    gdf["dist_bin"] = pd.cut(gdf["dist_to_center"],
                              bins=np.linspace(dmin, dmax, n_bins + 1),
                              include_lowest=True)
    return gdf


def compute_distance_regression(
    datasets: dict, center: Point, tract_gdf: gpd.GeoDataFrame,
    n_bins: int = 20, min_count: int = 10,
) -> tuple:
    """
    Method A: bin aggregation  — bin_match_rate ~ bin_mid_km
    Method C: tract level      — tract_match_rate ~ dist_to_center_km
    Each method runs both OLS and WLS (weight = n_poi).
    Returns: bin_df, reg_A_ols, reg_A_wls, reg_C_ols, reg_C_wls
    """
    bin_records = []
    rec_A_ols, rec_A_wls = [], []
    rec_C_ols, rec_C_wls = [], []

    tract_proj = tract_gdf.to_crs("EPSG:3857").copy()
    pt_gdf     = gpd.GeoDataFrame(geometry=[center], crs="EPSG:4326").to_crs("EPSG:3857")
    tract_proj["dist_to_center_km"] = (
        tract_proj.geometry.centroid.distance(pt_gdf.geometry.iloc[0]) / 1000
    )
    tract_meta = tract_proj[["GEOID", "dist_to_center_km"]].copy()

    for ds_name, gdf in datasets.items():

        # ── Method A ─────────────────────────────────────────────────────────
        gdf_b = _add_distance_bins(gdf, center, n_bins=n_bins)
        gdf_b["_match"] = _parse_true_match(gdf_b["is_true_match"]).astype(float)

        bs = (
            gdf_b.groupby("dist_bin", observed=True)["_match"]
            .agg(total_poi="count", matched_count="sum")
            .reset_index()
        )
        bs["match_rate"] = bs["matched_count"] / bs["total_poi"]
        bs["bin_mid_km"] = bs["dist_bin"].apply(
            lambda iv: (iv.left + iv.right) / 2 / 1000
        ).astype(float)
        bs["dataset"] = ds_name
        bin_records.append(bs)

        valid_A = bs[bs["total_poi"] >= min_count].dropna(
            subset=["bin_mid_km", "match_rate"]
        ).reset_index(drop=True)

        base = dict(dataset=ds_name, n_used=len(valid_A))
        if len(valid_A) >= 3:
            r_ols, r_wls = _fit_both(
                "match_rate ~ bin_mid_km", valid_A, "total_poi", "bin_mid_km"
            )
        else:
            r_ols = r_wls = dict(slope=np.nan, intercept=np.nan,
                                   r_squared=np.nan, p_value=np.nan, std_err=np.nan)
        rec_A_ols.append({**base, "method": "A_OLS_bin", **r_ols})
        rec_A_wls.append({**base, "method": "A_WLS_bin", **r_wls})

        # ── Method C ─────────────────────────────────────────────────────────
        gdf_c = _join_pop_to_poi(gdf, tract_gdf)
        gdf_c["_match"] = _parse_true_match(gdf_c["is_true_match"]).astype(float)
        gdf_c = gdf_c.dropna(subset=["GEOID", "_match"])

        tract_c = (
            gdf_c.groupby("GEOID")["_match"]
            .agg(n_poi="count", matched_count="sum")
            .reset_index()
        )
        tract_c["match_rate"] = tract_c["matched_count"] / tract_c["n_poi"]
        tract_c = tract_c.merge(tract_meta, on="GEOID", how="left")
        tract_c = tract_c[tract_c["n_poi"] >= min_count].dropna(
            subset=["dist_to_center_km", "match_rate"]
        ).reset_index(drop=True)

        base_c = dict(dataset=ds_name, n_used=len(tract_c))
        if len(tract_c) >= 3:
            r_ols, r_wls = _fit_both(
                "match_rate ~ dist_to_center_km", tract_c, "n_poi", "dist_to_center_km"
            )
        else:
            r_ols = r_wls = dict(slope=np.nan, intercept=np.nan,
                                   r_squared=np.nan, p_value=np.nan, std_err=np.nan)
        rec_C_ols.append({**base_c, "method": "C_OLS_tract", **r_ols})
        rec_C_wls.append({**base_c, "method": "C_WLS_tract", **r_wls})

    bin_df = pd.concat(bin_records, ignore_index=True)
    return (bin_df,
            pd.DataFrame(rec_A_ols), pd.DataFrame(rec_A_wls),
            pd.DataFrame(rec_C_ols), pd.DataFrame(rec_C_wls))


# ==============================================================================
# MODULE 4 — Population Density Regression (Methods A / C, each OLS + WLS)
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
        counties = (filtered["FIPS State Code"] + filtered["FIPS County Code"]).tolist()
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
            if resp.status_code != 200: continue
            data = resp.json()
            tract_records.append(pd.DataFrame(data[1:], columns=data[0]))

    tract_df = pd.concat(tract_records, ignore_index=True)
    tract_df["GEOID"]      = tract_df["state"] + tract_df["county"] + tract_df["tract"]
    tract_df["population"] = pd.to_numeric(tract_df["B01003_001E"], errors="coerce")

    geom_frames = []
    for state_fips in state_fips_list:
        gdf_s = gpd.read_file(
            f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
        )
        geom_frames.append(gdf_s[["GEOID", "geometry"]])

    all_county_fips = [c for cs in counties_by_state.values() for c in cs]
    merged = pd.concat(geom_frames, ignore_index=True).merge(
        tract_df[["GEOID", "population"]], on="GEOID", how="inner"
    )
    merged = merged[merged["GEOID"].str[:5].isin(all_county_fips)].copy()

    merged_proj           = merged.to_crs("EPSG:3857")
    merged["area_km2"]    = merged_proj.geometry.area / 1e6
    merged["pop_density"] = (merged["population"] / merged["area_km2"]
                              ).replace([np.inf, -np.inf], np.nan)
    return merged[["GEOID", "population", "area_km2", "pop_density", "geometry"]]


def _join_pop_to_poi(gdf: gpd.GeoDataFrame, tract_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        gdf.to_crs("EPSG:3857"),
        tract_gdf[["GEOID", "population", "area_km2", "pop_density", "geometry"
                   ]].to_crs("EPSG:3857"),
        how="left", predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return joined.to_crs("EPSG:4326")


def compute_population_regression(
    datasets: dict, tract_gdf: gpd.GeoDataFrame,
    n_bins: int = 20, min_count: int = 10,
) -> tuple:
    """
    Method A: bin aggregation  — bin_match_rate ~ log(bin_mid_pop)
    Method C: tract level      — tract_match_rate ~ log(pop_density)
    Each method runs both OLS and WLS (weight = n_poi).
    Returns: bin_df, reg_A_ols, reg_A_wls, reg_C_ols, reg_C_wls
    """
    bin_records = []
    rec_A_ols, rec_A_wls = [], []
    rec_C_ols, rec_C_wls = [], []

    for ds_name, gdf in datasets.items():

        gdf_pop = _join_pop_to_poi(gdf, tract_gdf)
        gdf_pop["_match"] = _parse_true_match(gdf_pop["is_true_match"]).astype(float)
        valid_mask = gdf_pop["pop_density"].notna()

        # ── Method A ─────────────────────────────────────────────────────────
        bins = np.unique(np.percentile(
            gdf_pop.loc[valid_mask, "pop_density"],
            np.linspace(0, 100, n_bins + 1)
        ))
        gdf_pop["pop_bin"] = pd.cut(gdf_pop["pop_density"], bins=bins, include_lowest=True)

        bs = (
            gdf_pop.groupby("pop_bin", observed=True)["_match"]
            .agg(total_poi="count", matched_count="sum")
            .reset_index()
        )
        bs["match_rate"]      = bs["matched_count"] / bs["total_poi"]
        bs["bin_mid_pop"]     = bs["pop_bin"].apply(
            lambda iv: (iv.left + iv.right) / 2
        ).astype(float)
        bs["log_bin_mid_pop"] = np.log(bs["bin_mid_pop"])
        bs["dataset"]         = ds_name
        bin_records.append(bs)

        valid_A = bs[bs["total_poi"] >= min_count].dropna(
            subset=["log_bin_mid_pop", "match_rate"]
        ).reset_index(drop=True)

        base = dict(dataset=ds_name, n_used=len(valid_A))
        if len(valid_A) >= 3:
            r_ols, r_wls = _fit_both(
                "match_rate ~ log_bin_mid_pop", valid_A, "total_poi", "log_bin_mid_pop"
            )
        else:
            r_ols = r_wls = dict(slope=np.nan, intercept=np.nan,
                                   r_squared=np.nan, p_value=np.nan, std_err=np.nan)
        rec_A_ols.append({**base, "method": "A_OLS_bin", **r_ols})
        rec_A_wls.append({**base, "method": "A_WLS_bin", **r_wls})

        # ── Method C ─────────────────────────────────────────────────────────
        gdf_c = gdf_pop.dropna(subset=["GEOID", "_match"]).copy()
        tract_c = (
            gdf_c.groupby("GEOID")["_match"]
            .agg(n_poi="count", matched_count="sum")
            .reset_index()
        )
        tract_c["match_rate"] = tract_c["matched_count"] / tract_c["n_poi"]
        tract_c = tract_c.merge(
            tract_gdf[["GEOID", "pop_density"]], on="GEOID", how="left"
        )
        tract_c = tract_c[
            (tract_c["n_poi"] >= min_count) & (tract_c["pop_density"] > 0)
        ].dropna(subset=["pop_density", "match_rate"]).reset_index(drop=True)
        tract_c["log_pop_density"] = np.log(tract_c["pop_density"])

        base_c = dict(dataset=ds_name, n_used=len(tract_c))
        if len(tract_c) >= 3:
            r_ols, r_wls = _fit_both(
                "match_rate ~ log_pop_density", tract_c, "n_poi", "log_pop_density"
            )
        else:
            r_ols = r_wls = dict(slope=np.nan, intercept=np.nan,
                                   r_squared=np.nan, p_value=np.nan, std_err=np.nan)
        rec_C_ols.append({**base_c, "method": "C_OLS_tract", **r_ols})
        rec_C_wls.append({**base_c, "method": "C_WLS_tract", **r_wls})

    bin_df = pd.concat(bin_records, ignore_index=True) if bin_records else pd.DataFrame()
    return (bin_df,
            pd.DataFrame(rec_A_ols), pd.DataFrame(rec_A_wls),
            pd.DataFrame(rec_C_ols), pd.DataFrame(rec_C_wls))


# ==============================================================================
# MASTER PIPELINE
# ==============================================================================

def run_mr_analysis(
    datasets:        dict,
    center:          Point,
    cbsa_code:       str,
    state_fips_list: list,
    api_key:         str,
    n_bins:          int = 20,
    min_count:       int = 10,
    census_year:     int = 2020,
) -> dict:
    """
    Parameters
    ----------
    datasets : {'ove': gplc_ove, 'sf': gplc_sf, 'fsq': gplc_fsq, 'osm': gplc_osm}
               Each GDF must have a msa_name column already.
    """
    print("[Module 1] Overall match rate")
    overall = compute_overall_match_rate(datasets)
    print(overall.to_string(index=False))

    print("\n[Module 2] Match rate by category")
    by_cat = compute_match_rate_by_category(datasets, min_count=min_count)
    print(f"  -> {len(by_cat)} rows")

    print("\n[Fetching Census tracts]")
    tract_gdf = fetch_msa_tracts(cbsa_code, state_fips_list, api_key, year=census_year)

    print("\n[Module 3] Distance regression")
    dist_bins, dist_A_ols, dist_A_wls, dist_C_ols, dist_C_wls = \
        compute_distance_regression(datasets, center, tract_gdf, n_bins, min_count)
    for tag, df in [("A OLS", dist_A_ols), ("A WLS", dist_A_wls),
                     ("C OLS", dist_C_ols), ("C WLS", dist_C_wls)]:
        print(f"  Method {tag}:")
        print(df[["dataset", "slope", "r_squared", "p_value", "n_used"]].to_string(index=False))

    print("\n[Module 4] Population density regression")
    pop_bins, pop_A_ols, pop_A_wls, pop_C_ols, pop_C_wls = \
        compute_population_regression(datasets, tract_gdf, n_bins, min_count)
    for tag, df in [("A OLS", pop_A_ols), ("A WLS", pop_A_wls),
                     ("C OLS", pop_C_ols), ("C WLS", pop_C_wls)]:
        print(f"  Method {tag}:")
        print(df[["dataset", "slope", "r_squared", "p_value", "n_used"]].to_string(index=False))

    return {
        "overall":        overall,
        "by_category":    by_cat,
        "distance_bins":  dist_bins,
        "distance_reg_A_ols": dist_A_ols,
        "distance_reg_A_wls": dist_A_wls,
        "distance_reg_C_ols": dist_C_ols,
        "distance_reg_C_wls": dist_C_wls,
        "pop_bins":       pop_bins,
        "pop_reg_A_ols":  pop_A_ols,
        "pop_reg_A_wls":  pop_A_wls,
        "pop_reg_C_ols":  pop_C_ols,
        "pop_reg_C_wls":  pop_C_wls,
        "tract_gdf":      tract_gdf,
    }