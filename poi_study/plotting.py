"""
POI Analysis — Unified Plotting Functions
==========================================
Pooled diagnostic plots for both match rate and location distance analysis.
All cities are pooled into one scatter cloud; one OLS + one WLS line is fitted.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import geopandas as gpd

from .mr_analysis import _parse_true_match
from .lc_analysis import _get_true_match_location_distance

# ==============================================================================
# CONSTANTS
# ==============================================================================

CITY_COLORS = {
    "chicago":  "#4C72B0",
    "new_york": "#DD8452",
    "la":       "#55A868",
    "bspo":     "#C44E52",
}

DS_ORDER  = ["ove", "sf", "fsq", "osm"]
DS_TITLES = {"ove": "OVE", "sf": "SF", "fsq": "FSQ", "osm": "OSM"}


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

def _fit_line_pooled(x, y, w=None):
    """Fit a line on pooled data. Returns (x_line, y_line)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        mask &= np.isfinite(w)
    x_c, y_c = x[mask], y[mask]
    w_c = w[mask] if w is not None else None
    if len(x_c) < 3:
        return np.array([]), np.array([])
    coeffs = np.polyfit(x_c, y_c, 1, w=w_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 300)
    return x_line, np.polyval(coeffs, x_line)


def _draw_pooled_panel(ax, pooled_df, x_col, y_col, weight_col,
                       x_label, y_label, log_x=False):
    """
    Draw scatter (colored by city) + one OLS line + one WLS line.

    Parameters
    ----------
    pooled_df  : DataFrame with columns [x_col, y_col, weight_col, city]
    log_x      : if True, apply log transform to x before plotting/fitting
    """
    df = pooled_df.dropna(subset=[x_col, y_col, weight_col]).copy()
    if df.empty:
        return

    x_raw    = np.log(df[x_col].values) if log_x else df[x_col].values
    y        = df[y_col].values
    w        = df[weight_col].values
    df["_x"] = x_raw

    for city, grp in df.groupby("city"):
        ax.scatter(
            grp["_x"], grp[y_col],
            color=CITY_COLORS.get(city, "gray"),
            alpha=0.35, s=14, zorder=2,
        )

    xl, yl = _fit_line_pooled(x_raw, y)
    if len(xl):
        ax.plot(xl, yl, color="black", linewidth=2,
                linestyle="-", label="OLS", zorder=4)

    xl, yl = _fit_line_pooled(x_raw, y, w=w)
    if len(xl):
        ax.plot(xl, yl, color="black", linewidth=2,
                linestyle="--", label="WLS", zorder=4)

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.yaxis.grid(True, linewidth=0.3, color="#ddd")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _make_pooled_legend(fig):
    handles = [
        mlines.Line2D([], [], color=c, linewidth=0,
                      marker="o", markersize=6, label=city, alpha=0.7)
        for city, c in CITY_COLORS.items()
    ] + [
        mlines.Line2D([], [], color="black", linewidth=2,
                      linestyle="-",  label="OLS (pooled)"),
        mlines.Line2D([], [], color="black", linewidth=2,
                      linestyle="--", label="WLS (pooled)"),
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=len(CITY_COLORS) + 2, fontsize=8.5,
        framealpha=0.9, edgecolor="#ccc",
    )


# ==============================================================================
# METHOD A — Pooled bin-level plots
# ==============================================================================

def plot_method_A_pooled(
    all_results:  dict,
    y_col:        str,
    bin_key_dist: str,
    bin_key_pop:  str,
    y_label:      str = None,
    save_prefix:  str = "method_A_pooled",
) -> None:
    """
    Pooled Method A diagnostic plots (bin level).
    Concatenates bin data from all cities; fits one OLS + one WLS on the pool.

    Produces 2 figures:
        {save_prefix}_distance.png  — y_col vs bin_mid_km
        {save_prefix}_pop.png       — y_col vs log(bin_mid_pop)

    Each figure: 2x2 subplots, one per dataset. Scatter colored by city.

    Parameters
    ----------
    all_results  : dict  output of run_mr_analysis() or run_lc_analysis()
    y_col        : str   'match_rate' or 'median_dist_m'
    bin_key_dist : str   'distance_bins' or 'distance_bins_loc'
    bin_key_pop  : str   'pop_bins' or 'pop_bins_loc'
    y_label      : str   y-axis label (defaults to y_col)
    save_prefix  : str   filename prefix

    Examples
    --------
    plot_method_A_pooled(
        all_results,
        y_col        = "match_rate",
        bin_key_dist = "distance_bins",
        bin_key_pop  = "pop_bins",
        y_label      = "match rate",
        save_prefix  = "mr_A",
    )

    plot_method_A_pooled(
        all_loc_results,
        y_col        = "median_dist_m",
        bin_key_dist = "distance_bins_loc",
        bin_key_pop  = "pop_bins_loc",
        y_label      = "median location distance (m)",
        save_prefix  = "lc_A",
    )
    """
    y_label = y_label or y_col

    for fig_tag, x_col, weight_col, x_label, log_x, bin_key in [
        ("distance", "bin_mid_km",  "total_poi",
         "distance from center (km)", False, bin_key_dist),
        ("pop",      "bin_mid_pop", "total_poi",
         "log population density",    True,  bin_key_pop),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"Method A (bin, pooled) — {y_label} vs {fig_tag}\n"
            f"OLS (solid) vs WLS (dashed), all cities combined",
            fontsize=12, y=1.01,
        )

        for ax, ds in zip(axes.flat, DS_ORDER):
            frames = []
            for city, res in all_results.items():
                df         = res[bin_key].copy()
                df         = df[df["dataset"] == ds]
                df["city"] = city
                frames.append(df)
            pooled = pd.concat(frames, ignore_index=True)

            _draw_pooled_panel(
                ax, pooled, x_col, y_col, weight_col,
                x_label, y_label, log_x=log_x,
            )
            ax.set_title(DS_TITLES[ds], fontsize=10, fontweight="bold")

        _make_pooled_legend(fig)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(f"{save_prefix}_{fig_tag}.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_prefix}_{fig_tag}.png")


# ==============================================================================
# METHOD C — Pooled tract-level plots
# ==============================================================================

def plot_method_C_pooled(
    all_results:  dict,
    msa_configs:  dict,
    y_col:        str,
    y_label:      str = None,
    save_prefix:  str = "method_C_pooled",
) -> None:
    """
    Pooled Method C diagnostic plots (tract level).
    Rebuilds tract-level data from tract_gdf stored in all_results.

    Produces 2 figures:
        {save_prefix}_distance.png  — y_col vs dist_to_center_km
        {save_prefix}_pop.png       — y_col vs log(pop_density)

    Each figure: 2x2 subplots, one per dataset. Scatter colored by city.

    Parameters
    ----------
    all_results  : dict  output of run_mr_analysis() or run_lc_analysis()
    msa_configs  : dict  MSA_CONFIGS dict (needs 'center' and 'datasets' per city)
    y_col        : str   'match_rate' or 'median_dist_m'
    y_label      : str   y-axis label (defaults to y_col)
    save_prefix  : str   filename prefix

    Examples
    --------
    plot_method_C_pooled(
        all_results, MSA_CONFIGS,
        y_col       = "match_rate",
        y_label     = "match rate",
        save_prefix = "mr_C",
    )

    plot_method_C_pooled(
        all_loc_results, MSA_CONFIGS,
        y_col       = "median_dist_m",
        y_label     = "median location distance (m)",
        save_prefix = "lc_C",
    )
    """
    y_label = y_label or y_col
    is_loc  = (y_col == "median_dist_m")

    for fig_tag, x_label, log_x in [
        ("distance", "distance from center (km)", False),
        ("pop",      "log population density",    True),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"Method C (tract, pooled) — {y_label} vs {fig_tag}\n"
            f"OLS (solid) vs WLS (dashed), all cities combined",
            fontsize=12, y=1.01,
        )

        for ax, ds in zip(axes.flat, DS_ORDER):
            frames = []
            for city, res in all_results.items():
                tract_gdf = res["tract_gdf"]
                gdf       = msa_configs[city]["datasets"][ds]

                gdf_j = gpd.sjoin(
                    gdf.to_crs("EPSG:3857"),
                    tract_gdf[["GEOID", "pop_density", "geometry"]
                               ].to_crs("EPSG:3857"),
                    how="left", predicate="within",
                ).drop(columns=["index_right"], errors="ignore").to_crs("EPSG:4326")

                # compute y value per POI
                if is_loc:
                    gdf_j["_y"] = _get_true_match_location_distance(gdf_j)
                else:
                    gdf_j["_y"] = _parse_true_match(
                        gdf_j["is_true_match"]
                    ).astype(float)

                agg_func = "median" if is_loc else "mean"
                tc = gdf_j.groupby("GEOID").agg(
                    total_poi = ("_y", "size"),
                    _y_agg    = ("_y", agg_func),
                ).reset_index()
                tc = tc.rename(columns={"_y_agg": y_col})

                if fig_tag == "distance":
                    center = msa_configs[city]["center"]
                    tp     = tract_gdf.to_crs("EPSG:3857").copy()
                    pt     = gpd.GeoDataFrame(
                        geometry=[center], crs="EPSG:4326"
                    ).to_crs("EPSG:3857")
                    tp["x_val"] = (
                        tp.geometry.centroid.distance(pt.geometry.iloc[0]) / 1000
                    )
                    tc = tc.merge(tp[["GEOID", "x_val"]], on="GEOID", how="left")
                else:
                    tc = tc.merge(
                        tract_gdf[["GEOID", "pop_density"]], on="GEOID", how="left"
                    )
                    tc = tc[tc["pop_density"] > 0].copy()
                    tc = tc.rename(columns={"pop_density": "x_val"})

                tc["city"] = city
                frames.append(
                    tc[["x_val", y_col, "total_poi", "city"]].dropna()
                )

            pooled = pd.concat(frames, ignore_index=True)
            _draw_pooled_panel(
                ax, pooled, "x_val", y_col, "total_poi",
                x_label, y_label, log_x=log_x,
            )
            ax.set_title(DS_TITLES[ds], fontsize=10, fontweight="bold")

        _make_pooled_legend(fig)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(f"{save_prefix}_{fig_tag}.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_prefix}_{fig_tag}.png")