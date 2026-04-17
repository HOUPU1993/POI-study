import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import geopandas as gpd
from .analysis import _parse_true_match


CITY_COLORS = {
    "chicago":  "#4C72B0",
    "new_york": "#DD8452",
    "la":       "#55A868",
    "bspo":     "#C44E52",
}
DS_ORDER  = ["ove", "sf", "fsq", "osm"]
DS_TITLES = {"ove": "OVE", "sf": "SF", "fsq": "FSQ", "osm": "OSM"}


def _fit_line_pooled(x, y, w=None):
    """Fit a line on pooled data, return (x_line, y_line)."""
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


def _draw_pooled_panel(ax, pooled_df, x_col, weight_col, x_label, log_x=False):
    """
    pooled_df: DataFrame with columns [x_col, match_rate, weight_col, city]
    Scatter colored by city; one OLS + one WLS line fitted on all points.
    """
    df = pooled_df.dropna(subset=[x_col, "match_rate", weight_col]).copy()
    if df.empty:
        return

    x_raw = np.log(df[x_col].values) if log_x else df[x_col].values
    y     = df["match_rate"].values
    w     = df[weight_col].values
    df["_x"] = x_raw

    # scatter — colored by city
    for city, grp in df.groupby("city"):
        ax.scatter(grp["_x"], grp["match_rate"],
                   color=CITY_COLORS.get(city, "gray"),
                   alpha=0.35, s=14, zorder=2, label=city)

    # OLS line — fitted on all pooled points (solid black)
    xl, yl = _fit_line_pooled(x_raw, y)
    if len(xl):
        ax.plot(xl, yl, color="black", linewidth=2, linestyle="-",
                label="OLS", zorder=4)

    # WLS line — weighted by n_poi (dashed black)
    xl, yl = _fit_line_pooled(x_raw, y, w=w)
    if len(xl):
        ax.plot(xl, yl, color="black", linewidth=2, linestyle="--",
                label="WLS", zorder=4)

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("match rate", fontsize=9)
    ax.set_ylim(0, 1)
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
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CITY_COLORS) + 2, fontsize=8.5,
               framealpha=0.9, edgecolor="#ccc")


# ==============================================================================
# Method A — pooled bin-level plots
# ==============================================================================

def plot_method_A_pooled(all_results: dict, save_prefix: str = "method_A_pooled"):
    """
    Pooled version of Method A plots.
    Concatenates bin data from all cities; fits one OLS + one WLS on the pool.
    Produces 2 figures: distance and population.
    """
    for fig_tag, x_col, weight_col, x_label, log_x in [
        ("distance", "bin_mid_km",  "total_poi", "distance from center (km)", False),
        ("pop",      "bin_mid_pop", "total_poi", "log population density",    True),
    ]:
        bin_key = "distance_bins" if fig_tag == "distance" else "pop_bins"

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"Method A (bin, pooled) — match rate vs {fig_tag}\n"
            f"OLS (solid) vs WLS (dashed), all cities combined",
            fontsize=12, y=1.01
        )

        for ax, ds in zip(axes.flat, DS_ORDER):
            # concat all cities for this dataset
            frames = []
            for city, res in all_results.items():
                df = res[bin_key].copy()
                df = df[df["dataset"] == ds]
                df["city"] = city
                frames.append(df)
            pooled = pd.concat(frames, ignore_index=True)

            _draw_pooled_panel(ax, pooled, x_col, weight_col, x_label, log_x=log_x)
            ax.set_title(DS_TITLES[ds], fontsize=10, fontweight="bold")

        _make_pooled_legend(fig)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()


# ==============================================================================
# Method C — pooled tract-level plots
# ==============================================================================

def plot_method_C_pooled(all_results: dict, save_prefix: str = "method_C_pooled"):
    """
    Pooled version of Method C plots.
    Rebuilds tract-level data from all_results (tract_gdf already stored).
    Produces 2 figures: distance and population.
    """
    for fig_tag, x_label, log_x in [
        ("distance", "distance from center (km)", False),
        ("pop",      "log population density",    True),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"Method C (tract, pooled) — match rate vs {fig_tag}\n"
            f"OLS (solid) vs WLS (dashed), all cities combined",
            fontsize=12, y=1.01
        )

        for ax, ds in zip(axes.flat, DS_ORDER):
            frames = []
            for city, res in all_results.items():
                # pull pre-computed reg tables to get tract-level x values
                reg_key = "distance_reg_C_ols" if fig_tag == "distance" \
                           else "pop_reg_C_ols"

                # rebuild tract match_rate from tract_gdf stored in results
                tract_gdf = res["tract_gdf"]
                gdf = MSA_CONFIGS[city]["datasets"][ds]

                gdf_j = gpd.sjoin(
                    gdf.to_crs("EPSG:3857"),
                    tract_gdf[["GEOID", "pop_density", "geometry"]
                               ].to_crs("EPSG:3857"),
                    how="left", predicate="within"
                ).drop(columns=["index_right"], errors="ignore").to_crs("EPSG:4326")

                gdf_j["_match"] = _parse_true_match(
                    gdf_j["is_true_match"]).astype(float)
                tc = (
                    gdf_j.groupby("GEOID")["_match"]
                    .agg(n_poi="count", matched="sum")
                    .reset_index()
                )
                tc["match_rate"] = tc["matched"] / tc["n_poi"]

                if fig_tag == "distance":
                    center = MSA_CONFIGS[city]["center"]
                    tp = tract_gdf.to_crs("EPSG:3857").copy()
                    pt = gpd.GeoDataFrame(
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

                tc["city"]    = city
                tc["total_poi"] = tc["n_poi"]
                frames.append(tc[["x_val", "match_rate", "total_poi", "city"]].dropna())

            pooled = pd.concat(frames, ignore_index=True)
            _draw_pooled_panel(ax, pooled, "x_val", "total_poi", x_label, log_x=log_x)
            ax.set_title(DS_TITLES[ds], fontsize=10, fontweight="bold")

        _make_pooled_legend(fig)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()