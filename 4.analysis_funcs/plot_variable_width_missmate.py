import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_variable_width_missmate(
    df,
    c_cols=("match_c_ove", "match_c_sf", "match_c_fsq", "match_c_osm"),
    den_cols=("total_den_google", "total_den_google", "total_den_google", "total_den_google"),
    primary_cat_col="primary_cat",
    n_cols=4,
    title="Category-level POI Match Rate Comparison relative to Google Places\n(Ribbon Width = Inverse of Google POI Density)"
):
    colors_dict = {"ove": "#EA4335", "sf": "#FBBC05", "fsq": "#34A853", "osm": "#8E44AD"}
    display_labels = {"ove": "Overture", "sf": "SafeGraph", "fsq": "Foursquare", "osm": "OSM"}

    cats = sorted(df[primary_cat_col].unique())
    n_cats = len(cats)
    n_rows = int(np.ceil(n_cats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols + 1.5, 3 * n_rows + 1.5),
        squeeze=False,
        sharex=True,
        sharey=True,
        facecolor='white'
    )

    for i, (ax, cat) in enumerate(zip(axes.flat, cats)):
        df_cat = df[df[primary_cat_col] == cat].sort_values("bin_id")
        x = df_cat["bin_id"].values

        for c_col, d_col in zip(c_cols, den_cols):
            src = c_col.split('_')[-1]
            color = colors_dict.get(src, "#333333")

            y = df_cat[c_col].values
            dens = df_cat[d_col].values

            # 过滤 NaN
            mask = ~(np.isnan(y) | np.isnan(dens))
            x_v, y_v, d_v = x[mask], y[mask], dens[mask]
            if len(x_v) < 4:
                continue

            # --- 多项式回归平滑 ---
            x_smooth = np.linspace(x_v.min(), x_v.max(), 300)
            deg = 2
            coef_y = np.polyfit(x_v, y_v, deg)
            coef_d = np.polyfit(x_v, d_v, deg)
            spl_y = np.polyval(coef_y, x_smooth)
            spl_d = np.clip(np.polyval(coef_d, x_smooth), 0, None)

            # --- per-line 归一化 + 幂次拉大对比 ---
            den_norm = (spl_d - spl_d.min()) / (spl_d.max() - spl_d.min() + 1e-9)
            ribbon_half = np.power(1 - den_norm, 3) * 0.10 + 0.003

            y_upper = np.clip(spl_y + ribbon_half, 0, 1.1)
            y_lower = np.clip(spl_y - ribbon_half, 0, 1.1)

            # 绘制 ribbon
            ax.fill_between(x_smooth, y_lower, y_upper,
                            color=color, alpha=0.25, zorder=2)
            # 绘制中心线
            ax.plot(x_smooth, spl_y, color=color, lw=1.5, alpha=0.85, zorder=3)

        ax.set_title(cat, fontsize=14, pad=10)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0.5, 20.5)
        ax.set_xticks(range(1, 21, 2))

        if i % n_cols == 0:
            ax.set_ylabel("Match Rate", fontsize=10)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Distance bin (1 = closest to CBD)", fontsize=10)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes.flat[n_cats:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=0.98, va="top")

    legend_handles = [
        Line2D([0], [0], color=colors_dict[s], lw=3, alpha=0.8, label=display_labels[s])
        for s in ["ove", "sf", "fsq", "osm"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        ncol=4,
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

plot_variable_width_missmate(all_bins)