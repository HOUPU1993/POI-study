import numpy as np
import matplotlib.pyplot as plt

def plot_missmate_by_dist(
    df,
    dens_cols=("total_den_google", "match_den_ove", "match_den_sf", "match_den_fsq", "match_den_osm"),
    colors=None,
    markers=True,
    primary_cat_col="primary_cat",
    xlabel="Distance bin (1 = closest to CBD)",
    ylabel="Density (per km²)",
    title="Category-level POI Match Density Comparison relative to Google Places",
    n_cols=4,
    xtick_fontsize=8,
):
    # 1. 扩展颜色配置，适应 5 个因素
    if colors is None:
        colors = {
            "total_den_google": "#4285F4",  # Google Blue
            "match_den_ove":    "#EA4335",  # Overture Red
            "match_den_sf":     "#FBBC05",  # SafeGraph Yellow
            "match_den_fsq":    "#34A853",  # Foursquare Green
            "match_den_osm":    "#8E44AD",  # OSM Purple
        }

    # 自定义图例显示的文字
    display_labels = {
        "total_den_google": "Google (Total)",
        "match_den_ove":    "Overture",
        "match_den_sf":     "SafeGraph",
        "match_den_fsq":    "Foursquare",
        "match_den_osm":    "OSM"
    }

    cats = sorted(df[primary_cat_col].unique())
    n_cats = len(cats)
    n_rows = int(np.ceil(n_cats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols + 1, 3 * n_rows + 1), # 稍微增加高度给顶部留白
        squeeze=False,
        sharex=True,
        sharey=True
    )

    for ax, cat in zip(axes.flat, cats):
        df_cat = df[df[primary_cat_col] == cat].sort_values("bin_id")
        x = df_cat["bin_id"]

        for col in dens_cols:
            if col in df_cat.columns:
                ax.plot(
                    x,
                    df_cat[col],
                    label=display_labels.get(col, col),
                    color=colors.get(col, "#333333"),
                    marker="o" if markers else None,
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_title(cat, fontsize=14, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        
        # 移除图框逻辑 (可选，为了保持极简风)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # 隐藏多余的子图
    for ax in axes.flat[n_cats:]:
        ax.axis("off")

    # 2. 核心调整：Legend 与 Title 分层布局
    # 首先设置大标题，固定在最上方
    fig.suptitle(title, fontsize=14, y=0.98, va="top")

    # 获取图例句柄
    handles, labels = axes.flat[0].get_legend_handles_labels()

    # 放置图例：bbox_to_anchor 的第二个参数 (0.93) 决定了它的垂直位置
    # 它应该位于 suptitle (0.98) 之下，子图区域 (0.90) 之上
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93), 
        frameon=False,
        ncol=len(dens_cols),
        fontsize=10
    )

    # 3. 调整子图区域，为顶部的 Title 和 Legend 留出空间 (rect=[left, bottom, right, top])
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()