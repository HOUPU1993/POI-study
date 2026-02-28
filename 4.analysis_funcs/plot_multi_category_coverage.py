import matplotlib.pyplot as plt
import numpy as np

def plot_multi_category_coverage(
    df,
    cat_col="primary_cat",
    # 定义需要对比的源及其对应的列名后缀
    sources=["ove", "sf", "fsq", "osm"],
    ref_col="google_count",
    figsize=(22, 7),
    title="Category-level POI Match Coverage Relative to Google Places"
):
    df = df.reset_index(drop=True)
    n_categories = len(df)
    n_sources = len(sources) + 1  # 加上 Google 参考栏
    
    # 调整宽度：总宽度为 0.8，平分给各个 source
    total_width = 0.8
    single_width = total_width / n_sources
    x = np.arange(n_categories)

    # 创建画布，frameon=False 去掉背景框
    fig, ax = plt.subplots(figsize=figsize, frameon=True)

    # 1. 绘制 Google 参考栏 (通常放在第一位)
    ax.bar(
        x - total_width/2 + single_width/2, 
        df[ref_col], 
        single_width, 
        label="Google (Ref)", 
        color="#4285F4", # Google Blue
        alpha=0.8
    )

    # 2. 循环绘制其他 Sources
    colors = ["#EA4335", "#FBBC05", "#34A853", "#8E44AD"] # 红, 黄, 绿, 紫
    for i, src in enumerate(sources):
        offset = (i + 1) * single_width
        pos = x - total_width/2 + offset + single_width/2
        
        count_col = f"{src}_count"
        ratio_col = f"{src}_google_m"
        
        bars = ax.bar(pos, df[count_col], single_width, label=src.upper(), color=colors[i])

        # 在每个柱子上方标注百分比
        for j, bar in enumerate(bars):
            val = df.loc[j, ratio_col] * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (df[ref_col].max() * 0.01), # 微调标注高度
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=7, rotation=0
            )

    # ---- 移除图框 (Spines) ----
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    
    # 移除刻度线，保留标签
    ax.tick_params(axis='both', which='both', length=0)

    # 设置样式
    ax.set_xticks(x)
    ax.set_xticklabels(df[cat_col], rotation=45, ha="right")
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel("POI Count")
    
    # 添加水平参考线（可选，增加可读性）
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()

# 调用函数
plot_multi_category_coverage(summarize_df_merge)