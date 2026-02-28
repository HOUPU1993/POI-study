import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_bubble_distancebins(
    df,
    primary_cat_col="primary_cat",
    bin_col="bin_id",
    sources=["ove", "sf", "fsq", "osm"],
):
    # 基础准备
    cats = sorted(df[primary_cat_col].unique(), reverse=True) 
    n_cats = len(cats)
    
    # 【优化】动态计算高度：根据类别数量自动调整 figsize，避免固定 38 导致的过度留白
    dynamic_height = n_cats * 1.8 + 4
    fig, ax = plt.subplots(figsize=(22, dynamic_height), facecolor='white')

    cmap = cm.get_cmap('RdYlGn') 
    
    for i, cat in enumerate(cats):
        base_y = i * 4  
        df_cat = df[df[primary_cat_col] == cat].sort_values(bin_col)
        
        ax.text(-0.6, base_y - 0.6, cat.upper(), ha='right', va='center', fontweight='bold', fontsize=14)

        for j, src in enumerate(sources):
            line_y = base_y - (j * 0.8) 
            x = df_cat[bin_col]
            den_val = df_cat[f"match_den_{src}"]
            c_val = df_cat[f"match_c_{src}"]
            
            ax.hlines(line_y, x.min(), x.max(), colors='gray', alpha=0.15, linewidth=1, zorder=1)
            ax.text(x.min() - 0.3, line_y, src.upper(), ha='right', va='center', fontsize=9, alpha=0.7)

            bubble_sizes = np.sqrt(den_val) * 3000 + 100 
            
            scatter = ax.scatter(
                x, [line_y] * len(x),
                s=bubble_sizes,
                c=c_val,
                cmap=cmap,
                vmin=0, vmax=1,
                edgecolors='black',
                linewidths=0.5,
                alpha=0.85,
                zorder=2
            )

    # 【优化】X 轴移到上方
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(sorted(df[bin_col].unique()))
    ax.set_xlabel("Distance Bins (Closest to CBD → Farthest)", fontsize=12, labelpad=20)
    
    for spine in ["right", "left", "yaxis"]:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    
    # 顶部和底部的脊柱线可以根据视觉需要保留或隐藏
    ax.spines["top"].set_color("#cccccc")
    ax.spines["bottom"].set_visible(False)

    # 3. 顶部横向图例（位置随 X 轴上移做了微调）
    fig.suptitle("POI Match Density (Size) & Match Rate (Color) by Category and Source", 
                 fontsize=22, y=0.98, va='top')

    # 颜色条
    cax = fig.add_axes([0.8, 0.93, 0.15, 0.01]) # 稍微下移，避免撞到主标题
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=cax, orientation='horizontal')
    cb.set_label('Match Ratio (match_c)', fontsize=11, labelpad=-35)

    # 尺寸图例
    sax = fig.add_axes([0.71, 0.90, 0.2, 0.02])
    sax.axis('off')
    legend_sizes = [0.01, 0.1, 0.4]
    x_positions = np.linspace(0, 1, len(legend_sizes))
    for idx, sz in enumerate(legend_sizes):
        sax.scatter(x_positions[idx], 0.5, s=np.sqrt(sz)*1000 + 100, c='gray', alpha=0.5, edgecolors='black')
        sax.text(x_positions[idx] + 0.08, 0.5, f'Density {sz}', va='center', fontsize=10)
    
    # 【优化】压缩绘图区，紧贴 X 轴上方
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    plt.show()

# 调用
plot_bubble_distancebins(all_bins)