import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# 1. 数据处理函数 (保持不变)
def process_stats(df, source_name):
    temp = df.copy()
    temp['is_mistake'] = (temp["is_true_match"] == "0")
    temp['is_miss'] = temp["is_true_match"].isna()
    
    stats = temp.groupby("primary_cat").agg(
        n_mistake=('is_mistake', 'sum'),
        n_miss=('is_miss', 'sum'),
        total_count=('primary_cat', 'count')
    ).reset_index()
    
    stats['total_non_match'] = stats['n_mistake'] + stats['n_miss']
    stats['miss_rate'] = stats['total_non_match'] / stats['total_count']
    stats['source'] = source_name
    return stats

# 2. 汇总数据
dfs_input = {
    "OVE": bspo_gplc_ove,
    "SF": bspo_gplc_sf,
    "FSQ": bspo_gplc_fsq,
    "OSM": bspo_gplc_osm
}

all_stats_list = [process_stats(d, name) for name, d in dfs_input.items()]
df_combined = pd.concat(all_stats_list)

# 3. 筛选并排序 (按总非匹配数从大到小)
TOP_N = 20
top_cats = df_combined.groupby("primary_cat")['total_non_match'].sum().nlargest(TOP_N).index.tolist()
top_cats = sorted(top_cats, reverse=True) 

# 4. 绘图
n_cats = len(top_cats)
sources = ["OVE", "SF", "FSQ", "OSM"]
n_src = len(sources)
colors = {"Mistake": '#e74c3c', "Miss": '#3498db'}

fig, ax = plt.subplots(figsize=(14, n_cats * 1.2), facecolor='white')

y_base = np.arange(n_cats) * (n_src + 1) 

for i, cat in enumerate(top_cats):
    cat_data = df_combined[df_combined['primary_cat'] == cat]
    
    for j, src in enumerate(sources):
        row_y = y_base[i] + (n_src - 1 - j) 
        src_row = cat_data[cat_data['source'] == src]
        
        if not src_row.empty:
            mistake = src_row['n_mistake'].values[0]
            miss = src_row['n_miss'].values[0]
            rate = src_row['miss_rate'].values[0]
            total = src_row['total_non_match'].values[0]
            
            ax.barh(row_y, mistake, color=colors["Mistake"], height=0.7)
            ax.barh(row_y, miss, left=mistake, color=colors["Miss"], height=0.7)
            
            # 【优化】增加负向偏移 (-15)，确保 OVE/SF 等文字不与 Y 轴刻度重叠
            ax.text(-15, row_y, src, ha='right', va='center', fontsize=8, alpha=0.5)
            
            if total > 0:
                ax.text(total + 8, row_y, f'{rate:.1%}', va='center', ha='left', 
                        fontsize=8, color='darkred', fontweight='bold')

# 5. 【优化】样式修正
ax.set_yticks(y_base + (n_src-1)/2)
# 去掉 fontweight='bold'，还原为普通字体
ax.set_yticklabels(top_cats, fontsize=11)
ax.tick_params(axis='y', which='major', pad=20)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_xlabel("Count of Non-matches", fontsize=11)
ax.set_title("Non-match Rate(Mistakematch + Missmatch) Composition by Category & Source", fontsize=14, pad=35)

# 统一图例
legend_elements = [Patch(facecolor=colors["Mistake"], label='Mistake Match'),
                   Patch(facecolor=colors["Miss"], label='Miss Match')]
ax.legend(handles=legend_elements, loc='upper right',  frameon=False)

# 【优化】通过 subplots_adjust 明确给左侧留出空间 (left=0.2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.08)

plt.show()