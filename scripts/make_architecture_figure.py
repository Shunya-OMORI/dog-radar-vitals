"""発表資料用のアーキテクチャ図 (2026-08-07)。

空間GNN (rpeak_spatial_gnn) + サブサンプル重心復号のパイプラインを6要素で図示する。
学習モジュール(青)と学習不要の後処理(橙)を塗り分ける。
出力: reports/mmecg_comparison/architecture_spatial_gnn.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'IPAexGothic'

C_LEARN = '#dce9f9'   # 学習モジュール (blue tint)
C_LEARN_EDGE = '#2a78d6'
C_POST = '#fbe3d8'    # 学習不要の後処理 (orange tint)
C_POST_EDGE = '#eb6834'
C_IO = '#efeeea'      # 入出力 (gray)
C_IO_EDGE = '#8a8984'
INK = '#0b0b0b'

fig, ax = plt.subplots(figsize=(13.2, 3.4), dpi=200)
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 3.4)
ax.axis('off')
fig.patch.set_facecolor('#fcfcfb')

BW, BH, Y = 1.72, 1.14, 1.5  # box width/height, main row y (center)

boxes = [
    (0.90, Y, 'レーダ入力 RCG\n50点 × 800サンプル\n(4秒窓, 200 Hz)', C_IO, C_IO_EDGE),
    (2.85, Y, '点ごとの時間エンコーダ\n1D CNN\n(50点で重み共有)', C_LEARN, C_LEARN_EDGE),
    (4.80, Y, '空間グラフ注意 GAT ×2\nk近傍グラフ (k=6)\n近い点同士のみ結合', C_LEARN, C_LEARN_EDGE),
    (6.75, Y, '時間デコーダ\n転置畳み込み ×3\n→ 800サンプルに復元', C_LEARN, C_LEARN_EDGE),
    (8.70, Y, 'R波 heatmap\n各時刻のR波確率 [0,1]\n時間分解能 5 ms を維持', C_IO, C_IO_EDGE),
    (10.65, Y, 'ピーク復号（学習不要）\nしきい値0.3 → 重心補正\nsoft-argmax ±85 ms', C_POST, C_POST_EDGE),
    (12.35, Y, '出力\nR波時刻列\n→ RR間隔・HRV', C_IO, C_IO_EDGE),
]

for i, (cx, cy, text, fc, ec) in enumerate(boxes):
    w = 1.30 if i == len(boxes) - 1 else BW
    box = FancyBboxPatch((cx - w / 2, cy - BH / 2), w, BH,
                         boxstyle='round,pad=0.06,rounding_size=0.10',
                         facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=8.6, color=INK, linespacing=1.55)

for i in range(len(boxes) - 1):
    x0 = boxes[i][0] + (1.30 if i == len(boxes) - 1 else BW) / 2 + 0.07
    x1 = boxes[i + 1][0] - (1.30 if i + 1 == len(boxes) - 1 else BW) / 2 - 0.07
    ax.add_patch(FancyArrowPatch((x0, Y), (x1, Y), arrowstyle='-|>',
                                 mutation_scale=14, color='#52514e', linewidth=1.4))

# posXYZ 側枝 (GAT層へ)
px, py = 4.80, 0.38
pbox = FancyBboxPatch((px - 1.55, py - 0.26), 3.1, 0.52,
                      boxstyle='round,pad=0.05,rounding_size=0.08',
                      facecolor=C_LEARN, edgecolor=C_LEARN_EDGE, linewidth=1.2)
ax.add_patch(pbox)
ax.text(px, py, '3次元座標 posXYZ の埋め込み (Linear 3→32)', ha='center', va='center',
        fontsize=8.0, color=INK)
ax.add_patch(FancyArrowPatch((px, py + 0.30), (px, Y - BH / 2 - 0.09), arrowstyle='-|>',
                             mutation_scale=12, color='#52514e', linewidth=1.2))

# タイトルと凡例
ax.text(0.10, 3.16, '空間GNNによるレーダR波検出パイプライン（学習パラメータ合計 0.034 M）',
        fontsize=11, color=INK, weight='bold', ha='left', va='center')
lx = 7.35
for fc, ec, label in [(C_LEARN, C_LEARN_EDGE, '学習されるモジュール'),
                      (C_POST, C_POST_EDGE, '学習不要の後処理'),
                      (C_IO, C_IO_EDGE, '入出力・中間表現')]:
    ax.add_patch(FancyBboxPatch((lx, 3.02), 0.28, 0.22, boxstyle='round,pad=0.02,rounding_size=0.04',
                                facecolor=fc, edgecolor=ec, linewidth=1.1))
    ax.text(lx + 0.36, 3.13, label, fontsize=8.2, va='center', color=INK)
    lx += 2.05

fig.savefig('/home/shunya/research/dog-radar-vitals/reports/mmecg_comparison/architecture_spatial_gnn.png',
            bbox_inches='tight', facecolor='#fcfcfb')
print('saved architecture_spatial_gnn.png')
