"""各段のテンソル次数とパラメータ数を実測して表にする。"""
import sys, torch, torch.nn as nn
sys.path.insert(0, '/home/shunya/research/dog-radar-vitals/src')
from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN

B, T, N, D = 2, 800, 50, 32
m = RPeakSpatialGNN(n_points=N, embed_dim=D, k_neighbors=6, n_gat_layers=2, n_downsample=3).eval()
rcg = torch.randn(B, T, N); pos = torch.randn(B, N, 3)

print(f"{'段':<44} {'次数':<28} {'要素数':>10}")
def row(name, shp):
    n = 1
    for s in shp: n *= s
    print(f"{name:<44} {str(tuple(shp)):<28} {n:>10,}")

row("入力 RCG", rcg.shape); row("入力 posXYZ", pos.shape)
x = rcg.transpose(1,2); row("転置 (点を先に)", x.shape)
h = x.reshape(B*N, 1, T); row("点ごとに分解 (重み共有のため)", h.shape)
for i, l in enumerate(m.point_encoder.conv):
    h = l(h)
    if isinstance(l, nn.Conv1d): row(f"  点エンコーダ conv{i//3+1} (stride 2)", h.shape)
pf = h.reshape(B, N, h.shape[1], h.shape[2]); row("点ごと時間特徴", pf.shape)
Tr = pf.shape[-1]
pe = m.pos_embed(pos); row("位置埋め込み (3 -> 32)", pe.shape)
pf = pf + pe.unsqueeze(-1); row("位置を加算後", pf.shape)
g = pf.permute(0,3,1,2).reshape(B*Tr*N, -1); row("グラフのノード行列", g.shape)
row("  グラフの辺 edge_index (無向化済み)", (2, B*Tr*N*6*2))
g = g.reshape(B, Tr, N, -1).mean(dim=2); row("点方向に平均プーリング", g.shape)
g = g.transpose(1,2); row("デコーダ入力", g.shape)
d = g
for i, l in enumerate(m.decoder.deconv):
    d = l(d)
    if isinstance(l, nn.ConvTranspose1d): row(f"  時間デコーダ deconv{i//3+1} (stride 2)", d.shape)
out = m.decoder.head(d).squeeze(1); row("出力 heatmap", out.shape)

print(f"\n{'モジュール':<34} {'パラメータ数':>12}")
tot = 0
for name, mod in [("点ごと時間エンコーダ", m.point_encoder), ("位置埋め込み", m.pos_embed),
                  ("空間集約 (GAT x2)", m.gat_layers), ("時間デコーダ", m.decoder)]:
    p = sum(x.numel() for x in mod.parameters()); tot += p
    print(f"{name:<34} {p:>12,}")
print(f"{'合計':<34} {tot:>12,}   ({tot/1e6:.4f} M)")
