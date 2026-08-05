"""単一拍ECGオートエンコーダをSchellenberger(ヒト30名、レーダ非依存の臨床ECG)で事前学習する。

ユーザ提案の「ECG基盤モデル/事前学習モデル」路線の実装。学習済みdecoderを
`models/deep/radarode_sceg.py`の`PretrainedShapeDecoder`が読み込み、McSharry型ODEデコーダの
代わりの形状事前分布として使う。

使い方:
    python scripts/pretrain_ecg_beat_autoencoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.data.ecg_beat_dataset import ECGBeatDataset  # noqa: E402
from dog_radar_vitals.models.deep.ecg_beat_autoencoder import BeatAutoencoder  # noqa: E402
from dog_radar_vitals.seeding import set_all_seeds  # noqa: E402

OUT_PATH = REPO_ROOT / "runs" / "ecg_beat_autoencoder_pretrained.pt"


def main() -> None:
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ECGBeatDataset(REPO_ROOT / "data" / "raw")
    print(f"total beats: {len(ds)}")
    n_val = int(len(ds) * 0.1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = BeatAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(30):
        model.train()
        train_loss, n = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            pred = model(x)
            loss = loss_fn(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, x).item()
                n += 1
        val_loss /= n
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUT_PATH)

    print(f"saved best model (val_loss={best_val_loss:.4f}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
