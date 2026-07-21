"""古典MLモデル(scikit-learn Estimator)の名前→コンストラクタの一元管理。

深層モデルと違い、古典MLは薄いラッパクラスを作るまでもなくscikit-learnの
Estimatorをそのまま使う。特徴量は `data/features.py` が生成する。
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSICAL_MODEL_REGISTRY: dict[str, Any] = {
    "ridge": lambda **kwargs: make_pipeline(StandardScaler(), Ridge(**kwargs)),
    "random_forest": lambda **kwargs: RandomForestRegressor(**kwargs),
    "gradient_boosting": lambda **kwargs: GradientBoostingRegressor(**kwargs),
}


def build_classical_model(name: str, **kwargs: Any):
    if name not in CLASSICAL_MODEL_REGISTRY:
        raise ValueError(f"unknown classical model '{name}'. known models: {sorted(CLASSICAL_MODEL_REGISTRY)}")
    return CLASSICAL_MODEL_REGISTRY[name](**kwargs)
