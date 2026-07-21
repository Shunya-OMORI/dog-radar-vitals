import torch

from dog_radar_vitals.models.classical.registry import build_classical_model
from dog_radar_vitals.models.deep.registry import build_deep_model


def test_all_deep_models_forward_pass():
    batch, seq_len, n_bins = 2, 500, 467
    x = torch.randn(batch, seq_len, n_bins)
    for name in ["transformer", "cnn1d", "lstm"]:
        model = build_deep_model(name, n_bins=n_bins, n_outputs=1)
        out = model(x)
        assert out.shape == (batch, 1), name


def test_all_classical_models_fit_predict():
    import numpy as np

    X = np.random.randn(20, 9)
    y = np.random.randn(20)
    for name, kwargs in [
        ("ridge", {"alpha": 1.0}),
        ("random_forest", {"n_estimators": 5, "random_state": 42}),
        ("gradient_boosting", {"n_estimators": 5, "random_state": 42}),
    ]:
        model = build_classical_model(name, **kwargs)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (20,), name
