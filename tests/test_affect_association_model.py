import torch

from dog_radar_vitals.affect.association_model import (
    DeltaRegressor,
    JointEmbeddingModel,
    info_nce_loss,
)


def test_joint_embedding_shapes_and_grad():
    model = JointEmbeddingModel(text_dim=1024, delta_dim=1, joint_dim=128)
    text_emb = torch.randn(8, 1024)
    delta_s = torch.randn(8, 1)
    z_text, z_delta = model(text_emb, delta_s)
    assert z_text.shape == (8, 128)
    assert z_delta.shape == (8, 128)
    assert torch.allclose(z_text.norm(dim=-1), torch.ones(8), atol=1e-4)

    loss = info_nce_loss(z_text, z_delta)
    assert loss.ndim == 0
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # NaNでない


def test_delta_regressor_shapes_and_grad():
    reg = DeltaRegressor(text_dim=1024, out_dim=1)
    text_emb = torch.randn(4, 1024)
    pred = reg(text_emb)
    assert pred.shape == (4, 1)
    target = torch.randn(4, 1)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    assert all(p.grad is not None for p in reg.parameters())
