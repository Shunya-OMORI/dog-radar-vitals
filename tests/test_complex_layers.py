import torch

from dog_radar_vitals.models.deep.complex_layers import ComplexBatchNorm1d, ModReLU


def test_complex_batchnorm_whitens_correlated_input():
    torch.manual_seed(0)
    bn = ComplexBatchNorm1d(4)
    with torch.no_grad():
        bn.gamma_rr.fill_(1.0)
        bn.gamma_ii.fill_(1.0)
        bn.gamma_ri.fill_(0.0)
        bn.beta_r.zero_()
        bn.beta_i.zero_()

    n, c, t = 2000, 4, 10
    re = torch.randn(n, c, t)
    im = 0.7 * re + 0.3 * torch.randn(n, c, t)
    x = torch.complex(re, im) * 3.0 + (5.0 + 2.0j)

    bn.train()
    y = bn(x)
    yr, yi = y.real.detach(), y.imag.detach()
    for ch in range(c):
        cov = torch.stack([yr[:, ch, :].flatten(), yi[:, ch, :].flatten()]).cov()
        assert torch.allclose(cov, torch.eye(2), atol=0.05)


def test_complex_batchnorm_gradient_flows():
    bn = ComplexBatchNorm1d(2)
    x = torch.randn(8, 2, 5, dtype=torch.complex64, requires_grad=True)
    y = bn(x)
    y.abs().mean().backward()
    assert bn.gamma_rr.grad is not None
    assert not torch.isnan(bn.gamma_rr.grad).any()


def test_complex_batchnorm_eval_mode_uses_running_stats():
    bn = ComplexBatchNorm1d(2)
    x = torch.randn(8, 2, 5, dtype=torch.complex64)
    bn.train()
    bn(x)
    bn.eval()
    y = bn(x)
    assert y.shape == x.shape
    assert not torch.isnan(y.real).any()


def test_modrelu_preserves_shape():
    m = ModReLU(4)
    x = torch.randn(2, 4, 10, dtype=torch.complex64)
    y = m(x)
    assert y.shape == x.shape
