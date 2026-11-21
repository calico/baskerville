"""DepthwiseConv1d compiled (shifted sum) matches eager (F.conv1d), forward and grads.

CPU uses aot_eager: same traced graph, no C++ toolchain needed.
"""

import pytest
import torch

from baskerville.layers import DepthwiseConv1d

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("kernel_size", [4, 7])
@pytest.mark.parametrize("bias", [False, True])
def test_compiled_matches_eager(device, kernel_size, bias):
    torch.manual_seed(0)
    conv = DepthwiseConv1d(24, kernel_size, bias=bias).to(device)
    x = torch.randn(2, 24, 100, device=device, requires_grad=True)
    backend = "inductor" if device == "cuda" else "aot_eager"
    compiled = torch.compile(conv, backend=backend, fullgraph=True)

    results = []
    for f in (conv, compiled):
        conv.zero_grad()
        x.grad = None
        y = f(x)
        y.backward(torch.cos(y))
        results.append((y, x.grad, conv.weight.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast on CUDA")
def test_compiled_bf16_autocast():
    torch.manual_seed(0)
    conv = DepthwiseConv1d(64, 7).cuda()
    x = torch.randn(2, 64, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_eager = conv(x)
        y_comp = torch.compile(conv)(x)
    assert y_comp.dtype == y_eager.dtype == torch.bfloat16
    torch.testing.assert_close(y_comp.float(), y_eager.float(), rtol=2e-2, atol=2e-2)
