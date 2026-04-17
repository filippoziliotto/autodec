from types import SimpleNamespace

import torch


def test_head_returns_exist_logit_and_probability():
    from autodec.models.heads import SuperDecHead

    ctx = SimpleNamespace(rotation6d=False, extended=False)
    head = SuperDecHead(emb_dims=8, ctx=ctx)

    out = head(torch.randn(2, 3, 8))

    assert out["exist_logit"].shape == (2, 3, 1)
    assert out["exist"].shape == (2, 3, 1)
    assert torch.allclose(out["exist"], torch.sigmoid(out["exist_logit"]))
    assert out["rotation_quat"].shape == (2, 3, 4)
    assert out["rotate"].shape == (2, 3, 3, 3)
