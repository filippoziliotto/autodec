import torch


def build_flow_batch(e0, e1=None, t=None):
    if e1 is None:
        e1 = torch.randn_like(e0)
    if t is None:
        t = torch.rand(e0.shape[0], device=e0.device, dtype=e0.dtype)
    t_tokens = t.view(-1, 1, 1)
    et = (1.0 - t_tokens) * e0 + t_tokens * e1
    return {
        "E0": e0,
        "E1": e1,
        "Et": et,
        "velocity_target": e1 - e0,
        "t": t,
    }
