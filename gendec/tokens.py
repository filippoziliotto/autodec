import torch


PRIMITIVE_COUNT = 16
TOKEN_DIM = 15

SCALE_SLICE = slice(0, 3)
SHAPE_SLICE = slice(3, 5)
TRANS_SLICE = slice(5, 8)
ROT6D_SLICE = slice(8, 14)
EXIST_LOGIT_SLICE = slice(14, 15)


def build_scaffold_tokens(scale, shape, trans, rot6d, exist_logit):
    return torch.cat([scale, shape, trans, rot6d, exist_logit], dim=-1)


def split_scaffold_tokens(tokens):
    return {
        "scale": tokens[..., SCALE_SLICE],
        "shape": tokens[..., SHAPE_SLICE],
        "trans": tokens[..., TRANS_SLICE],
        "rot6d": tokens[..., ROT6D_SLICE],
        "exist_logit": tokens[..., EXIST_LOGIT_SLICE],
    }
