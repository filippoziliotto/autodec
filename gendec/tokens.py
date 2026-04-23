import torch


PRIMITIVE_COUNT = 16
TOKEN_DIM = 15

SCALE_SLICE = slice(0, 3)
SHAPE_SLICE = slice(3, 5)
TRANS_SLICE = slice(5, 8)
ROT6D_SLICE = slice(8, 14)
EXIST_LOGIT_SLICE = slice(14, 15)

# Phase 2: joint (E, Z) token layout
RESIDUAL_DIM_DEFAULT = 64
JOINT_TOKEN_DIM = TOKEN_DIM + RESIDUAL_DIM_DEFAULT  # 79

# Slices into the joint token [E | Z]
JOINT_E_SLICE = slice(0, TOKEN_DIM)           # 0:15
JOINT_Z_SLICE = slice(TOKEN_DIM, None)        # 15:79  (adapts to actual residual_dim)


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


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def build_joint_tokens(tokens_e, tokens_z):
    """Concatenate explicit scaffold tokens and residual tokens along last dim.

    Args:
        tokens_e: [..., 15]
        tokens_z: [..., D]

    Returns:
        tokens_ez: [..., 15 + D]
    """
    return torch.cat([tokens_e, tokens_z], dim=-1)


def split_joint_tokens(tokens_ez, residual_dim=RESIDUAL_DIM_DEFAULT):
    """Split a joint token tensor back into (tokens_e, tokens_z).

    Args:
        tokens_ez: [..., 15 + residual_dim]
        residual_dim: dimension of the residual branch

    Returns:
        dict with keys ``tokens_e`` and ``tokens_z``
    """
    return {
        "tokens_e": tokens_ez[..., :TOKEN_DIM],
        "tokens_z": tokens_ez[..., TOKEN_DIM:TOKEN_DIM + residual_dim],
    }
