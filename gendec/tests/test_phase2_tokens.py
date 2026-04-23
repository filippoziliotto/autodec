import torch


def test_build_joint_tokens_concatenates_along_last_dim():
    from gendec.tokens import build_joint_tokens

    e = torch.zeros(4, 16, 15)
    z = torch.ones(4, 16, 64)
    ez = build_joint_tokens(e, z)

    assert ez.shape == (4, 16, 79)
    assert torch.all(ez[..., :15] == 0.0)
    assert torch.all(ez[..., 15:] == 1.0)


def test_split_joint_tokens_recovers_original_slices():
    from gendec.tokens import build_joint_tokens, split_joint_tokens

    e = torch.randn(2, 16, 15)
    z = torch.randn(2, 16, 64)
    ez = build_joint_tokens(e, z)

    split = split_joint_tokens(ez)
    assert torch.allclose(split["tokens_e"], e)
    assert torch.allclose(split["tokens_z"], z)


def test_split_joint_tokens_custom_residual_dim():
    from gendec.tokens import split_joint_tokens

    ez = torch.randn(1, 16, 47)  # 15 + 32
    split = split_joint_tokens(ez, residual_dim=32)
    assert split["tokens_e"].shape == (1, 16, 15)
    assert split["tokens_z"].shape == (1, 16, 32)


def test_joint_token_dim_constant():
    from gendec.tokens import JOINT_TOKEN_DIM, RESIDUAL_DIM_DEFAULT, TOKEN_DIM

    assert JOINT_TOKEN_DIM == TOKEN_DIM + RESIDUAL_DIM_DEFAULT
    assert JOINT_TOKEN_DIM == 79
