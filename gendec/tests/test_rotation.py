import torch


def test_matrix_to_rot6d_uses_first_two_columns():
    from gendec.models.rotation import matrix_to_rot6d

    matrix = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )

    rot6d = matrix_to_rot6d(matrix)

    assert rot6d.shape == (1, 6)
    assert torch.allclose(rot6d[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_rot6d_to_matrix_returns_valid_rotation():
    from gendec.models.rotation import rot6d_to_matrix

    rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    matrix = rot6d_to_matrix(rot6d)

    assert matrix.shape == (1, 3, 3)
    eye = torch.eye(3).unsqueeze(0)
    assert torch.allclose(matrix.transpose(-1, -2) @ matrix, eye, atol=1e-5)
    assert torch.allclose(torch.det(matrix), torch.ones(1), atol=1e-5)
