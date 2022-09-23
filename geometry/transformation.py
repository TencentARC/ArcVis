# -*- coding: utf-8 -*-

import numpy as np
import torch


def normalize(vec):
    """Normalize vector. Support numpy and torch

    Args:
        vec: (B, N, 3) or (B, 3)

    Returns:
        vec: (B, N, 3) or (B, 3)
    """
    if isinstance(vec, torch.Tensor):
        vec = vec / (torch.norm(vec, dim=-1).unsqueeze(-1) + 1e-8)
    elif isinstance(vec, np.ndarray):
        vec = vec / (np.linalg.norm(vec, axis=-1)[..., None] + 1e-8)

    return vec


def batch_dot_product(a: torch.Tensor, b: torch.Tensor):
    """Dot product in batch

    Args:
        a: (B, v)
        b: (B, v)

    Returns:
        dot_prod: (B,)
    """
    assert len(a.shape) == 2 and a.shape == b.shape
    dot_prod = torch.bmm(a.unsqueeze(1), b.unsqueeze(-1))[:, 0, 0]

    return dot_prod


def rotate_points(points: torch.Tensor, rot: torch.Tensor, rotate_only=False):
    """Rotate points by a rot

    Args:
        points: points, torch.tensor(B, N, 3)
        rot: rot matrix, torch.tensor(B, 4, 4). If rotate_only, can be (B, 3, 3)
        rotate_only: If True, only do the rotation using rot (B, 3, 3)

    Returns:
        rotated points in (B, N, 3)
    """
    proj_points = torch.einsum('bki,bji->bjk', rot[:, :3, :3], points)
    if not rotate_only:
        proj_points += rot[:, :3, 3]

    return proj_points


def rotate_matrix(rot: torch.Tensor, source: torch.Tensor):
    """Rotate a matrix by a rot

    Args:
        source: the origin matrix in (B, i, j)
        rot: the applied transformation in (B, k, j)

    Returns:
        rotated matrix in (B, k, j)
    """
    rot_mat = torch.einsum('bki,bij->bkj', rot, source)

    return rot_mat


def get_rotate_matrix_from_vec(vec_a: torch.Tensor, vec_b: torch.Tensor, eps=1e-5):
    """Get the rotation matrix from vec_a to vec_b
    Consider the case of same dir and reverse dir.

    Args:
        vec_a: the start in (B, 3)
        vec_b: the end vec in (B, 3)
        eps: threshold for comparing dot value, bt default 1e-5

    Returns:
        rotated matrix in (B, 3, 3)
    """
    assert vec_a.shape[1] == 3 and vec_b.shape[1] == 3, 'Please input vecs with (b, 3) dim'
    vec_a_norm = normalize(vec_a)  # do not change to original value
    vec_b_norm = normalize(vec_b)

    # consider the special case
    vec_dot = batch_dot_product(vec_a_norm, vec_b_norm)
    invalid_pos = torch.abs(vec_dot - 1.0) < eps
    invalid_neg = torch.abs(vec_dot + 1.0) < eps
    valid = ~torch.logical_or(invalid_pos, invalid_neg)

    n = torch.cross(vec_a_norm, vec_b_norm, dim=-1)
    n = normalize(n)  # (B, 3)

    base_a = torch.cat([vec_a_norm.unsqueeze(1),
                        torch.cross(n, vec_a_norm, dim=-1).unsqueeze(1),
                        n.unsqueeze(1)],
                       dim=1)  # (B, 3, 3)
    base_b = torch.cat([vec_b_norm.unsqueeze(1),
                        torch.cross(n, vec_b_norm, dim=-1).unsqueeze(1),
                        n.unsqueeze(1)],
                       dim=1)  # (B, 3, 3)

    matrix_valid = torch.matmul(base_b[valid], torch.inverse(base_a[valid]))

    # get full matrix
    matrix = torch.eye(3, dtype=vec_a.dtype, device=vec_a.device).unsqueeze(0)
    matrix = torch.repeat_interleave(matrix, vec_a.shape[0], dim=0)
    matrix[valid] = matrix_valid
    matrix[invalid_neg] = -1.0 * matrix[invalid_neg]

    return matrix
