# -*- coding: utf-8 -*-

import numpy as np
import torch


def np_wrapper(func, *args):
    """ Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    Reference from VideoPose3d: https://github.com/facebookresearch/VideoPose3D/blob/master/common/utils.py
    """
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        return result.numpy()
    else:
        return result


def torch_to_np(tensor):
    """Torch tensor to numpy array"""
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise NotImplementedError('Please use torch tensor or np array')


def set_tensor_to_zeros(tensor: torch.Tensor, a_tol=1e-5):
    """Set tensor with very small value as 0"""
    tensor[torch.abs(tensor) < a_tol] = 0.0

    return tensor
