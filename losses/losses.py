import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    difference = torch.subtract(input_tensor, target)
    pow2 = torch.pow(difference, 2)
    return pow2.mean()
    #return utils.not_implemented()
