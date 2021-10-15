import torch
from torch import Tensor


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    assert mask.dtype == torch.bool
    x = torch.where(mask, x, torch.zeros_like(x))
    return torch.sum(x, dim=dim) / torch.count_nonzero(mask, dim=dim)
