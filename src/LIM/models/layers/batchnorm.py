from typing import Any

import torch
from multimethod import multimethod

from LIM.data.structures import PCloud
from LIM.log import log


class BatchNorm(torch.nn.Module):
    in_dim: int
    momentum: float

    def __init__(self, in_dim: int, momentum: float) -> None:
        super(BatchNorm, self).__init__()
        self.in_dim = in_dim
        self.momentum = momentum
        self._batch_norm = torch.nn.InstanceNorm1d(in_dim, momentum=momentum)

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x.unsqueeze(2).transpose(0, 2)
            x = self._batch_norm(x)
            x = x.transpose(0, 2).squeeze(2)
        except Exception as e:
            log.error(f"Something went wrong in BatchNorm: {e}\n{x.shape=}")
        return x

    @multimethod
    def forward(self, cloud: PCloud) -> PCloud:
        cloud.features = self.forward(cloud.features)
        return cloud

    def __repr__(self) -> str:
        return f"BatchNorm(in_dim={self.in_dim}, momentum={self.momentum})"
