import torch
from LIM.data.structures import Cloud
from debug.decorators import identify_method

class MaxPool(torch.nn.Module):
    def __init__(self) -> None:
        super(MaxPool, self).__init__()
    
    def __repr__(self) -> str:
        return "MaxPool()"

    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = cloud.features.max(dim=-1, keepdim=True)[0]
        return cloud


class MaxPoolNeighbors(torch.nn.Module):
    def __init__(self) -> None:
        super(MaxPoolNeighbors, self).__init__()

    # @identify_method
    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = torch.cat((cloud.features, torch.zeros_like(cloud.features[:1, :])), 0)
        cloud.features, _ = torch.max(self._gather(cloud.features, cloud.pools), dim=1)
        return cloud

    # @identify_method
    def _gather(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        for idx, value in enumerate(indices.size()[1:]):
            x = x.unsqueeze(idx + 1)
            new_s = list(x.size())
            new_s[idx + 1] = value
            x = x.expand(new_s)

        n = len(indices.size())
        for idx, value in enumerate(x.size()[n:]):
            indices = indices.unsqueeze(idx + n)
            new_s = list(indices.size())
            new_s[idx + n] = value
            indices = indices.expand(new_s)

        return x.gather(0, indices)
