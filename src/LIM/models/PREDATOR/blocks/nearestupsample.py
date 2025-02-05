import torch
from LIM.data.structures.cloud import Cloud
from debug.decorators import identify_method

class NearestUpsample(torch.nn.Module):

    def __init__(self) -> None:
        super(NearestUpsample, self).__init__()
    
    def forward(self, cloud: Cloud) -> Cloud:
        if len(cloud.subpoints.upsamples) == 0:
            raise "Couldn't find upsamples for cloud"
        cloud.features = self._closest_pool(cloud.features, cloud.subpoints.upsamples)
        return cloud

    def _closest_pool(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        features = torch.cat((features, torch.zeros_like(features[:1, :])), dim=0)
        return self._gather(features, indices[:, 0])

    def _gather(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Custom gather operation for faster backpropagation

        Args:
            x: input with shape [N, D_1, ..., D_d]
            indices: indexing tensor with shape [N, ..., N_m]

        Returns:
            torch.Tensor: ...
        """
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
