import torch


class NearestUpsample(torch.nn.Module):
    layer_ind: int  # the index of the layer where the skip features come from

    def __init__(self, layer_ind: int) -> None:
        super(NearestUpsample, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self._closest_pool(features, indices[self.layer_ind - 1])

    def _closest_pool(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        features = torch.cat((features, torch.zeros_like(features[:1, :])), 0)
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
