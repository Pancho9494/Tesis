import torch


class MaxPool(torch.nn.Module):
    def __init__(self) -> None:
        super(MaxPool, self).__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.max(dim=-1, keepdim=True)[0]
