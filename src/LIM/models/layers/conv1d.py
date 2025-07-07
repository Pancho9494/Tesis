import torch
from LIM.data.structures import PCloud
from LIM.models.layers import BatchNorm
from config.config import settings


class Bias(torch.nn.Module):
    in_dim: int
    device: torch.device

    def __init__(self, in_dim: int) -> None:
        super(Bias, self).__init__()
        self.device = torch.device(settings.DEVICE)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.parameter.Parameter(
            torch.zeros(self.in_dim, dtype=torch.float32, device=self.device), requires_grad=True
        )


class Conv1D(torch.nn.Module):
    """
    Called UnaryBlock in the original code

    TODO: Now this weird name is giving me trouble because we need an actual torch.nn.Conv1d adapter for the
    CrossAttention module. For now I'll just name the new one as adapter

    """

    in_dim: int
    out_dim: int
    with_batch_norm: bool
    with_leaky_relu: bool

    def __init__(self, in_dim: int, out_dim: int, with_batch_norm: bool, with_leaky_relu: bool) -> None:
        super(Conv1D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.with_batch_norm = with_batch_norm
        self.with_leaky_relu = with_leaky_relu
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim, bias=False),
            BatchNorm(in_dim=out_dim, momentum=0.02) if with_batch_norm else Bias(in_dim=out_dim),
            torch.nn.LeakyReLU(negative_slope=0.1) if with_leaky_relu else torch.nn.Identity(),
        )

    def __repr__(self) -> str:
        out = f"Conv1D(in_dim: {self.in_dim}, out_dim: {self.out_dim}"
        out += ", batch_norm" if self.with_batch_norm else ""
        out += ", leaky_relu" if self.with_leaky_relu else ""
        out += ")"
        return out

    def forward(self, cloud: PCloud) -> PCloud:
        cloud.features = self.layers(cloud.features)
        return cloud


class Conv1DAdapter(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool
    _conv1dAdapter: torch.nn.Module

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, debug_mode: bool = False
    ) -> None:
        super(Conv1DAdapter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self._conv1dAdapter = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)

        self.debug_mode = debug_mode

    def __repr__(self) -> str:
        out = f"Conv1DAdapter(in_channels: {self.in_channels}, out_channels: {self.out_channels}, "
        out += f"kernel_size: {self.kernel_size}), bias: {self.bias})"
        return out

    def forward(self, cloud: PCloud) -> PCloud:
        if self.debug_mode:
            cloud.features = cloud.features.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        cloud.features = self._conv1dAdapter(cloud.features)
        if self.debug_mode:
            cloud.features = cloud.features.transpose(1, 2).squeeze(0)
        return cloud
