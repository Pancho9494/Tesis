import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from enum import Enum


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in: int, size_out: Optional[int] = None, size_h: Optional[int] = None) -> None:
        super(ResnetBlockFC, self).__init__()
        size_out = size_out if size_out is not None else size_in
        size_h = size_h if size_h is not None else min(size_in, size_out)

        self.layers = torch.nn.ModuleList([nn.Linear(size_in, size_h), nn.Linear(size_h, size_out)])
        self.actvn = nn.ReLU()
        self.shortcut = nn.Linear(size_in, size_out, bias=False) if size_in != size_out else None

        nn.init.zeros_(self.layers[1].weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(input) if self.shortcut is not None else input
        for layer in self.layers:
            input = layer(self.actvn(input))
        return input + shortcut


class LocalDecoder(nn.Module):
    class SampleModes(Enum):
        BILINEAR = "bilinear"

        def __str__(self) -> str:
            return f"{self.name.lower()}"

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        n_blocks: int,
        leaky: bool = False,
        sample_mode: SampleModes = SampleModes.BILINEAR,
        padding: float = 0.1,
        d_dim: Optional[int] = None,
    ):
        super(LocalDecoder, self).__init__()
        self.latent_dim = latent_dim if d_dim is None else d_dim
        self.n_blocks = n_blocks

        self.fc_c = nn.ModuleList([nn.Linear(self.latent_dim, hidden_size) for i in range(n_blocks)])

        self.fc_p = nn.Linear(3, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, 1)

        self.actvn = F.relu if (not leaky) else lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.th = nn.Tanh()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def sample_feature_grid(self, points: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        p_nor = self._normalize_3d_coordinate(points.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        latent_vector = (
            F.grid_sample(
                input=latent_vector, grid=vgrid, mode=self.sample_mode.value, padding_mode="border", align_corners=True
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return latent_vector.transpose(1, 2)

    def forward(self, points: torch.Tensor, feature_grid: torch.Tensor) -> torch.Tensor:
        c = self.sample_feature_grid(points, feature_grid)
        points = points.float()
        net = self.fc_p(points)
        for i in range(self.n_blocks):
            if self.latent_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        out = self.th(out)
        out = torch.abs(out)  # scannet label is udf
        return out

    def _normalize_3d_coordinate(self, points: torch.Tensor, padding: float = 0.1):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        """

        p_nor = points / (1 + padding + 10e-4)  # (-0.5, 0.5)
        p_nor = p_nor + 0.5  # range (0, 1)
        p_nor = torch.clamp(p_nor, min=0.0, max=1 - 10e-4)
        return p_nor
