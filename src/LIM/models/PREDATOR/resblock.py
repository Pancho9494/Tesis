import torch
from LIM.models.PREDATOR import Conv1D, KPConv


class ResBlock_A(torch.nn.Module):
    in_dim: int

    def __init__(self, in_dim: int) -> None:
        super(ResBlock_A, self).__init__()
        self.in_dim = in_dim
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.layers = torch.nn.ModuleList(
            [
                Conv1D(in_dim=in_dim, out_dim=in_dim // 2, with_batch_norm=True, with_leaky_relu=True),
                KPConv(in_dim=in_dim // 2, out_dim=in_dim // 2, KP_radius=0.06, KP_extent=0.05, n_kernel_points=15),
                self.leaky_relu,
                Conv1D(in_dim=in_dim // 2, out_dim=2 * in_dim, with_batch_norm=True, with_leaky_relu=True),
            ]
        )

    def __repr__(self) -> str:
        return f"ResBlock_A(in_dim: {self.in_dim}, out_dim: {2 * self.in_dim})"

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.layers(batch) + batch)


class ResBlock_B(torch.nn.Module):
    in_dim: int

    def __init__(self, in_dim: int) -> None:
        super(ResBlock_B, self).__init__()
        self.in_dim = in_dim
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.main = torch.nn.ModuleList(
            [
                Conv1D(in_dim=in_dim, out_dim=in_dim // 2, with_batch_norm=True, with_leaky_relu=True),
                KPConv(in_dim=in_dim // 2, out_dim=in_dim // 2, KP_radius=0.06, KP_extent=0.05, n_kernel_points=15),
                self.leaky_relu,
                Conv1D(in_dim=in_dim // 2, out_dim=2 * in_dim, with_batch_norm=True, with_leaky_relu=True),
            ]
        )
        self.shortcut = torch.nn.ModuleList(
            [
                Conv1D(in_dim=in_dim, out_dim=2 * in_dim, with_batch_norm=True, with_leaky_relu=True),
            ]
        )

    def __repr__(self) -> str:
        return f"ResBlock_B(in_dim: {self.in_dim}, out_dim: {2 * self.in_dim})"

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.main(batch) + self.shortcut(batch))
