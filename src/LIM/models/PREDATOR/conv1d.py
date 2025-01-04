import torch


class BatchNorm(torch.nn.Module):
    in_dim: int

    def __init__(self, in_dim: int, momentum: float) -> None:
        super(BatchNorm, self).__init__()
        self.in_dim = in_dim
        self._batch_norm = torch.nn.InstanceNorm1d(in_dim, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2).transpose(0, 2)
        x = self._batch_norm(x)
        x = x.transpose(0, 2).squeeze(2)
        return x


class Bias(torch.nn.Module):
    in_dim: int

    def __init__(self, in_dim: int) -> None:
        super(Bias, self).__init__()
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.parameter.Parameter(torch.zeros(self.in_dim, dtype=torch.float32), requires_grad=True)


class Conv1D(torch.nn.Module):
    """
    Called UnaryBlock in the original code

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
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_dim, out_dim, bias=False),
                BatchNorm(in_dim=out_dim, momentum=...) if with_batch_norm else Bias(in_dim=out_dim),
                torch.nn.LeakyReLU(negative_slope=0.1) if with_leaky_relu else torch.nn.Identity(),
            ]
        )

    def __repr__(self) -> str:
        return f"Conv1D(in_dim: {self.in_dim}, out_dim: {self.out_dim}, batch_norm: {self.with_batch_norm}, leaky_relu: {self.with_leaky_relu})"

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.layers(batch)
        return x
