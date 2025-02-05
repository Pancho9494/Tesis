import torch
from LIM.data.structures.cloud import Cloud
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

np.random.seed(42)


class TF(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super(TF, self).__init__()

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def forward(self, input: Cloud) -> Cloud: ...


class BreakSymmetry(TF):
    std_dev: float

    def __init__(self, std_dev: float) -> None:
        super(BreakSymmetry, self).__init__()
        self.std_dev = std_dev

    def __repr__(self) -> str:
        return f"BreakSymmetry(std_dev={self.std_dev:0.4f})"

    def forward(self, input: Cloud) -> Cloud:
        """
        Applies only to the coordinates, not the features
        """
        input.tensor = torch.add(
            input.tensor, torch.mul(self.std_dev, torch.rand(input.tensor.shape, device=input.device))
        )
        return input


class CenterZRandom(TF):
    ratio: float

    def __init__(self, base_ratio: float) -> None:
        super(CenterZRandom, self).__init__()
        self.ratio = base_ratio * torch.rand(1).item()

    def __repr__(self) -> str:
        return f"CenterZRandom(ratio={self.ratio:0.4f})"

    def forward(self, input: Cloud) -> Cloud:
        """
        Marks the points to be deleted as nan, then they are removed in downsample, so this transformation
        must be called after dowsnample
        """
        BATCH_SIZE, N_POINTS, N_DIM = input.shape
        if BATCH_SIZE > 1:
            return input
        points = input.tensor
        random_ratio = 0.5 * torch.rand(BATCH_SIZE).to(input.device)

        min_x, _ = torch.min(points[:, :, 0], dim=1)
        max_x, _ = torch.max(points[:, :, 0], dim=1)
        min_y, _ = torch.min(points[:, :, 1], dim=1)
        max_y, _ = torch.max(points[:, :, 1], dim=1)

        min_x = min_x.to(input.device)
        max_x = max_x.to(input.device)
        min_y = min_y.to(input.device)
        max_y = max_y.to(input.device)

        remove_size_x, remove_size_y = (max_x - min_x) * random_ratio, (max_y - min_y) * random_ratio
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        start_x, start_y = center_x - (remove_size_x / 2), center_y - (remove_size_y / 2)

        start_x, start_y = start_x.unsqueeze(1), start_y.unsqueeze(1)
        remove_size_x = remove_size_x.unsqueeze(1)
        remove_size_y = remove_size_y.unsqueeze(1)

        mask_x = (points[:, :, 0] > start_x) & (points[:, :, 0] < (start_x + remove_size_x))
        mask_y = (points[:, :, 1] > start_y) & (points[:, :, 1] < (start_y + remove_size_y))
        crop_mask = mask_x & mask_y

        input.tensor[crop_mask] = float("nan")
        input.features[crop_mask] = float("nan")

        return input


class Downsample(TF):
    n_points: int

    def __init__(self, n_points: int) -> None:
        super(Downsample, self).__init__()
        self.n_points = n_points

    def __repr__(self) -> str:
        return f"Downsample(n_points={self.n_points})"

    def forward(self, input: Cloud) -> Cloud:
        """
        Removes points with nan values and then downsamples to the desired size
        Applies to every tensor stored in the input: coordinates, features, colors and normals
        """

        BATCH_SIZE, N_POINTS, N_DIM = input.shape
        points = []
        features = []
        for idx in range(BATCH_SIZE):
            batch_p = input.tensor[idx]
            batch_f = input.features[idx]
            points.append(batch_p[~torch.isnan(batch_p).any(dim=1)])
            features.append(batch_f[~torch.isnan(batch_f).any(dim=1)])

        input.tensor = torch.stack(points, dim=0)
        input.features = torch.stack(features, dim=0)

        return input.downsample(self.n_points, Cloud.DOWNSAMPLE_MODE.RANDOM)


class Noise(TF):
    noise: float

    def __init__(self, noise: float) -> None:
        super(Noise, self).__init__()
        self.noise = noise

    def __repr__(self) -> str:
        return f"Noise(noise={self.noise:0.4f})"

    def forward(self, input: Cloud) -> Cloud:
        input.tensor = torch.add(input.tensor, torch.mul(self.noise, torch.rand_like(input.tensor)))
        return input


def transform_factory(inputs: Dict[str, Dict[str, Any]]) -> List[torch.nn.Module]:
    if not inputs:
        return []
    mappings: Dict[str, TF] = {
        "CENTERZRANDOM": CenterZRandom,
        "DOWNSAMPLE": Downsample,
        "NOISE": Noise,
        "BREAKSYMMETRY": BreakSymmetry,
    }

    out = []
    for name, kwargs in inputs.items():
        func = mappings[name]
        kwargs = {k.lower(): v for k, v in kwargs.items()}
        out.append(func(**kwargs))

    return out
