import torch
import numpy as np
from LIM.data.structures.cloud import Cloud


class Downsample(torch.nn.Module):
    n_points: int

    def __init__(self, n_points: int) -> None:
        self.n_points = n_points

    def forward(self, input: Cloud) -> Cloud:
        """
        Applies to every tensor stored in the input: coordinates, features, colors and normals
        """
        return input.downsample(self.n_points, Cloud.DOWNSAMPLE_MODE.RANDOM)


class BreakSymmetry(torch.nn.Module):
    std_dev: float

    def __init__(self, std_dev: float) -> None:
        self.std_dev = std_dev

    def forward(self, input: Cloud) -> Cloud:
        """
        Applies only to the coordinates, not the features
        """
        input.pcd.positions += self.std_dev * torch.randn(input.tensor.shape, device=input.device)
        return input


class CenterZRandom(torch.nn.Module):
    ratio: float

    def __init__(self, base_ratio: float) -> None:
        self.ratio = base_ratio * torch.rand(1).item()

    def forward(self, input: Cloud) -> Cloud:
        arr = input.arr
        random_ratio = 0.5 * np.random.random()
        min_x = arr[:, 0].min()
        max_x = arr[:, 0].max()

        min_y = arr[:, 1].min()
        max_y = arr[:, 1].max()

        remove_size_x = (max_x - min_x) * random_ratio
        remove_size_y = (max_y - min_y) * random_ratio

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        start_x = center_x - (remove_size_x / 2)
        start_y = center_y - (remove_size_y / 2)

        crop_x_idx = np.where((arr[:, 0] < (start_x + remove_size_x)) & (arr[:, 0] > start_x))[0]
        crop_y_idx = np.where((arr[:, 1] < (start_y + remove_size_y)) & (arr[:, 1] > start_y))[0]

        crop_idx = np.intersect1d(crop_x_idx, crop_y_idx)

        valid_mask = np.ones(len(arr), dtype=bool)
        valid_mask[crop_idx] = 0

        input.pcd.positions = input.pcd.positions[valid_mask]
        input.features = input.features[valid_mask]
        return input


class Noise(torch.nn.Module):
    noise: float

    def __init__(self, noise: float) -> None:
        self.noise = noise

    def forward(self, input: Cloud) -> Cloud:
        input.pcd.positions += self.noise * torch.rand_like(input.pcd.positions)
        return input
