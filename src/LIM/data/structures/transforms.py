import torch
from LIM.data.structures.cloud import Cloud


class BreakSymmetry(torch.nn.Module):
    std_dev: float

    def __init__(self, std_dev: float) -> None:
        super(BreakSymmetry, self).__init__()
        self.std_dev = std_dev

    def __str__(self) -> str:
        return f"BreakSymmetry(std_dev={self.std_dev:0.4f})"

    def forward(self, input: Cloud) -> Cloud:
        """
        Applies only to the coordinates, not the features
        """
        input.tensor = torch.add(
            input.tensor, torch.mul(self.std_dev, torch.rand(input.tensor.shape, device=input.device))
        )
        return input


class CenterZRandom(torch.nn.Module):
    ratio: float

    def __init__(self, base_ratio: float) -> None:
        super(CenterZRandom, self).__init__()
        self.ratio = base_ratio * torch.rand(1).item()

    def __str__(self) -> str:
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
        random_ratio = 0.5 * torch.rand(BATCH_SIZE)

        min_x, _ = torch.min(points[:, :, 0], dim=1)
        max_x, _ = torch.max(points[:, :, 0], dim=1)
        min_y, _ = torch.min(points[:, :, 1], dim=1)
        max_y, _ = torch.max(points[:, :, 1], dim=1)

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


class Downsample(torch.nn.Module):
    n_points: int

    def __init__(self, n_points: int) -> None:
        super(Downsample, self).__init__()
        self.n_points = n_points

    def __str__(self) -> str:
        return f"Downsample(n_points={self.n_points})"

    def forward(self, input: Cloud) -> Cloud:
        """
        Removes points with nan values and then downsamples to the desired size
        Applies to every tensor stored in the input: coordinates, features, colors and normals
        """

        try:  # drop nan rows, i.e. called Downsample without having calleds CenterZRandom
            BATCH_SIZE, N_POINTS, N_DIM = input.shape
            points = []
            features = []
            for idx in range(BATCH_SIZE):
                batch_p = input.tensor[idx]
                batch_f = input.features[idx]
                points.append(batch_p[~torch.isnan(batch_p).any(dim=1)])
                features.append(batch_f[~torch.isnan(batch_f)])

            input.tensor = torch.stack(points, dim=0)
            input.features = torch.stack(features, dim=0)
        except IndexError:  # no nans in the tensors
            pass
        return input.downsample(self.n_points, Cloud.DOWNSAMPLE_MODE.RANDOM)


class Noise(torch.nn.Module):
    noise: float

    def __init__(self, noise: float) -> None:
        super(Noise, self).__init__()
        self.noise = noise

    def __str__(self) -> str:
        return f"Noise(noise={self.noise:0.4f})"

    def forward(self, input: Cloud) -> Cloud:
        input.tensor = torch.add(input.tensor, torch.mul(self.noise, torch.rand_like(input.tensor)))
        return input
