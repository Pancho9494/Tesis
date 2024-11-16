import torch
from LIM.data.structures.cloud import Cloud
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Sequential,
    Conv1d,
    Conv2d,
    LeakyReLU,
)
import torch_scatter
from submodules.IAE.src.encoder.unet3d import UNet3D


class DGCNN(torch.nn.Module):
    def __init__(self, k: int = 20, emb_dims: int = 1024, latent_dim: int = 128, padding: float = 0.1) -> None:
        super(DGCNN, self).__init__()
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LATENT_DIM = latent_dim
        self.PADDING = padding
        self.K = k  # ?
        self.GRID_RESOLUTION = 32

        self.layers = [
            Sequential(
                Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=1, bias=False),
                BatchNorm2d(64),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
                BatchNorm2d(64),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, bias=False),
                BatchNorm2d(64),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
                BatchNorm2d(64),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, bias=False),
                BatchNorm2d(64),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv1d(in_channels=192, out_channels=emb_dims, kernel_size=1, bias=False),
                BatchNorm1d(emb_dims),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv1d(in_channels=1216, out_channels=512, kernel_size=1, bias=False),
                BatchNorm1d(512),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            Sequential(
                Conv1d(in_channels=512, out_channels=latent_dim, kernel_size=1, bias=False),
                BatchNorm1d(latent_dim),
                LeakyReLU(negative_slope=0.2),
            ).to(self.__device),
            UNet3D(in_channels=256, out_channels=256, num_levels=4, f_maps=32).to(self.__device),
        ]

    def forward(self, cloud: Cloud) -> torch.Tensor:
        features = cloud.tensor.permute(0, 2, 1).contiguous()
        BATCH_SIZE, NUM_DIMS, NUM_POINTS = features.size()

        features = self._get_graph_feature(features, self.K)
        features = self.layers[0](features)
        features = self.layers[1](features)
        features1 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features1, self.K)
        features = self.layers[2](features)
        features = self.layers[3](features)
        features2 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features2, self.K)
        features = self.layers[4](features)
        features3 = features.max(dim=-1, keepdim=False)[0]
        features = torch.cat((features1, features2, features3), dim=1)

        features = self.layers[5](features)
        features = features.max(dim=-1, keepdim=True)[0]
        features = features.repeat(1, 1, NUM_POINTS)
        features = torch.cat((features, features1, features2, features3), dim=1)

        features = self.layers[6](features)
        features = self.layers[7](features)

        features = features.permute(0, 2, 1).contiguous()
        grid_featrues = self._generate_grid_features(cloud.tensor, features)
        return self.layers[8](grid_featrues)

    def _get_graph_feature(self, x: torch.Tensor, k: int = 20) -> torch.Tensor:
        BATCH_SIZE, NUM_POINTS = x.size(0), x.size(2)
        x = x.view(BATCH_SIZE, -1, NUM_POINTS)
        idx = self._knn(x, k=k) + torch.arange(0, BATCH_SIZE, device=self.__device).view(-1, 1, 1) * NUM_POINTS
        idx = idx.view(-1)

        NUM_DIMS = x.size(1)

        x = x.transpose(2, 1).contiguous()
        feature = x.view(BATCH_SIZE * NUM_POINTS, -1)[idx, :]
        feature = feature.view(BATCH_SIZE, NUM_POINTS, k, NUM_DIMS)
        x = x.view(BATCH_SIZE, NUM_POINTS, 1, NUM_DIMS).repeat(1, 1, k, 1)

        return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    def _knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        return pairwise_distance.topk(k=k, dim=-1)[1]

    def _generate_grid_features(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        points_nor = self._normalize_3d_coordinate(points.clone(), padding=self.PADDING)
        index = self._coordinate2index(points_nor, self.GRID_RESOLUTION)
        feature_grid = features.new_zeros(points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION**3)
        features = features.permute(0, 2, 1)
        feature_grid = torch_scatter.scatter_mean(features, index, out=feature_grid)
        feature_grid = feature_grid.reshape(
            points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION, self.GRID_RESOLUTION, self.GRID_RESOLUTION
        )
        return feature_grid

    def _normalize_3d_coordinate(self, points: torch.Tensor, padding: float = 0.1):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        IAE does it like this:
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0

        So there' the max value after is potentially 1, in ours the max value will be always 1-10e-4

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        """

        p_nor = points / (1 + padding + 10e-4)  # (-0.5, 0.5)
        p_nor = p_nor + 0.5  # range (0, 1)
        p_nor = torch.clamp(p_nor, min=0.0, max=1 - 10e-4)
        return p_nor

    def _coordinate2index(self, coordinates: torch.Tensor, resolution: int):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            coordinates (tensor): ...
            resolution (int): ...
        """
        coordinates = (coordinates * resolution).long()
        index = coordinates[:, :, 0] + resolution * (coordinates[:, :, 1] + resolution * coordinates[:, :, 2])
        index = index[:, None, :]
        return index
