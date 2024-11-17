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
    def __init__(self, knn: int = 20, emb_dims: int = 1024, latent_dim: int = 128, padding: float = 0.1) -> None:
        super(DGCNN, self).__init__()
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LATENT_DIM = latent_dim
        self.PADDING = padding
        self.KNN = knn
        self.GRID_RESOLUTION = 32

        NUM_DIMS = 3
        self.layers = [
            Sequential(
                Conv2d(in_channels=NUM_DIMS * 2, out_channels=64, kernel_size=1, bias=False),
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

        features = self._get_graph_feature(features)
        features = self.layers[0](features)
        features = self.layers[1](features)
        features1 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features1)
        features = self.layers[2](features)
        features = self.layers[3](features)
        features2 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features2)
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

    def _get_graph_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            x [BATCH_SIZE, NUM_DIMS, NUM_POINTS]: _description_
            k (int, optional): _description_. Defaults to 20.

        Returns:
            torch.Tensor [BATCH_SIZE, 2 * NUM_DIMS, NUM_POINTS, K]:
        """
        BATCH_SIZE, NUM_POINTS = x.size(0), x.size(2)
        x = x.view(BATCH_SIZE, -1, NUM_POINTS)
        idx = self._knn(x) + torch.arange(0, BATCH_SIZE, device=self.__device).view(-1, 1, 1) * NUM_POINTS
        idx = idx.view(-1)

        NUM_DIMS = x.size(1)

        x = x.transpose(2, 1).contiguous()
        feature = x.view(BATCH_SIZE * NUM_POINTS, -1)[idx, :]
        feature = feature.view(BATCH_SIZE, NUM_POINTS, self.KNN, NUM_DIMS)
        x = x.view(BATCH_SIZE, NUM_POINTS, 1, NUM_DIMS).repeat(1, 1, self.KNN, 1)

        return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    def _knn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the K nearest neighbors for each point in the input tensor

        Args:
            x [BATCH_SIZE, NUM_DIMS, NUM_POINTS]: The input tensor

        Returns:
            torch.Tensor [BATCH_SIZE, NUM_DIMS, K]: The K nearest neighbors
        """
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        return pairwise_distance.topk(k=self.KNN, dim=-1)[1]

    def _generate_grid_features(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregates the feature tensor into the unit cube grid around the input tensor

        Args:
            points [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: _description_
            features [BATCH_SIZE, NUM_POINTS, LATENT_DIM]: _description_

        Returns:
            torch.Tensor [BATCH_SIZE, LATENT_DIM, GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION]:
        """
        points_nor = self._normalize_3d_coordinate(points.clone(), padding=self.PADDING)
        index = self._coordinate2index(points_nor)
        feature_grid = features.new_zeros(points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION**3)
        features = features.permute(0, 2, 1)
        feature_grid = torch_scatter.scatter_mean(features, index, out=feature_grid)
        feature_grid = feature_grid.reshape(
            points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION, self.GRID_RESOLUTION, self.GRID_RESOLUTION
        )
        return feature_grid

    def _normalize_3d_coordinate(self, points: torch.Tensor, padding: float = 0.1) -> torch.Tensor:
        """
        Normalize coordinates to [0, 1] for unit cube experiments.

        Args:
            points [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: point
            padding: conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        Returns:
            torch.Tensor [BATCH_SIZE, NUM_POINTS, NUM_DIMS]:
        """

        p_nor = points / (1 + padding + 10e-4)  # (-0.5, 0.5)
        p_nor = p_nor + 0.5  # range (0, 1)
        return torch.clamp(p_nor, min=0.0, max=1 - 10e-4)

    def _coordinate2index(self, coordinates: torch.Tensor):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            coordinates [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: ...
        Returns:
            torch.Tensor [BATCH_SIZE, 1, NUM_POINTS]:
        """
        coordinates = (coordinates * self.GRID_RESOLUTION).long()
        index = coordinates[:, :, 0] + self.GRID_RESOLUTION * (
            coordinates[:, :, 1] + self.GRID_RESOLUTION * coordinates[:, :, 2]
        )
        index = index[:, None, :]
        return index
