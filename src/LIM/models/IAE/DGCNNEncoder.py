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


class DGCNN(torch.nn.Module):
    def __init__(self, knn: int = 20, emb_dims: int = 1024, latent_dim: int = 128) -> None:
        super(DGCNN, self).__init__()
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LATENT_DIM = latent_dim
        self.KNN = knn  # yes

        NUM_DIMS = 3
        self.conv1 = Sequential(
            Conv2d(in_channels=NUM_DIMS * 2, out_channels=64, kernel_size=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = Sequential(
            Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = Sequential(
            Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = Sequential(
            Conv1d(in_channels=192, out_channels=emb_dims, kernel_size=1, bias=False),
            BatchNorm1d(emb_dims),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv7 = Sequential(
            Conv1d(in_channels=1216, out_channels=512, kernel_size=1, bias=False),
            BatchNorm1d(512),
            LeakyReLU(negative_slope=0.2),
        )
        self.conv8 = Sequential(
            Conv1d(in_channels=512, out_channels=self.LATENT_DIM, kernel_size=1, bias=False),
            BatchNorm1d(self.LATENT_DIM),
            LeakyReLU(negative_slope=0.2),
        )

    def forward(self, cloud: Cloud) -> torch.Tensor:
        features = cloud.tensor.permute(0, 2, 1).contiguous()
        BATCH_SIZE, NUM_DIMS, NUM_POINTS = features.size()

        features = self._get_graph_feature(features)
        features = self.conv1(features)
        features = self.conv2(features)
        features1 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features1)
        features = self.conv3(features)
        features = self.conv4(features)
        features2 = features.max(dim=-1, keepdim=False)[0]

        features = self._get_graph_feature(features2)
        features = self.conv5(features)
        features3 = features.max(dim=-1, keepdim=False)[0]

        features = torch.cat((features1, features2, features3), dim=1)

        features = self.conv6(features)
        features = features.max(dim=-1, keepdim=True)[0]
        features = features.repeat(1, 1, NUM_POINTS)
        features = torch.cat((features, features1, features2, features3), dim=1)

        features = self.conv7(features)
        features = self.conv8(features)

        return features.permute(0, 2, 1).contiguous()

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
