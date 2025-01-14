import torch
from LIM.data.structures.cloud import Cloud


class KNNGraph(torch.nn.Module):
    knn: int

    def __init__(self, knn: int) -> None:
        super(KNNGraph, self).__init__()
        self.knn = knn

    def __repr__(self) -> str:
        return f"KNNGraph({self.knn})"

    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = cloud.features.squeeze(-1)
        B, C, N = cloud.shape
        print(cloud.shape, cloud.features.shape)
        dist = self._square_distance(cloud.tensor.transpose(1, 2), cloud.tensor.transpose(1, 2))
        print(dist.shape)

        idx = dist.topk(k=self.knn + 1, dim=-1, largest=False, sorted=True)
        print(idx)
        idx = idx[1]
        idx = idx[:, :, 1:]
        idx = idx.unsqueeze(1).repeat(1, C, 1, 1)

        all_feats = cloud.features.unsqueeze(2).repeat(1, 1, N, 1)
        neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)

        cloud.features = cloud.features.unsqueeze(-1).repeat(1, 1, 1, self.knn)
        cloud.features = torch.cat((cloud.features, neighbor_feats - cloud.features), dim=1)
        return cloud

    def _square_distance(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = -2 * torch.matmul(source, target.permute(0, 2, 1))
        dist += torch.sum(source**2, dim=-1)[:, :, None]
        dist += torch.sum(target**2, dim=-1)[:, None, :]
        dist = torch.clamp(dist, min=1e-12, max=None)
        return dist
