import torch


class KNNGraph(torch.nn.Module):
    knn: int

    def __init__(self, knn: int) -> None:
        super(KNNGraph, self).__init__()
        self.knn = knn

    def __repr__(self) -> str:
        return f"KNNGraph(k:{self.knn})"

    def forward(self, coordinates: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        features = features.squeeze(-1)
        B, C, N = features.size()
        dist = self._square_distance(coordinates.transpose(1, 2), coordinates.transpose(1, 2))

        idx = dist.topk(k=self.knn + 1, dim=-1, largest=False, sorted=True)[1]
        idx = idx[:, :, 1:]
        idx = idx.unsqueeze(1).repeat(1, C, 1, 1)

        all_feats = features.unsqueeze(2).repeat(1, 1, N, 1)
        neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)

        features = features.unsqueeze(-1).repeat(1, 1, 1, self.knn)
        return torch.cat((features, neighbor_feats - features), dim=1)

    def _square_distance(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = -2 * torch.matmul(source, target.permute(0, 2, 1))
        dist += torch.sum(source**2, dim=-1)[:, :, None]
        dist += torch.sum(target**2, dim=-1)[:, None, :]
        dist = torch.clamp(dist, min=1e-12, max=None)
        return dist
