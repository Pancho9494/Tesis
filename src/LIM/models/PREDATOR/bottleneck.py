import torch
import copy
from typing import Tuple
from LIM.models.PREDATOR import EdgeConv
from LIM.data.structures.cloud import Cloud


class GNN(torch.nn.Module):
    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        super(GNN, self).__init__()
        self.edgeconv1 = EdgeConv(in_dim=2 * feature_dim, out_dim=feature_dim, maxPool=True, knn=10)
        self.edgeconv2 = EdgeConv(in_dim=2 * feature_dim, out_dim=2 * feature_dim, maxPool=True, knn=10)
        self.edgeconv3 = EdgeConv(in_dim=4 * feature_dim, out_dim=feature_dim, maxPool=False)

    def forward(self, cloud: Cloud) -> Cloud:
        cloud.features = cloud.features.unsqueeze(-1)
        cloud = self.edgeconv1(x0 := cloud)
        cloud = self.edgeconv2(x1 := cloud)
        cloud.features = torch.cat((x0.features, x1.features, cloud.features), dim=1)
        cloud = self.edgeconv3(x3 := cloud)
        cloud.features = x3.features.view(cloud.shape[0], -1, cloud.shape[2])
        return cloud


class CrossAttention(torch.nn.Module):
    dim: int  # Dimension of each attention head
    num_heads: int  # Number of attention heads

    def __init__(self, num_heads: int, feature_dim: int) -> None:
        super(CrossAttention, self).__init__()
        assert feature_dim % num_heads == 0, f"feature_dim must be divisible by num_heads in CrossAttention, \
            got {feature_dim} % {num_heads} = {feature_dim % num_heads} != 0"
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        self.proj = torch.nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])
        self.mlp = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(in_channels=2 * feature_dim, out_channels=2 * feature_dim, kernel_size=1, bias=True),
                torch.nn.InstanceNorm1d(num_features=2 * feature_dim),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=2 * feature_dim, out_channels=feature_dim, kernel_size=1, bias=True),
            ]
        )

    def forward(self, source: Cloud, target: Cloud) -> Tuple[Cloud, Cloud]:
        src_attention = self._pipeline(source.features, target.features)
        source.features += self.mlp(torch.cat([source.features, src_attention], dim=1))

        tgt_attention = self._pipeline(target.features, source.features)
        target.features += self.mlp(torch.cat([target.features, tgt_attention], dim=1))

        return source, target

    def _pipeline(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        query, key, value = source, target, target
        BATCH_SIZE = query.size(0)
        query, key, value = [
            layer(x).view(BATCH_SIZE, self.dim, self.num_heads, -1) for layer, x in zip(self.proj, (query, key, value))
        ]
        x = self._attention(query, key, value)
        return self.merge(x.contiguous().view(BATCH_SIZE, self.dim * self.num_heads, -1))

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """

        Args:
            query [batch_size, depth, num_heads, seq_length_query]
            key [batch_size, depth, num_heads, seq_length_key]
            value [batch_size, depth, num_heads, seq_len_key]

        Returns:
            torch.Tensor: [batch_size, depth, num_heads, seq_len]
        """
        scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / query.shape[1] ** 0.5
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum("bhnm,bdhm->bdhn", probabilities, value)


class BottleNeck(torch.nn.Module):
    def __init__(self) -> None:
        super(BottleNeck, self).__init__()
        self.self_attention = GNN(feature_dim=256)
        self.cross_attention = CrossAttention(num_heads=4, feature_dim=256)

    def forward(self, source: Cloud, target: Cloud) -> Tuple[Cloud, Cloud]:
        print("===== SELF ATTENTION ========")
        source.features, target.features = self.self_attention(source), self.self_attention(target)
        print("===== CROSS ATTENTION ========")
        source.features, target.features = self.cross_attention(source, target)
        print("===== SELF ATTENTION ========")
        source.features, target.features = self.self_attention(source), self.self_attention(target)

        return source, target
