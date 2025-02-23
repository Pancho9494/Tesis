import torch
import copy
from typing import Tuple, Callable
from LIM.models.PREDATOR.blocks import EdgeConv, Conv1DAdapter, InstanceNorm1D, ReLU
from LIM.data.structures import PCloud, Pair
from debug.decorators import identify_method


class GNN(torch.nn.Module):
    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        super(GNN, self).__init__()
        self.feature_dim = feature_dim
        self.edgeconv1 = EdgeConv(in_dim=2 * feature_dim, out_dim=feature_dim, maxPool=True, knn=10)
        self.edgeconv2 = EdgeConv(in_dim=2 * feature_dim, out_dim=2 * feature_dim, maxPool=True, knn=10)
        self.edgeconv3 = EdgeConv(in_dim=4 * feature_dim, out_dim=feature_dim, maxPool=False)

    def __repr__(self) -> str:
        return f"GNN(feature_dim: {self.feature_dim})"

    @identify_method
    def forward(self, cloud: PCloud) -> PCloud:
        def call_and_copy(edgeconv: Callable, cloud: PCloud) -> Tuple[PCloud, PCloud]:
            return edgeconv(cloud), copy.copy(cloud)

        cloud.features = cloud.features.unsqueeze(-1)
        x0 = copy.copy(cloud)
        x1, cloud = call_and_copy(self.edgeconv1, cloud)
        x2, cloud = call_and_copy(self.edgeconv2, cloud)

        x3 = copy.copy(cloud)
        x3.features = torch.cat((x0.features, x1.features, x2.features), dim=1)
        x3 = self.edgeconv3(x3)
        cloud.features = x3.features.view(cloud.shape[0], -1, cloud.shape[2])
        return cloud


class CrossAttention(torch.nn.Module):
    dim: int  # Dimension of each attention head
    num_heads: int  # Number of attention heads

    def __init__(self, num_heads: int, feature_dim: int) -> None:
        super(CrossAttention, self).__init__()
        assert feature_dim % num_heads == 0, (
            f"feature_dim must be divisible by num_heads in CrossAttention, \
            got {feature_dim} % {num_heads} = {feature_dim % num_heads} != 0"
        )
        self.feature_dim = feature_dim
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = Conv1DAdapter(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        self.proj = torch.nn.Sequential(*[copy.deepcopy(self.merge) for _ in range(3)])
        self.mlp = torch.nn.Sequential(
            Conv1DAdapter(in_channels=2 * feature_dim, out_channels=2 * feature_dim, kernel_size=1, bias=True),
            InstanceNorm1D(num_features=2 * feature_dim),
            ReLU(),
            Conv1DAdapter(in_channels=2 * feature_dim, out_channels=feature_dim, kernel_size=1, bias=True),
        )
        torch.nn.init.constant_(self.mlp[-1]._conv1dAdapter.bias, 0.0)

    def __repr__(self) -> str:
        return f"CrossAttention(feature_dim: {self.feature_dim}, dim: {self.dim}, num_heads: {self.num_heads})"

    @identify_method
    def forward(self, source: PCloud, target: PCloud) -> Tuple[PCloud, PCloud]:
        src_attention = self._pipeline(source, target)
        src_attention.features = torch.cat([source.features, src_attention.features], dim=1)
        src_attention = self.mlp(src_attention)
        source.features += src_attention.features

        tgt_attention = self._pipeline(target, source)
        tgt_attention.features = torch.cat([target.features, tgt_attention.features], dim=1)
        tgt_attention = self.mlp(tgt_attention)
        target.features += tgt_attention.features

        return source, target

    @identify_method
    def _pipeline(self, source: PCloud, target: PCloud) -> torch.Tensor:
        query, key, value = copy.copy(source), copy.copy(target), copy.copy(target)
        BATCH_SIZE = query.features.size(0)

        for layer, temp in zip(self.proj, (query, key, value)):
            temp = layer(temp)
            temp.features = temp.features.view(BATCH_SIZE, self.dim, self.num_heads, -1)

        src_attn = copy.copy(source)
        src_attn.features = self._attention(query.features, key.features, value.features)
        src_attn.features = src_attn.features.contiguous().view(BATCH_SIZE, self.dim * self.num_heads, -1)
        return self.merge(src_attn)

    @identify_method
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
        self.pre_self_attention = GNN(feature_dim=256)
        self.post_self_attention = GNN(feature_dim=256)
        self.cross_attention = CrossAttention(num_heads=4, feature_dim=256)
        self.feature_projection = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=True)
        self.score_projection = torch.nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, bias=True)
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

    def __repr__(self) -> str:
        return "BottleNeck()"

    @identify_method
    def forward(self, pair: Pair) -> Pair:
        pair.mix.points = pair.mix.points.reshape(1, pair.mix.points.shape[1], -1)
        pair = pair.split()
        pair.source, pair.target = self.pre_self_attention(pair.source), self.pre_self_attention(pair.target)
        pair.source, pair.target = self.cross_attention(pair.source, pair.target)
        pair.source, pair.target = self.post_self_attention(pair.source), self.post_self_attention(pair.target)

        pair.mix.points = torch.cat(tensors=(pair.source.points, pair.target.points), dim=-1)
        pair.mix.features = torch.cat(tensors=(pair.source.features, pair.target.features), dim=-1)
        return self._merge_features(pair)

    @identify_method
    def _merge_features(self, pair: Pair) -> Pair:
        feats_c = self.feature_projection(pair.mix.features)
        scores_c = self.score_projection(feats_c)
        feats_gnn_norm = torch.nn.functional.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)  # [N, C]
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        split = pair.source.shape[-1]
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:split], feats_gnn_norm[split:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:split], scores_c_raw[split:]

        temperature = torch.exp(self.epsilon) + 0.03
        scores_saliency = torch.cat(
            (
                torch.matmul(
                    torch.nn.functional.softmax(inner_products / temperature, dim=1),
                    tgt_scores_c,
                ),
                torch.matmul(
                    torch.nn.functional.softmax(inner_products.transpose(0, 1) / temperature, dim=1),
                    src_scores_c,
                ),
            ),
            dim=0,
        )
        pair.mix.features = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)
        return pair
