import torch
import copy
from typing import Tuple, Callable
from LIM.models.PREDATOR.blocks import EdgeConv, Conv1DAdapter, InstanceNorm1D, ReLU
from LIM.data.structures.cloud import Cloud
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

    @identify_method(after_msg="\n\n")
    def forward(self, cloud: Cloud) -> Cloud:
        def call_and_copy(edgeconv: Callable, cloud: Cloud) -> Tuple[Cloud, Cloud]:
            return edgeconv(cloud), copy.copy(cloud)
        
        cloud.features = cloud.features.unsqueeze(-1)
        x0 = copy.copy(cloud)
        x1, cloud = call_and_copy(self.edgeconv1, cloud)
        x2, cloud = call_and_copy(self.edgeconv2, cloud)
        
        cloud.features = torch.cat((x0.features, x1.features, x2.features), dim=1)
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
        self.merge = Conv1DAdapter(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        self.proj = torch.nn.Sequential(*[copy.deepcopy(self.merge) for _ in range(3)])
        self.mlp = torch.nn.Sequential(
            Conv1DAdapter(in_channels=2 * feature_dim, out_channels=2 * feature_dim, kernel_size=1, bias=True),
            InstanceNorm1D(num_features=2 * feature_dim),
            ReLU(),
            Conv1DAdapter(in_channels=2 * feature_dim, out_channels=feature_dim, kernel_size=1, bias=True),
        )
    def __repr__(self) -> str:
        return f"CrossAttention(dim: {self.dim}, num_heads: {self.num_heads})"

    @identify_method
    def forward(self, source: Cloud, target: Cloud) -> Tuple[Cloud, Cloud]:
        src_attention = self._pipeline(source, target)
        src_attention.features = torch.cat([source.features, src_attention.features], dim=1)
        src_attention = self.mlp(src_attention)
        source.features += src_attention.features

        tgt_attention = self._pipeline(target, source)
        tgt_attention.features = torch.cat([target.features, tgt_attention.features], dim=1)
        tgt_attention = self.mlp(tgt_attention)
        target.features += tgt_attention.features

        return source, target

    def _pipeline(self, source: Cloud, target: Cloud) -> torch.Tensor:
        query, key, value = copy.copy(source), copy.copy(target), copy.copy(target)
        BATCH_SIZE = query.features.size(0)
        
        for layer, temp in zip(self.proj, (query, key, value)):
            temp = layer(temp)
            temp.features = temp.features.view(BATCH_SIZE, self.dim, self.num_heads, -1)
           
        src_attn = copy.copy(source)
        src_attn.features = self._attention(query.features, key.features, value.features)
        src_attn.features = src_attn.features.contiguous().view(BATCH_SIZE, self.dim * self.num_heads, -1)
        return self.merge(src_attn)

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
        self.feature_projection = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=True)
        self.score_projection = torch.nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, bias=True)
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        
    def __repr__(self) -> str:
        return "BottleNeck()"

    @identify_method
    def forward(self, source: Cloud, target: Cloud) -> Tuple[Cloud, Cloud]:
        source.tensor = source.tensor.reshape(1, source.tensor.shape[1], -1)
        target.tensor = target.tensor.reshape(1, target.tensor.shape[1], -1)
        source.features, target.features = (
            source.features.reshape(1, -1, source.features.shape[0]),
            target.features.reshape(1, -1, target.features.shape[0]),
        )
        
        source, target = self.self_attention(source), self.self_attention(target)
        source, target = self.cross_attention(source, target)
        source, target = self.self_attention(source), self.self_attention(target)

        return self._merge_features(source, target)
        
    def _merge_features(self, source: Cloud, target: Cloud) -> Tuple[Cloud, Cloud]:
        def _get_scores(cloud: Cloud) -> torch.Tensor:
            scores = self.score_projection(cloud.features).squeeze(0).transpose(0, 1)
            cloud.features = self.feature_projection(cloud.features).squeeze(0).transpose(0, 1)
            return scores
        
        temperature = torch.exp(self.epsilon) + 0.03
        source_scores, target_scores = _get_scores(source), _get_scores(target)
        inner_product = torch.matmul(source.features, target.features.transpose(0, 1))
        
        source.features = torch.cat(
            [
                source_scores,
                torch.matmul(
                    torch.nn.functional.softmax(inner_product / temperature, dim=1),
                    target_scores,
                ),
                source.features
            ],
            dim=1
        )
        target.features = torch.cat(
            [
                target_scores,
                torch.matmul(
                    torch.nn.functional.softmax(inner_product.transpose(0, 1) / temperature, dim=1), 
                    source_scores,
                ),
                target.features,
            ],
            dim=1,
        )
        
        return source, target