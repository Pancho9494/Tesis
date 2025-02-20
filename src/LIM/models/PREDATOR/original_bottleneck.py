import torch
from typing import Any, List
from multimethod import multimethod
from LIM.data.structures import Pair
from debug.decorators import identify_method
import copy


def square_distance(src, dst, normalised=False):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalised:
        dist += 2
    else:
        dist += torch.sum(src**2, dim=-1)[:, :, None]
        dist += torch.sum(dst**2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def get_graph_feature(coords, feats, k=10):
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1, 2), coords.transpose(1, 2))
    idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True)[
        1
    ]  # [B, N, K+1], here we ignore the smallest element as it's the query itself
    idx = idx[:, :, 1:]  # [B, N, K]

    idx = idx.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)  # [B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1, 1, 1, k)
    feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)

    return feats_cat


class SelfAttention(torch.nn.Module):
    def __init__(self, feature_dim, k=10):
        self.feature_dim = feature_dim
        super(SelfAttention, self).__init__()
        self.conv1 = torch.nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = torch.nn.InstanceNorm2d(feature_dim)

        self.conv2 = torch.nn.Conv2d(feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False)
        self.in2 = torch.nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = torch.nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = torch.nn.InstanceNorm2d(feature_dim)

        self.k = k

    def __repr__(self) -> str:
        # return f"SelfAttention(k: {self.k})"
        return f"SelfAttention(in={self.feature_dim * 2}, out={self.feature_dim}, k={self.k})"

    @identify_method
    def forward(self, coords, features):
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = torch.nn.functional.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = torch.nn.functional.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = torch.nn.functional.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(torch.nn.InstanceNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(torch.nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_heads={self.num_heads}, d_model={self.d_model})"

    @identify_method
    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(torch.nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in: {self.feature_dim}, num_heads: {self.num_heads})"

    @identify_method
    def forward(self, source, target):
        message = self.attn(source, target, target)
        return self.mlp(torch.cat([source, message], dim=1))


class BottleneckAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super(BottleneckAdapter, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                SelfAttention(feature_dim=256),
                AttentionalPropagation(feature_dim=256, num_heads=4),
                SelfAttention(feature_dim=256),
            ]
        )
        GNN_FEATS_DIM = 256
        self.proj_gnn = torch.nn.Conv1d(GNN_FEATS_DIM, GNN_FEATS_DIM, kernel_size=1, bias=True)
        self.proj_score = torch.nn.Conv1d(GNN_FEATS_DIM, 1, kernel_size=1, bias=True)
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

    def __repr__(self) -> str:
        return "BottleneckAdapter()"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, pair: Pair) -> Pair:
        pair.mix.points = pair.mix.points.reshape(1, pair.mix.points.shape[1], -1)
        source, target = pair.split()
        for layer in self.layers:
            if isinstance(layer, AttentionalPropagation):
                foo = layer(source.features, target.features)
                source.features = source.features + foo
                target.features = target.features + layer(target.features, source.features)
            elif isinstance(layer, SelfAttention):
                source.features = layer(source.points, source.features)
                target.features = layer(target.points, target.features)
        pair.join()
        feats_c = self.proj_gnn(pair.mix.features)
        scores_c = self.proj_score(feats_c)
        feats_gnn_norm = torch.nn.functional.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)

        len_src_c = len(pair.source.first.points)
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(torch.nn.functional.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(
            torch.nn.functional.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c
        )
        scores_saliency = torch.cat((s1, s2), dim=0)

        pair.mix.features = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)
        return pair
