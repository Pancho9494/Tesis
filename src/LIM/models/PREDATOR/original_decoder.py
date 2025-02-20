import torch
from debug.decorators import identify_method
from multimethod import multimethod
from typing import Any, List
from LIM.data.structures import Pair


def gather(x, idx, method=2):
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        ss = idx.size()
        for i, ni in enumerate(ss[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError("Unkown method")


def closest_pool(x, inds):
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    return gather(x, inds[:, 0])


class NearestUpsampleBlock(torch.nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind

        # print(f"\tnearest_upsample has {sum([p.numel() for p in self.parameters()])} parameters")
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @identify_method
    def forward(self, x, batch):
        return closest_pool(x, batch["upsamples"][self.layer_ind - 1])


class BatchNormBlock(torch.nn.Module):
    def __init__(self, in_dim, use_bn, bn_momentum):
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = torch.nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        torch.nn.init.zeros_(self.bias)

    @identify_method
    def forward(self, x):
        if self.use_bn:
            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        if self.use_bn:
            return f"BatchNormBlock(in_feat: {self.in_dim}, momentum: {self.bn_momentum})"
        else:
            return "Bias()"


class UnaryBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    @identify_method
    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        out = f"UnaryBlock(in={self.in_dim}, out={self.out_dim})"
        out += f" -> {self.batch_norm.__repr__()}"
        out += f" -> {self.leaky_relu}" if not self.no_relu else ""
        return out


class LastUnaryBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LastUnaryBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = torch.nn.Linear(in_dim, out_dim, bias=False)
        return

    @identify_method
    def forward(self, x, batch=None):
        x = self.mlp(x)
        return x

    def __repr__(self):
        return "LastUnaryBlock(in_feat: {:d}, out_feat: {:d})".format(self.in_dim, self.out_dim)


class DecoderAdapter(torch.nn.Module):
    """
    self.decoder_blocks
    ModuleList(
    (0): NearestUpsampleBlock(layer: 3 -> 2)
    (1): UnaryBlock(in=770, out=129) -> BatchNormBlock(in_feat: 129, momentum: 0.02) -> LeakyReLU(negative_slope=0.1)
    (2): NearestUpsampleBlock(layer: 2 -> 1)
    (3): UnaryBlock(in=385, out=64) -> BatchNormBlock(in_feat: 64, momentum: 0.02) -> LeakyReLU(negative_slope=0.1)
    (4): NearestUpsampleBlock(layer: 1 -> 0)
    (5): LastUnaryBlock(in_feat: 192, out_feat: 34)
    )
    """

    def __init__(self) -> None:
        super(DecoderAdapter, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                NearestUpsampleBlock(layer_ind=3),
                UnaryBlock(in_dim=770, out_dim=129, use_bn=True, bn_momentum=0.02, no_relu=False),
                NearestUpsampleBlock(layer_ind=2),
                UnaryBlock(in_dim=385, out_dim=64, use_bn=True, bn_momentum=0.02, no_relu=False),
                NearestUpsampleBlock(layer_ind=1),
                LastUnaryBlock(in_dim=192, out_dim=34),
            ]
        )
        self.FINAL_FEATS_DIM = 32

    def __repr__(self) -> str:
        return "DecoderAdapter()"

    def regular_score(self, score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, pair: Pair, skip_connections: List[torch.Tensor]) -> Pair:
        sigmoid = torch.nn.Sigmoid()

        batch = pair.to_legacy()
        x = pair.mix.features
        for idx, block in enumerate(self.layers):
            if isinstance(block, (UnaryBlock, LastUnaryBlock)):
                x = torch.cat([x, skip_connections.pop()], dim=1)
            x = block(x, batch)

        pair.overlaps.mix = x[:, self.FINAL_FEATS_DIM]
        pair.saliencies.mix = x[:, self.FINAL_FEATS_DIM + 1]
        pair.mix.features = x[:, : self.FINAL_FEATS_DIM]

        pair.overlaps.mix = torch.clamp(sigmoid(pair.overlaps.mix.view(-1)), min=0, max=1)
        pair.saliencies.mix = torch.clamp(sigmoid(pair.saliencies.mix.view(-1)), min=0, max=1)
        pair.overlaps.mix = self.regular_score(pair.overlaps.mix)
        pair.saliencies.mix = self.regular_score(pair.saliencies.mix)

        pair.mix.features = torch.nn.functional.normalize(pair.mix.features, p=2, dim=1)

        return pair
