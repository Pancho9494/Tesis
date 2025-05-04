import torch
from typing import List, Any, Tuple
from LIM.data.structures import PCloud, Pair
from LIM.models.blocks import Conv1D
from LIM.models.blocks.nearestupsample import NearestUpsample
from config.config import settings
from multimethod import multimethod


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.nearest_upsample = NearestUpsample()
        enter_dim = settings.MODEL.LATENT_DIM + 2
        self.inner_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv1D(
                        in_dim=(enter_dim := enter_dim + 2 ** (idx)),
                        out_dim=(enter_dim := enter_dim // 6),
                        with_batch_norm=True,
                        with_leaky_relu=True,
                    )
                )
                for idx in range(7 + settings.MODEL.ENCODER.N_HIDDEN_LAYERS, 7, -1)
            ]
        )

        self.exit = torch.nn.Sequential(
            Conv1D(
                in_dim=(enter_dim := enter_dim + 2 ** (7)),
                out_dim=enter_dim // 6,
                with_batch_norm=False,
                with_leaky_relu=False,
            )
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.FINAL_FEATS_DIM = self.exit[0].out_dim - 2

    def __repr__(self) -> str:
        return "Decoder()"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, cloud: PCloud, skip_connections: List[torch.Tensor]) -> Tuple[PCloud, torch.Tensor, torch.Tensor]:
        for block, skip in zip([*self.inner_layers, self.exit], reversed(skip_connections)):
            cloud = self.nearest_upsample(cloud)
            cloud.features = torch.cat([cloud.features, skip], dim=1)
            cloud = block(cloud)
            cloud._sub.features = cloud.features
            cloud = cloud._sub

        overlap_score = torch.nan_to_num(
            torch.clamp(self.sigmoid(cloud.features[:, self.FINAL_FEATS_DIM]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        saliency_score = torch.nan_to_num(
            torch.clamp(self.sigmoid(cloud.features[:, self.FINAL_FEATS_DIM + 1]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        cloud.features = torch.nn.functional.normalize(
            cloud.features[:, : self.FINAL_FEATS_DIM],
            p=2,
            dim=1,
        )
        return (cloud, overlap_score, saliency_score)

    @multimethod
    def forward(self, pair: Pair, skip_connections: List[torch.Tensor]) -> Tuple[PCloud, torch.Tensor, torch.Tensor]:
        for block, skip in zip([*self.inner_layers, self.exit], reversed(skip_connections)):
            pair.mix = self.nearest_upsample(pair.mix)
            pair.mix.features = torch.cat([pair.mix.features, skip], dim=1)
            pair.mix = block(pair.mix)
            pair.mix._sub.features = pair.mix.features
            pair.mix = pair.mix._sub
            pair.source = pair.source._sub
            pair.target = pair.target._sub

        overlap_score = torch.nan_to_num(
            torch.clamp(self.sigmoid(pair.mix.features[:, self.FINAL_FEATS_DIM]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        saliency_score = torch.nan_to_num(
            torch.clamp(self.sigmoid(pair.mix.features[:, self.FINAL_FEATS_DIM + 1]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pair.mix.features = torch.nn.functional.normalize(pair.mix.features[:, : self.FINAL_FEATS_DIM], p=2, dim=1)
        return (pair, overlap_score, saliency_score)
