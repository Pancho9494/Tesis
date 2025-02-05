import torch
from typing import List
from LIM.data.structures.cloud import Cloud
from LIM.models.PREDATOR.blocks import Conv1D
from LIM.models.PREDATOR.blocks.nearestupsample import NearestUpsample
from debug.decorators import identify_method


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.nearest_upsample = NearestUpsample()
        self.block1 = torch.nn.Sequential(
            Conv1D(in_dim=770, out_dim=129, with_batch_norm=True, with_leaky_relu=True),
        )
        self.block2 = torch.nn.Sequential(
            Conv1D(in_dim=385, out_dim=64, with_batch_norm=True, with_leaky_relu=True),
        )
        self.block3 = torch.nn.Sequential(
            Conv1D(in_dim=192, out_dim=34, with_batch_norm=False, with_leaky_relu=False),
        )

    def __repr__(self) -> str:
        return "Decoder()"

    @identify_method
    def forward(self, cloud: Cloud, skip_connections: List[torch.Tensor]) -> Cloud:
        for block, skip in zip([self.block1, self.block2, self.block3], reversed(skip_connections)):
            cloud = self.nearest_upsample(cloud)
            cloud.features = torch.cat([cloud.features, skip], dim=1)
            cloud = block(cloud)
            cloud.subpoints.features = cloud.features
            cloud = cloud.subpoints

        if torch.isnan(cloud.features).any():
            print("?")
        return cloud
