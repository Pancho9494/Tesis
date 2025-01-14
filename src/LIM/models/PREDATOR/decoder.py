import torch
from typing import List
from LIM.data.structures.cloud import Cloud
from LIM.models.PREDATOR import Conv1D
from LIM.models.PREDATOR.nearestupsample import NearestUpsample


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.block1 = torch.nn.ModuleList(
            [
                NearestUpsample(2),
                Conv1D(in_dim=770, out_dim=129, with_batch_norm=True, with_leaky_relu=True),
            ]
        )
        self.block2 = torch.nn.ModuleList(
            [
                NearestUpsample(1),
                Conv1D(in_dim=385, out_dim=64, with_batch_norm=True, with_leaky_relu=True),
            ]
        )
        self.block3 = torch.nn.ModuleList(
            [
                NearestUpsample(0),
                Conv1D(in_dim=192, out_dim=34, with_batch_norm=False, with_leaky_relu=False),
            ]
        )

    def forward(self, cloud: Cloud, skip_connections: List[torch.Tensor]) -> Cloud:
        for block, skip in zip([self.block1, self.block2, self.block3], skip_connections):
            cloud.features = torch.cat([cloud.features, skip], dim=1)
            cloud.features = block(cloud.features)
        return cloud
