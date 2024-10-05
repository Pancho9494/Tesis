import numpy as np
import torch
from enum import Enum
from customModels.IAE.encoder import KPConvFPN
from customModels.IAE.decoder import LocalDecoder
from typing import List
from database.point import Point
from database.cloud import Cloud

class IAE(torch.nn.Module):
    class Implicit(Enum):
        SDF = 0,
        UDF = 1
    
        def function(self, value) -> torch.Tensor:
            return torch.abs(value) if (self.value == "UDF") else value
                
    
    def __init__(self, implicit: Implicit) -> None:
        super(IAE, self).__init__()
        self.encoder = KPConvFPN(
            input_dim=1024,
            output_dim=256,
            init_dim=64,
            kernel_size=15,
            init_radius=2.5 * 0.025,
            init_sigma=2.0 * 0.025,
            group_norm=32
        )
        self.decoder = LocalDecoder()
        self.implicit = implicit
        
    def forward(self, cloud: Cloud) -> List:
        """
        "We apply the sliding window strategy and crop each instance into d x d x d cubes, where d = 3.0 m. We randomly
        subsample 10_000 points from each point cloud as the input to the encoder"
        """
        inputs = cloud.arr[np.random.choice(cloud.arr, 10_000, replace=False)]
        query = Point()
        
        latent = self.encoder(
            inputs, 
            {
                'points': [],
                'neighbors': [],
                'subsampling': [],
                'upsampling': []
            }
        )
        implicit = self.implicit.function(self.decoder(query, latent))
        return implicit
        
        
    # s_feats (Tensor): (N, C_in)
    # def forward(self, nFeats: torch.Tensor, data: dict) -> List:
    #     features = self.backbone(nFeats, data)
    #     print(features)
    #     return []
        # return self.decoder(features)