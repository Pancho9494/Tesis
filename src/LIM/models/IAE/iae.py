import sys
from from_root import from_root

# I very much don't like having to do this, but import shenanigans won
sys.path.append(f"{from_root()}/src/submodules/GeoTransformer")
sys.path.append(f"{from_root()}/src/submodules/IAE")
from LIM.models.IAE.DGCNNEncoder import DGCNN
from LIM.models.IAE.decoder import LocalDecoder
from LIM.data.structures.cloud import Cloud
import torch
from dataclasses import dataclass, field


class IAE(torch.nn.Module):
    # __device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    __device: torch.device = torch.device("cpu")

    @dataclass
    class Params:
        LATENT_DIM: int = field(default=256)
        HIDDEN_SIZE: int = field(default=32)

    def __init__(self) -> None:
        super(IAE, self).__init__()
        self.params = IAE.Params(
            LATENT_DIM=256,
            HIDDEN_SIZE=32,
        )

        self.encoder = DGCNN(
            latent_dim=self.params.LATENT_DIM,
            padding=0.1,
        )

        self.decoder = LocalDecoder(
            latent_dim=self.params.LATENT_DIM,
            hidden_size=self.params.HIDDEN_SIZE,
            n_blocks=5,
            leaky=False,
            sample_mode=LocalDecoder.SampleModes.BILINEAR,
            padding=0.1,
            d_dim=None,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, cloud: Cloud, implicit: Cloud) -> torch.Tensor:
        """
        IAE has three steps:
            1. Generate cloud features with encoder
            2. Reshape features as feature grid
            3. Predict DF with decoder

        Args:
            cloud: The input cloud scan
            implicit: The cloud's implicit representation, i.e. a discretized unit cube whose features are the df values

        Returns:
            torch.Tensor: The predicted DF
        """
        feature_grid = self.encoder(cloud)
        predicted_df = self.decoder(implicit.tensor, feature_grid)
        return predicted_df
