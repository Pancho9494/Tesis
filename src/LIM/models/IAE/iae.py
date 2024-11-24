import sys
from from_root import from_root

# I very much don't like having to do this, but import shenanigans won
sys.path.append(f"{from_root()}/src/submodules/GeoTransformer")
from LIM.models.IAE.decoder import LocalDecoder
from LIM.data.structures.cloud import Cloud
import torch
from config import settings


class IAE(torch.nn.Module):
    __device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, encoder: torch.nn.Module) -> None:
        super(IAE, self).__init__()
        print(settings.MODEL)
        self.encoder = encoder
        self.decoder = LocalDecoder(
            latent_dim=settings.MODEL.LATENT_DIM,
            hidden_size=settings.MODEL.DECODER.HIDDEN_SIZE,
            n_blocks=settings.MODEL.DECODER.N_BLOCKS,
            leaky=False,
            sample_mode=LocalDecoder.SampleModes.BILINEAR,
            padding=settings.MODEL.DECODER.PADDING,
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
