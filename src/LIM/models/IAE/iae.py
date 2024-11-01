import sys
from from_root import from_root

# I very much don't like having to do this, but import shenanigans won
sys.path.append(f"{from_root()}/src/submodules/GeoTransformer")
sys.path.append(f"{from_root()}/src/submodules/IAE")
from LIM.models.IAE.encoder import KPConvFPN
from LIM.models.IAE.decoder import LocalDecoder
from LIM.data.structures.cloud import Cloud
import torch
from torch_scatter import scatter_mean
from submodules.IAE.src.encoder.unet3d import UNet3D


class IAE(torch.nn.Module):
    __device: torch.device

    def __init__(self) -> None:
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(IAE, self).__init__()
        self.latent_dim = 256

        self.encoder = KPConvFPN(
            inDim=1,
            outDim=self.latent_dim,
            iniDim=64,
            kerSize=15,
            iniRadius=2.5 * 0.025,
            iniSigma=2.0 * 0.025,
            groupNorm=32,
        )
        self.decoder = LocalDecoder(
            latent_dim=self.latent_dim,
            hidden_size=384,
            n_blocks=5,
            leaky=False,
            sample_mode=LocalDecoder.SampleModes.BILINEAR,
            padding=0.1,
            d_dim=None,
        )

        self.unet3d = UNet3D(num_levels=4, f_maps=32, in_channels=self.latent_dim, out_channels=self.latent_dim)

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
        latent_vector = self.encoder(cloud)

        implicit.downsample(latent_vector.shape[0], mode=Cloud.DOWNSAMPLE_MODE.RANDOM)
        feature_grid = self.__generate_grid_features(implicit.tensor.unsqueeze(0), latent_vector.unsqueeze(0))

        predicted_df = self.decoder(implicit.tensor, feature_grid)
        return predicted_df

    def __generate_grid_features(
        self, points: torch.Tensor, latent_vector: torch.Tensor, grid_resolution: int = 32
    ) -> torch.Tensor:
        p_nor = self.__normalize_3d_coordinate(points.clone(), padding=0.1)
        index = self.__coordinate2index(p_nor, grid_resolution)
        fea_grid = latent_vector.new_zeros(points.size(0), self.latent_dim, grid_resolution**3)
        latent_vector = latent_vector.permute(0, 2, 1)
        fea_grid = scatter_mean(latent_vector, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            points.size(0), self.latent_dim, grid_resolution, grid_resolution, grid_resolution
        )  # sparce matrix (B x 512 x reso x reso)
        fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def __normalize_3d_coordinate(self, points: torch.Tensor, padding: float = 0.1):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        """

        p_nor = points / (1 + padding + 10e-4)  # (-0.5, 0.5)
        p_nor = p_nor + 0.5  # range (0, 1)
        p_nor = torch.clamp(p_nor, min=0.0, max=1 - 10e-4)
        return p_nor

    def __coordinate2index(self, coordinates: torch.Tensor, resolution: int):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            coordinates (tensor): ...
            resolution (int): ...
        """
        coordinates = (coordinates * resolution).long()
        index = coordinates[:, :, 0] + resolution * (coordinates[:, :, 1] + resolution * coordinates[:, :, 2])
        index = index[:, None, :]
        return index
