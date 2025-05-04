from LIM.models.IAE.decoder import LocalDecoder
from LIM.data.structures import PCloud
import torch
from config.config import settings
from LIM.models.blocks.unet3d import UNet3D
import torch_scatter
from LIM.models.modelI import Model
from LIM.models.IAE.DGCNNEncoder import DGCNN


class IAE(Model):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super(IAE, self).__init__()
        self.LATENT_DIM = settings.MODEL.LATENT_DIM
        self.PADDING = settings.MODEL.PADDING
        self.GRID_RESOLUTION = settings.MODEL.ENCODER.GRID_RES

        self.encoder = encoder
        self.decoder = LocalDecoder(
            latent_dim=self.LATENT_DIM,
            hidden_size=settings.MODEL.DECODER.HIDDEN_SIZE,
            n_blocks=settings.MODEL.DECODER.N_BLOCKS,
            leaky=False,
            sample_mode=LocalDecoder.SampleModes.BILINEAR,
            padding=self.PADDING,
            d_dim=None,
        )
        self.unet3d = UNet3D(in_channels=self.LATENT_DIM, out_channels=self.LATENT_DIM, num_levels=4, f_maps=32)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, cloud: PCloud, implicit: PCloud) -> torch.Tensor:
        """
        Args:
            cloud: The input cloud scan
            implicit: The cloud's implicit representation, i.e. a discretized unit cube whose features are the df values

        Returns:
            torch.Tensor: The predicted DF
        """
        (pred_cloud, _skip_connections) = self.encoder(cloud)
        cloud.points, implicit.points = cloud.points.reshape((1, -1, 3)), implicit.points.reshape((1, -1, 3))
        pred_cloud.points = pred_cloud.points.reshape((1, -1, 3))
        feature_grid = self._generate_grid_features(pred_cloud.points, pred_cloud.features)
        feature_grid = self.unet3d(feature_grid)
        predicted_df = self.decoder(implicit.points, feature_grid)
        return predicted_df

    def _generate_grid_features(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregates the feature tensor into the unit cube grid around the input tensor

        Args:
            points [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: _description_
            features [BATCH_SIZE, NUM_POINTS, LATENT_DIM]: _description_

        Returns:
            torch.Tensor [BATCH_SIZE, LATENT_DIM, GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION]:
        """
        points_nor = self._normalize_3d_coordinate(points.clone(), padding=self.PADDING)
        index = self._coordinate2index(points_nor)
        feature_grid = features.new_zeros(points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION**3)
        feature_grid = torch_scatter.scatter_mean(features, index, out=feature_grid)
        feature_grid = feature_grid.reshape(
            points.size(0), self.LATENT_DIM, self.GRID_RESOLUTION, self.GRID_RESOLUTION, self.GRID_RESOLUTION
        )
        return feature_grid

    def _normalize_3d_coordinate(self, points: torch.Tensor, padding: float = 0.1) -> torch.Tensor:
        """
        Normalize coordinates to [0, 1] for unit cube experiments.

        Args:
            points [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: point
            padding: conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        Returns:
            torch.Tensor [BATCH_SIZE, NUM_POINTS, NUM_DIMS]:
        """

        p_nor = points / (1 + padding + 10e-4)  # (-0.5, 0.5)
        p_nor = p_nor + 0.5  # range (0, 1)
        return torch.clamp(p_nor, min=0.0, max=1 - 10e-4)

    def _coordinate2index(self, coordinates: torch.Tensor):
        """
        Normalize coordinate to [0, 1] for unit cube experiments.

        Args:
            coordinates [BATCH_SIZE, NUM_POINTS, NUM_DIMS]: ...
        Returns:
            torch.Tensor [BATCH_SIZE, 1, NUM_POINTS]:
        """
        coordinates = (coordinates * self.GRID_RESOLUTION).long()
        index = coordinates[:, :, 0] + self.GRID_RESOLUTION * (
            coordinates[:, :, 1] + self.GRID_RESOLUTION * coordinates[:, :, 2]
        )
        index = index[:, None, :]
        return index
