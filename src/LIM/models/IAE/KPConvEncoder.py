import numpy as np
import torch
from submodules.GeoTransformer.geotransformer.modules.kpconv.modules import (
    ConvBlock,
    ResidualBlock,
    UnaryBlock,
    LastUnaryBlock,
    nearest_upsample,
)

import submodules.KPConv.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import submodules.KPConv.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from LIM.data.structures.cloud import Cloud
from typing import Tuple, List


class KPConvFPN(torch.nn.Module):
    __device: torch.device

    def __init__(
        self, inDim: int, outDim: int, iniDim: int, kerSize: int, iniRadius: float, iniSigma: float, groupNorm: int
    ):
        # self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = torch.device("cpu")
        super(KPConvFPN, self).__init__()
        self.latent_dim = outDim
        self.blocks = [
            [
                ConvBlock(inDim, iniDim, kerSize, iniRadius, iniSigma, groupNorm).to(self.__device),
                ResidualBlock(iniDim, 2 * iniDim, kerSize, iniRadius, iniSigma, groupNorm),
            ],
            [
                ResidualBlock(2 * iniDim, 2 * iniDim, kerSize, 1 * iniRadius, 1 * iniSigma, groupNorm, strided=True),
                ResidualBlock(2 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm),
                ResidualBlock(4 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm),
            ],
            [
                ResidualBlock(4 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm, strided=True),
                ResidualBlock(4 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm),
                ResidualBlock(8 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm),
            ],
            [
                ResidualBlock(8 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm, strided=True),
                ResidualBlock(8 * iniDim, 16 * iniDim, kerSize, 8 * iniRadius, 8 * iniSigma, groupNorm),
                ResidualBlock(16 * iniDim, 16 * iniDim, kerSize, 8 * iniRadius, 8 * iniSigma, groupNorm),
            ],
            [
                UnaryBlock(24 * iniDim, 8 * iniDim, groupNorm),
                LastUnaryBlock(12 * iniDim, outDim),
            ],
        ]

        self.blocks = [[layer.to(self.__device) for layer in block] for block in self.blocks]

    def forward(self, cloud: Cloud) -> torch.Tensor:
        features = cloud.features
        a = torch.zeros_like(features.cpu()).to(self.__device)
        print(a.device, features.device)

        voxel_size = 0.025
        radius = 2.5 * voxel_size
        subsamples, neighbors = self._subsample_and_neighbors(cloud, voxel_size, radius)
        subsample_neighbors, upsample_neighbors = self._sub_up_neighbors(subsamples, radius * voxel_size)

        residuals = {0: features}
        residuals[0] = self.blocks[0][0](residuals[0], subsamples[0], subsamples[0], neighbors[0])
        residuals[0] = self.blocks[0][1](residuals[0], subsamples[0], subsamples[0], neighbors[0])

        residuals[1] = self.blocks[1][0](residuals[0], subsamples[1], subsamples[0], subsample_neighbors[0])
        residuals[1] = self.blocks[1][1](residuals[1], subsamples[1], subsamples[1], neighbors[1])
        residuals[1] = self.blocks[1][2](residuals[1], subsamples[1], subsamples[1], neighbors[1])

        residuals[2] = self.blocks[2][0](residuals[1], subsamples[2], subsamples[1], subsample_neighbors[1])
        residuals[2] = self.blocks[2][1](residuals[2], subsamples[2], subsamples[2], neighbors[2])
        residuals[2] = self.blocks[2][2](residuals[2], subsamples[2], subsamples[2], neighbors[2])

        residuals[3] = self.blocks[3][0](residuals[2], subsamples[3], subsamples[2], subsample_neighbors[2])
        residuals[3] = self.blocks[3][1](residuals[3], subsamples[3], subsamples[3], neighbors[3])
        residuals[3] = self.blocks[3][2](residuals[3], subsamples[3], subsamples[3], neighbors[3])

        latent_vectors = {3: residuals[3]}
        latent_vectors[2] = nearest_upsample(latent_vectors[3], upsample_neighbors[2])
        latent_vectors[2] = torch.cat([latent_vectors[2], residuals[2]], dim=1)
        latent_vectors[2] = self.blocks[4][0](latent_vectors[2])

        latent_vectors[1] = nearest_upsample(latent_vectors[2], upsample_neighbors[1])
        latent_vectors[1] = torch.cat([latent_vectors[1], residuals[1]], dim=1)
        latent_vectors[1] = self.blocks[4][1](latent_vectors[1])

        features = [latent_vectors[stage] for stage in reversed(sorted(latent_vectors))]
        # cloud.features = latent_vectors[1]
        return latent_vectors[1]

    def _subsample_and_neighbors(self, cloud: Cloud, vox_size: float = 0.025, radius: float = 2.5) -> Tuple[List, List]:
        """
        Computes the grid subsampling and neighbor radius search for each layer in self.blocks

        Args:
            cloud: The input point cloud
            vox_size: The desired voxel size for downsampling at the first layer.
            radius: The desired radius for neighbor searching at the first layer.

        Returns:
            Tuple[List, List]: A pair of lists containing the outputs of each layer, as tensors
        """
        subsamples = []
        neighbors = []
        radius *= vox_size
        for stage in range(len(self.blocks)):
            points = cpp_subsampling.subsample(points=cloud.arr, sampleDl=vox_size, verbose=0)
            subsamples.append(torch.tensor(points, device=self.__device))
            neighbors_indices = cpp_neighbors.batch_query(
                queries=points,
                supports=points,
                q_batches=np.array([len(points)], dtype=np.int32),
                s_batches=np.array([len(points)], dtype=np.int32),
                radius=radius,
            )
            neighbors.append(torch.tensor(neighbors_indices, device=self.__device))
            vox_size *= 2
            radius *= 2

        return subsamples, neighbors

    def _sub_up_neighbors(self, subsamples: List[torch.Tensor], radius: float) -> Tuple[List, List]:
        """
        Searches for neighbors between the layers in self.blocks, i.e. the closests points each time we subsample or
        upsample.

        Args:
            subsamples: The outputs of the grid subsampling in each layer in self.blocks
            radius: The radius used to get the subsamples at the first layer

        Returns:
            Tuple[List, List]: A pair of lists containing the indices of the neighbors of each layer, as tensors
        """
        subsample_neighbors = []
        upsample_neighbors = []

        for idx in range(len(self.blocks) - 1):
            current_points = subsamples[idx]
            subsample = subsamples[idx + 1]

            sub_indices = cpp_neighbors.batch_query(
                queries=subsample.cpu().numpy(),
                supports=current_points.cpu().numpy(),
                q_batches=np.array([len(subsample)], dtype=np.int32),
                s_batches=np.array([len(current_points)], dtype=np.int32),
                radius=radius,
            )
            subsample_neighbors.append(torch.tensor(sub_indices, device=self.__device))

            up_indices = cpp_neighbors.batch_query(
                queries=current_points.cpu().numpy(),
                supports=subsample.cpu().numpy(),
                q_batches=np.array([len(current_points)], dtype=np.int32),
                s_batches=np.array([len(subsample)], dtype=np.int32),
                radius=radius * 2,
            )
            upsample_neighbors.append(torch.tensor(up_indices, device=self.__device))

            radius *= 2

        return subsample_neighbors, upsample_neighbors
