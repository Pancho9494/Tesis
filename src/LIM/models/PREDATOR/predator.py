# I don't like using sys.path.append, but given that OverlapPredator is not a package, its kinda hard not to

# from easydict import EasyDict as edict
# from dataclasses import dataclass, field
# from LIM.data.structures.cloud import Cloud

# from submodules.OverlapPredator.models.architectures import KPFCNN
# from submodules.OverlapPredator.lib.utils import load_config
# from submodules.OverlapPredator.configs.models import architectures
# from submodules.OverlapPredator.datasets.dataloader import collate_fn_descriptor
# from submodules.OverlapPredator.lib.benchmark_utils import ransac_pose_estimation

# from models.modelI import ModelI
# from data.pairs import Pairs

# from config import settings
from typing import Tuple, Optional, List

import torch
from LIM.models.PREDATOR import Conv1D, KPConv, ResBlock_A, ResBlock_B, KNNGraph, MaxPool


# @dataclass
# class Config:
#     """
#     Utility class expected in PREDATOR's architecture, these values are originally read from a .yaml
#     """

#     deform_radius: float = field(default=5.0)
#     first_subsampling_dl: float = field(default=0.025)
#     conv_radius: float = field(default=2.5)
#     architecture: list = field(default_factory=list)
#     num_layers: int = field(default=4)
#     n_points: int = field(default=1000)


# class ConvBlock(torch.nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, radius: float) -> None:
#         self.kpconv = ...
#         self.batch_norm = ...
#         self.leaky_relu = ...

#     def forward(self, cloud: Cloud) -> torch.Tensor:
#         q_pts = ...
#         s_pts = ...
#         neighbor_idxs = ...

#         return self.leaky_relu(
#             self.batch_norm(
#                 self.kpconv(q_pts, s_pts, neighbor_idxs, cloud.pcd.points),
#             )
#         )


class Encoder(torch.nn.Module):
    skip_connections: List[torch.Tensor]

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.skip_connections = []
        self.block1 = torch.nn.ModuleList(
            [
                KPConv(in_dim=1, out_dim=64),
                torch.nn.LeakyReLU(negative_slope=0.1),
                ResBlock_B(in_dim=64),
                ResBlock_A(in_dim=128),
            ]
        )
        self.block2 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=128),
                ResBlock_B(in_dim=256),
                ResBlock_A(in_dim=256),
            ]
        )
        self.block3 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=256),
                ResBlock_B(in_dim=512),
                ResBlock_A(in_dim=512),
            ]
        )
        self.block4 = torch.nn.ModuleList(
            [
                ResBlock_B(in_dim=512),
                ResBlock_B(in_dim=1024),
                Conv1D(in_dim=1024, out_dim=256, with_batch_norm=True, with_leaky_relu=True),
            ]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        for block in [self.block1, self.block2, self.block3]:
            for layer in block:
                batch = layer(batch)
            self.skip_connections.append(batch)

        for layer in self.block4:
            batch = layer(batch)

        return batch


class EdgeConv(torch.nn.Module):
    def __init__(self, knn: int, in_dim: int, out_dim: int, with_KNN: bool, maxPool: bool) -> None:
        super(EdgeConv, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.layers = torch.nn.ModuleList(
            [
                KNNGraph(k=knn) if with_KNN else torch.nn.Identity(),
                torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False),
                torch.nn.InstanceNorm2d(num_features=out_dim),
                self.leaky_relu,
                MaxPool() if maxPool else torch.nn.Identity(),
            ]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # TODO: knngraph needs to get coordinates and features in the forward method
        for layer in self.layers:
            batch = layer(batch)
        return batch


class GNN(torch.nn.Module):
    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        super(GNN, self).__init__()
        self.edgeconv1 = EdgeConv(knn=10, in_dim=2 * feature_dim, out_dim=feature_dim, with_KNN=True, maxPool=True)
        self.edgeconv2 = EdgeConv(knn=10, in_dim=2 * feature_dim, out_dim=2 * feature_dim, with_KNN=True, maxPool=True)
        self.edgeconv3 = EdgeConv(knn=10, in_dim=4 * feature_dim, out_dim=feature_dim, with_KNN=False, maxPool=False)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x0 = batch.unsqueeze(-1)
        x1 = self.edgeconv1(x0)
        x2 = self.edgeconv2(x1)
        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = self.edgeconv3(x3)


# class Predator(ModelI):
#     """
#     Wrapper for the PREDATOR model that follows the ModelI interface

#     00   'simple',               KPConv
#     01   'resnetb',              ResnetB
#     02   'resnetb_strided',      ResnetA

#     03   'resnetb',              ResnetB
#     04   'resnetb',              ResnetB
#     05   'resnetb_strided',      ResnetA

#     06   'resnetb',              ResnetB
#     07   'resnetb',              ResnetB
#     08   'resnetb_strided',      ResnetA


#     09   'resnetb',              ResnetB
#     10   'resnetb',              ResnetB
#                                  Conv1D (it's called self.bottle)

#     11   'nearest_upsample',
#     12   'unary',
#     13   'nearest_upsample',
#     14   'unary',
#     15   'nearest_upsample',
#     16   'last_unary'
#     """

#     model: torch.nn.Module
#     config: Config
#     _mode: str
#     _device: torch.device

#     def __init__(self, mode: str) -> None:
#         config = load_config(Path(f"./submodules/OverlapPredator/configs/test/{mode}.yaml"))
#         config["architecture"] = architectures[mode]
#         self.model = KPFCNN(edict(config))
#         self.model.load_state_dict(
#             torch.load(f"./customModels/weights/PREDATOR/{mode}.pth", weights_only=True)["state_dict"],
#         )

#         self.config = Config()
#         self.config.architecture = architectures[mode]
#         self._mode = mode
#         self._device = torch.device(settings.DEVICE)

#     def __repr__(self) -> str:
#         return f"Predator {self._mode}"

#     def __call__(self, pair: Pairs) -> np.ndarray:
#         self.model.eval()
#         with torch.no_grad():
#             wrapper = (
#                 pair.src.arr,  # src_pcd
#                 pair.target.arr,  # tgt_pcd
#                 np.ones_like(pair.src.arr[:, :1]).astype(np.float32),  # src_feats
#                 np.ones_like(pair.target.arr[:, :1]).astype(np.float32),  # tgt_feats
#                 np.ones_like(pair.truth[:3, :3]).astype(np.float32),  # rot
#                 np.ones_like(pair.truth[:, 3]).astype(np.float32),  # trans
#                 torch.ones(1, 2).long(),  # correspondences
#                 pair.src.arr,  # src_raw
#                 pair.target.arr,  # tgt_raw
#                 torch.ones(1),  # samplee
#             )
#             batched_input = collate_fn_descriptor(
#                 [wrapper],
#                 self.config,
#                 neighborhood_limits=[int(np.ceil(4 / 3 * np.pi * (self.config.deform_radius + 1) ** 3))] * 5,
#             )
#             feats, overlaps, saliencies = self.model(batched_input)
#             len_src = batched_input["stack_lengths"][0][0]
#             pair.src.features, pair.target.features = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
#             src_overlap, src_saliency = overlaps[:len_src].detach().cpu(), saliencies[:len_src].detach().cpu()
#             tgt_overlap, tgt_saliency = overlaps[len_src:].detach().cpu(), saliencies[len_src:].detach().cpu()

#             src_overlap_cloud = pair.src.prob_subsample(self.config.n_points, src_overlap, src_saliency)
#             target_overlap_cloud = pair.target.prob_subsample(self.config.n_points, tgt_overlap, tgt_saliency)

#             # THere's something weird here, when the datased is loaded inside the threeDLoMatch class the ground truth
#             # transformation is meant to transform the target to th be position of the source, but the PREDATOR
#             # prediction is trasforming the source to the target. So, in the predator demo.py file the
#             # ransac_pose_estimation function receives(src_pcd, tgt_pcd, src_feats, tgt_feats), but here the order
#             # between src and target is flipped, so the transformation is consistent
#             transform = ransac_pose_estimation(
#                 target_overlap_cloud.arr,
#                 src_overlap_cloud.arr,
#                 target_overlap_cloud.features,
#                 src_overlap_cloud.features,
#                 mutual=False,
#             )
#             return transform
