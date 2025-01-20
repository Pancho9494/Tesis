from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from LIM.data.structures.cloud import Cloud
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from debug.decorators import identify_method

class KPConv(torch.nn.Module, ABC):
    in_dim: int
    out_dim: int
    KP_radius: float
    KP_extent: float
    n_kernel_points: int
    weights: torch.nn.parameter.Parameter
    kernel_points: torch.nn.parameter.Parameter
    root: Path  # Path to the directory where the kernel points are stored / loaded from

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        KP_radius: float = 0.06,
        KP_extent: float = 0.05,
        n_kernel_points: int = 15,
    ) -> None:
        super(KPConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.KP_radius = KP_radius
        self.KP_extent = KP_extent
        self.n_kernel_points = n_kernel_points
        self.weights = torch.nn.parameter.Parameter(
            torch.zeros(size=(self.n_kernel_points, in_dim, out_dim), dtype=torch.float32),
            requires_grad=True,
        )
        self.root: Path = Path("src/models/PREDATOR/kernels/dispositions")
        self._load_kernel_points()

    def __repr__(self) -> str:
        return f"KPConv(in_dim: {self.in_dim}, out_dim: {self.out_dim}, radius: {self.KP_radius:.2f}, extent: {self.KP_extent:.2f}, n_kernel_points: {self.n_kernel_points})"

    @abstractmethod
    def fetch(self, cloud: Cloud, max_neighbors: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @identify_method
    def forward(self, cloud: Cloud) -> Cloud:
        q_pts, s_pts, neighbor_idxs = self.fetch(cloud)

        x = cloud.features

        # add dummy point to filter all indices equal to len(s_pts) + 1
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :] + 1e6)), dim=0)

        neighbors = s_pts[neighbor_idxs, :]
        neighbors -= q_pts.unsqueeze(1)
        neighbors = neighbors.unsqueeze(dim=2)
        differences = neighbors - self.kernel_points
        sq_distances = torch.sum(differences**2, dim=3)
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        all_weights = torch.transpose(all_weights, 1, 2)

        features = torch.cat((x, torch.zeros_like(x[:1, :])), dim=0)
        neighbors_x = self._gather(features, neighbor_idxs)
        weighted_features = torch.matmul(all_weights, neighbors_x).permute((1, 0, 2))

        kernel_outputs = torch.matmul(weighted_features, self.weights)

        output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

        neighbor_features_sum = torch.sum(neighbors_x, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        cloud.features = output_features / neighbor_num.unsqueeze(1)
        return cloud

    def _gather(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Custom gather operation for faster backpropagation

        Args:
            x: input with shape [N, D_1, ..., D_d]
            indices: indexing tensor with shape [N, ..., N_m]

        Returns:
            torch.Tensor: ...
        """
        ss = indices.size()
        for idx, value in enumerate(ss[1:]):
            x = x.unsqueeze(idx + 1)
            new_s = list(x.size())
            new_s[idx + 1] = value
            x = x.expand(new_s)

        n = len(indices.size())
        for idx, value in enumerate(x.size()[n:]):
            indices = indices.unsqueeze(idx + n)
            new_s = list(indices.size())
            new_s[idx + n] = value
            indices = indices.expand(new_s)

        return x.gather(0, indices)

    def _load_kernel_points(self) -> None:
        if not self._search_existing_kernel_points():
            self.kernel_points = self._make_kernel_points()

        # Random rotations
        R = np.eye(3)
        THETA = np.random.rand() * 2 * np.pi
        cos, sin = np.cos(THETA), np.sin(THETA)
        R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
        self.kernel_points = self.kernel_points + np.random.normal(scale=0.01, size=self.kernel_points.shape)
        self.kernel_points = self.KP_radius * self.kernel_points
        self.kernel_points = np.matmul(self.kernel_points, R)
        self.kernel_points = torch.nn.Parameter(
            torch.tensor(self.kernel_points.astype(np.float32), dtype=torch.float32, requires_grad=False),
        )

    def _search_existing_kernel_points(self) -> bool:
        # TODO: log either if the kernel points are found or not
        if not (self.root / f"k_{self.n_kernel_points:03d}_3D.ply").exists():
            return False
        pcd = o3d.io.read_point_cloud(str(self.root / f"k_{self.n_kernel_points:03d}_3D.ply"))
        self.kernel_points = np.asarray(pcd.points)
        return True

    def _make_kernel_points(self) -> torch.Tensor:
        print("No kernel points found, generating new ones")

        RADIUS0 = 1
        DIAMETER0 = 2
        MOVING_FACTOR = 1e-2
        CONTINUOUS_MOVING_DECAY = 0.9995
        GRADIENT_THRESHOLD = 1e-5
        GRADIENT_CLIP = 0.05 * RADIUS0
        DIMENSION = 3
        RADIUS_RATIO = 0.66

        # Kernel Initialization
        kernel_points = np.random.rand(self.n_kernel_points - 1, DIMENSION) * DIAMETER0 - RADIUS0
        while kernel_points.shape[0] < self.n_kernel_points:
            new_points = np.random.rand(self.n_kernel_points - 1, DIMENSION) * DIAMETER0 - RADIUS0
            kernel_points = np.vstack((kernel_points, new_points))
            kernel_points = kernel_points[np.sum(np.power(kernel_points, 2), axis=1) < 0.5 * RADIUS0**2, :]
        kernel_points = kernel_points[: self.n_kernel_points, :].reshape((1, self.n_kernel_points, -1))
        kernel_points[:, 0, :] *= 0

        # Kernel Optimization
        saved_gradient_norms = np.zeros((10_000, self.n_kernel_points))
        old_gradient_norms = np.zeros((1, self.n_kernel_points))
        for iteration in tqdm(range(10_000)):
            # Compute gradients

            ## Derivative of the sum of potentials of all points
            kernel_expanded_1 = np.expand_dims(kernel_points, axis=1)
            kernel_expanded_2 = np.expand_dims(kernel_points, axis=2)
            squared_dist = np.sum(np.power(kernel_expanded_2 - kernel_expanded_1, 2), axis=-1)
            inter_grads = (kernel_expanded_2 - kernel_expanded_1) / (
                np.power(np.expand_dims(squared_dist, axis=-1), 3 / 2) + 1e-6
            )
            inter_grads = np.sum(inter_grads, axis=1)

            ## All gradients
            circle_grads = 10 * kernel_points
            gradients = inter_grads + circle_grads

            # Stop condition

            ## Compute normal of gradients
            gradient_normals = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
            saved_gradient_norms[iteration, :] = np.max(gradient_normals, axis=1)
            LOW_GRADIENT_DIFF = np.max(np.abs(old_gradient_norms[:, 1:] - gradient_normals[:, 1:])) < GRADIENT_THRESHOLD
            if LOW_GRADIENT_DIFF:
                break
            old_gradient_norms = gradient_normals

            # Move points

            ## Clip gradients to get moving distances
            moving_distances = np.minimum(MOVING_FACTOR * gradient_normals, GRADIENT_CLIP)
            moving_distances[:, 0] = 0

            ## Move the points
            kernel_points -= (
                np.expand_dims(moving_distances, axis=-1) * gradients / np.expand_dims(gradient_normals + 1e-6, axis=-1)
            )

            ## Decay moving factor
            MOVING_FACTOR *= CONTINUOUS_MOVING_DECAY

        # Re-scale radius to fit the desired radius
        r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
        kernel_points *= RADIUS_RATIO / np.mean(r[:, 1:])
        kernel_points = kernel_points * self.KP_radius
        best_k = np.argmin(saved_gradient_norms[-1, :])
        kernel_points = kernel_points[best_k, :, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(kernel_points)
        self.root.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(self.root / f"k_{self.n_kernel_points:03d}_3D.ply"), pcd)
        return kernel_points


class KPConvNeighbors(KPConv):
    neighbor_radius: float
    sampleDL: float

    def __init__(self, neighbor_radius: float, sampleDL: float, **kwargs) -> None:
        super(KPConvNeighbors, self).__init__(**kwargs)
        self.neighbor_radius = neighbor_radius
        self.sampleDL = sampleDL

    def fetch(self, cloud: Cloud) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cloud.layers.within(sampleDL=self.sampleDL, radius=self.neighbor_radius)
        if self.sampleDL is not None:
            cloud.layers.within(sampleDL=2 * self.sampleDL, radius=2 * self.neighbor_radius)

        q_pts = cloud.layers.points[f"{self.neighbor_radius:2.04f}"]
        s_pts = cloud.layers.points[f"{self.neighbor_radius:2.04f}"]
        neighbors = cloud.layers.neighbors[f"{self.neighbor_radius:2.04f}"]
        return q_pts, s_pts, neighbors


class KPConvPools(KPConv):
    neighbor_radius: float
    sampleDL: float

    def __init__(self, neighbor_radius: float, sampleDL: float, **kwargs) -> None:
        super(KPConvPools, self).__init__(**kwargs)
        self.neighbor_radius = neighbor_radius
        self.sampleDL = sampleDL

    def fetch(self, cloud: Cloud) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cloud.layers.within(sampleDL=self.sampleDL, radius=self.neighbor_radius)
        cloud.layers.within(sampleDL=2 * self.sampleDL, radius=2 * self.neighbor_radius)

        q_pts = cloud.layers.points[f"{2 * self.neighbor_radius:2.04f}"]
        s_pts = cloud.layers.points[f"{self.neighbor_radius:2.04f}"]
        pools = cloud.layers.pools[f"{self.neighbor_radius:2.04f}"]
        return q_pts, s_pts, pools
