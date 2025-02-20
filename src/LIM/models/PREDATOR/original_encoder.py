import torch
from debug.decorators import identify_method
from multimethod import multimethod
from typing import Any, List
from LIM.data.structures import Pair
import math
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from dataclasses import dataclass, field


def max_pool(x, inds):
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def gather(x, idx, method=2):
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        ss = idx.size()
        for i, ni in enumerate(ss[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError("Unkown method")


def create_3D_rotations(axis, angle):
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack(
        [t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20, t11 - t12, t19 + t20, t1 + t2 * t24],
        axis=1,
    )

    return np.reshape(R, (-1, 3, 3))


def kernel_point_optimization_debug(
    radius, num_points, num_kernels=1, dimension=3, fixed="center", ratio=0.66, verbose=0
):
    print("making kernel points")
    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[: num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == "center":
        kernel_points[:, 0, :] *= 0
    if fixed == "verticals":
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == "verticals":
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == "center" and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == "verticals" and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == "center":
            moving_dists[:, 0] = 0
        if fixed == "verticals":
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    root: Path = Path("src/LIM/models/PREDATOR/kernels/dispositions")
    # Kernel directory
    # kernel_dir = "kernels/dispositions"
    kernel_dir = str(root)
    if not os.path.exists(kernel_dir):
        os.makedirs(kernel_dir)

    # Kernel_file
    kernel_file = os.path.join(kernel_dir, "k_{:03d}_{:s}_{:d}D.ply".format(num_kpoints, fixed, dimension))

    # Check if already done
    if not os.path.exists(kernel_file):
        print(f"could not find {kernel_file}")
        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(
            1.0, num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0
        )

        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        kernel_points = kernel_points[best_k, :, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(kernel_points)
        root.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(root / f"k_{num_kpoints:03d}_{fixed:s}_{dimension:d}D.ply"), pcd)

    else:
        pcd = o3d.io.read_point_cloud(str(root / f"k_{num_kpoints:03d}_{fixed:s}_{dimension:d}D.ply"))
        kernel_points = np.asarray(pcd.points)

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)


class KPConv(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        p_dim,
        in_channels,
        out_channels,
        KP_extent,
        radius,
        fixed_kernel_points="center",
        KP_influence="linear",
        aggregation_mode="sum",
        deformable=False,
        modulated=False,
    ):
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = torch.nn.parameter.Parameter(
            torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32), requires_grad=True
        )

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(
                self.K,
                self.p_dim,
                self.in_channels,
                self.offset_dim,
                KP_extent,
                radius,
                fixed_kernel_points=fixed_kernel_points,
                KP_influence=KP_influence,
                aggregation_mode=aggregation_mode,
            )
            self.offset_bias = torch.nn.parameter.Parameter(
                torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True
            )

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def __repr__(self) -> str:
        return f"KPConv(in={self.in_channels}, out={self.out_channels}, radius={self.radius}, K={self.K}, p_dim={self.p_dim}, KP_extent: {self.KP_extent})"

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            torch.nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius, self.K, dimension=self.p_dim, fixed=self.fixed_kernel_points)

        # return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False)
        return torch.tensor(K_points_numpy, dtype=torch.float32, device=torch.device("cuda"))

    @identify_method
    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point in the last row for shadow neighbors

        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood

        neighbors -= q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpointsi
        sq_distances = torch.sum(differences**2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        all_weights = torch.transpose(all_weights, 1, 2)

        # In case of closest mode, only the closest KP can influence each point
        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))

        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        # return torch.sum(kernel_outputs, dim=0)
        output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

        # normalization term.
        neighbor_features_sum = torch.sum(neighb_x, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_features = output_features / neighbor_num.unsqueeze(1)

        return output_features


class UnaryBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    @identify_method
    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        out = f"UnaryBlock(in={self.in_dim}, out={self.out_dim})"
        out += f" -> {self.batch_norm.__repr__()}"
        out += f" -> {self.leaky_relu}" if not self.no_relu else ""
        return out


class ResnetBottleneckBlock(torch.nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        super(ResnetBottleneckBlock, self).__init__()
        self.radius = radius

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)

        # KPConv block
        self.KPConv = KPConv(
            config.num_kernel_points,
            config.in_points_dim,
            out_dim // 4,
            out_dim // 4,
            current_extent,
            radius,
            fixed_kernel_points=config.fixed_kernel_points,
            KP_influence=config.KP_influence,
            aggregation_mode=config.aggregation_mode,
            deformable="deform" in block_name,
            modulated=config.modulated,
        )
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = torch.nn.Identity()

        # Other operations
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        return

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}{'_strided' if 'strided' in self.block_name else ''}("
            + f"in={self.in_dim}, "
            + f"out={self.out_dim}, "
            + f"radius={self.radius}, "
            + ")"
        )

    @identify_method
    def forward(self, features, batch):
        if "strided" in self.block_name:
            q_pts = batch["points"][self.layer_ind + 1]
            s_pts = batch["points"][self.layer_ind]
            neighb_inds = batch["pools"][self.layer_ind]
        else:
            q_pts = batch["points"][self.layer_ind]
            s_pts = batch["points"][self.layer_ind]
            neighb_inds = batch["neighbors"][self.layer_ind]

        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if "strided" in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)
        out = self.leaky_relu(x + shortcut)

        return out


class BatchNormBlock(torch.nn.Module):
    def __init__(self, in_dim, use_bn, bn_momentum):
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = torch.nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        torch.nn.init.zeros_(self.bias)

    @identify_method
    def forward(self, x):
        if self.use_bn:
            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        if self.use_bn:
            return f"BatchNormBlock(in_feat: {self.in_dim}, momentum: {self.bn_momentum})"
        else:
            return "Bias()"


class SimpleBlock(torch.nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(
            config.num_kernel_points,
            config.in_points_dim,
            in_dim,
            out_dim // 2,
            current_extent,
            radius,
            fixed_kernel_points=config.fixed_kernel_points,
            KP_influence=config.KP_influence,
            aggregation_mode=config.aggregation_mode,
            deformable="deform" in block_name,
            modulated=config.modulated,
        )

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={self.in_dim}, out={self.out_dim})"

    @identify_method
    def forward(self, x, batch):
        x = self.KPConv(
            q_pts=batch["points"][self.layer_ind],
            s_pts=batch["points"][self.layer_ind],
            neighb_inds=batch["neighbors"][self.layer_ind],
            x=x,
        )

        return self.leaky_relu(self.batch_norm(x))


class EncoderAdapter(torch.nn.Module):
    @dataclass
    class ConfigWrapper:
        modulated: bool = field(default=False)
        aggregation_mode: str = field(default="sum")
        KP_influence: str = field(default="linear")
        fixed_kernel_points: str = field(default="center")
        in_points_dim: int = field(default=3)
        num_kernel_points: int = field(default=15)
        use_batch_norm: bool = field(default=True)
        batch_norm_momentum: float = field(default=0.02)
        conv_radius: float = field(default=2.5)
        KP_extent: float = field(default=2.0)

    def __init__(self) -> None:
        super(EncoderAdapter, self).__init__()
        cfg = EncoderAdapter.ConfigWrapper()
        self.layers = torch.nn.ModuleList(
            [  # 0 simple
                SimpleBlock(block_name="simple", in_dim=1, out_dim=128, radius=0.0625, layer_ind=0, config=cfg),
                # 1 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=64, out_dim=128, radius=0.0625, layer_ind=0, config=cfg
                ),
                # 2 resnetb_strided
                ResnetBottleneckBlock(
                    block_name="resnetb_strided", in_dim=128, out_dim=128, radius=0.0625, layer_ind=0, config=cfg
                ),
                # 3 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=128, out_dim=256, radius=0.125, layer_ind=1, config=cfg
                ),
                # 4 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=256, out_dim=256, radius=0.125, layer_ind=1, config=cfg
                ),
                # 5 resnetb_strided
                ResnetBottleneckBlock(
                    block_name="resnetb_strided", in_dim=256, out_dim=256, radius=0.125, layer_ind=1, config=cfg
                ),
                # 6 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=256, out_dim=512, radius=0.25, layer_ind=2, config=cfg
                ),
                # 7 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=512, out_dim=512, radius=0.25, layer_ind=2, config=cfg
                ),
                # 8 resnetb_strided
                ResnetBottleneckBlock(
                    block_name="resnetb_strided", in_dim=512, out_dim=512, radius=0.25, layer_ind=2, config=cfg
                ),
                # 9 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=512, out_dim=1024, radius=0.5, layer_ind=3, config=cfg
                ),
                # 10 resnetb
                ResnetBottleneckBlock(
                    block_name="resnetb", in_dim=1024, out_dim=1024, radius=0.5, layer_ind=3, config=cfg
                ),
            ]
        )
        # 11 bottle
        self.bottle = torch.nn.Conv1d(1024, 256, kernel_size=1, bias=True)

    def __repr__(self) -> str:
        return "EncoderAdapter()"

    @multimethod
    def forward(self, *args, **kwargs) -> Any: ...

    @multimethod
    def forward(self, pair: Pair) -> Pair:
        # pre compute neighbors
        current: Pair = pair
        for c_radius, c_dl in zip([(2**i) * 0.0625 for i in range(4)], [(2**i) * 0.05 for i in range(3)] + [None]):
            current.compute_neighbors(c_radius, c_dl)
            current.mix = current.mix._super
            current.source = current.source._super if current.source._super is not None else current.source
            current.target = current.target._super if current.target._super is not None else current.target

        pair.mix = pair.mix._sub
        pair.mix._super = None
        current.mix = current.mix.first
        current.source = current.source.first
        current.target = current.target.first

        skip_x = []
        batch = pair.to_legacy()
        x = pair.mix.features
        current: Pair = pair
        for idx, block in enumerate(self.layers):
            if idx in [2, 5, 8, 11]:
                skip_x.append(x)
            x = block(x, batch)

        feats_c = x.transpose(0, 1).unsqueeze(0)
        feats_c = self.bottle(feats_c)
        pair.mix = pair.mix.last
        pair.mix.features = feats_c
        pair.source = pair.source.last
        pair.target = pair.target.last
        return pair, skip_x
