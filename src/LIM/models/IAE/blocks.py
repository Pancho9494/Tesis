import torch
import torch.nn as nn
import math
import os.path as osp
import numpy as np
from os import makedirs
from os.path import join, exists
import open3d as o3d
import matplotlib.pyplot as plt

# TODO: deal with this monster file later, right now we're just testing if it works


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def nearest_upsample(x, upsample_indices):
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def knn_interpolate(s_feats, q_points, s_points, neighbor_indices, k, eps=1e-8):
    s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (M + 1, 3)
    s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (M + 1, C)
    knn_indices = neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, k, 3)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, k, C)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)  # (N, k)
    knn_masks = torch.ne(knn_indices, s_points.shape[0] - 1).float()  # (N, k)
    knn_weights = knn_masks / (knn_sq_distances + eps)  # (N, k)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)  # (N, k)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return q_feats


class KNNInterpolate(nn.Module):
    def __init__(self, k, eps=1e-8):
        super(KNNInterpolate, self).__init__()
        self.k = k
        self.eps = eps

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        if self.k == 1:
            return nearest_upsample(s_feats, neighbor_indices)
        else:
            return knn_interpolate(s_feats, q_points, s_points, neighbor_indices, self.k, eps=self.eps)


def maxpool(x, neighbor_indices):
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = neighbor_feats.max(1)[0]
    return pooled_feats


class MaxPool(nn.Module):
    @staticmethod
    def forward(s_feats, neighbor_indices):
        return maxpool(s_feats, neighbor_indices)


def global_avgpool(x, batch_lengths):
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        averaged_features.append(torch.mean(x[i0 : i0 + length], dim=0))
        i0 += length
    x = torch.stack(averaged_features)
    return x


class GlobalAvgPool(nn.Module):
    @staticmethod
    def forward(feats, lengths):
        return global_avgpool(feats, lengths)


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        r"""Initialize a group normalization block.

        Args:
            num_groups: number of groups
            num_channels: feature dimension
        """
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x.squeeze()


class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm, has_relu=True, bias=True, layer_norm=False):
        r"""Initialize a standard unary block with GroupNorm and LeakyReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            group_norm: number of groups in group normalization
            bias: If True, use bias
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        if has_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = None

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x


class LastUnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        r"""Initialize a standard last_unary block without GN, ReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
        """
        super(LastUnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.mlp(x)
        return x


def spherical_Lloyd(
    radius,
    num_cells,
    dimension=3,
    fixed="center",
    approximation="monte-carlo",
    approx_n=5000,
    max_iter=500,
    momentum=0.9,
    verbose=0,
):
    """
    Creation of kernel point via Lloyd algorithm. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: Radius of the kernels
    :param num_cells: Number of cell (kernel points) in the Voronoi diagram.
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points (Uniform distribution in a sphere)
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[np.logical_and(d2 < radius0**2, (0.9 * radius0) ** 2 < d2), :]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    # Optional fixing
    if fixed == "center":
        kernel_points[0, :] *= 0
    if fixed == "verticals":
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    ##############################
    # Approximation initialization
    ##############################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    # Initialize discretization in this method is chosen
    if approximation == "discretization":
        side_n = int(np.floor(approx_n ** (1.0 / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError("Unsupported dimension (max is 4)")
    elif approximation == "monte-carlo":
        X = np.zeros((0, dimension))
    else:
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))

    # Only points inside the sphere are used
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]

    #####################
    # Kernel optimization
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    max_moves = np.zeros((0,))

    for iter in range(max_iter):
        # In the case of monte-carlo, renew the sampled points
        if approximation == "monte-carlo":
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]

        # Get the distances matrix [n_approx, K, dim]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)

        # Compute cell centers
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = cell_inds == c
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])

        # Update kernel points with low pass filter to smooth mote carlo
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # Check moves for convergence
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        # Optional fixing
        if fixed == "center":
            kernel_points[0, :] *= 0
        if fixed == "verticals":
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

        if verbose:
            print("iter {:5d} / max move = {:f}".format(iter, np.max(np.linalg.norm(moves, axis=1))))
            if warning:
                print("{:}WARNING: at least one point has no cell{:}".format(bcolors.WARNING, bcolors.ENDC))
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker=".", cmap=plt.get_cmap("tab20"))
            # plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], "k+")
            circle = plt.Circle((0, 0), radius0, color="r", fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect("equal")
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

    ###################
    # User verification
    ###################

    # Show the convergence to ask user if this kernel is correct
    if verbose:
        if dimension == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.duplicate_removal(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker=".", cmap=plt.get_cmap("tab20"))
            # plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], "k+")
            circle = plt.Circle((0, 0), radius0, color="r", fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect("equal")
            plt.title("Check if kernel is correct.")
            plt.draw()
            plt.show()

        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title("Check if kernel is correct.")
            plt.show()

    return kernel_points * radius


def kernel_point_optimization_debug(
    radius, num_points, num_kernels=1, dimension=3, fixed="center", ratio=0.66, verbose=0
):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

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

    # Optional fixing
    if fixed == "center":
        kernel_points[:, 0, :] *= 0
    if fixed == "verticals":
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

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

        if verbose:
            print("iter {:5d} / max grad = {:f}".format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], ".")
            circle = plt.Circle((0, 0), radius, color="r", fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect("equal")
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """
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


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    # Kernel directory
    # kernel_dir = osp.join('kernels', 'dispositions')
    kernel_dir = osp.join(osp.dirname(osp.abspath(__file__)), "dispositions")
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # To many points switch to Lloyds
    if num_kpoints > 30:
        lloyd = True

    # Kernel_file
    kernel_file = join(kernel_dir, "k_{:03d}_{:s}_{:d}D.ply".format(num_kpoints, fixed, dimension))

    # Check if already done
    if not exists(kernel_file):
        if lloyd:
            # Create kernels
            kernel_points = spherical_Lloyd(1.0, num_kpoints, dimension=dimension, fixed=fixed, verbose=0)
        else:
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
        o3d.io.write_point_cloud(kernel_file, pcd)
    else:
        pcd = o3d.io.read_point_cloud(kernel_file)
        kernel_points = np.array(pcd.points).astype(np.float32)

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


class KPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        bias=False,
        dimension=3,
        inf=1e6,
        eps=1e-9,
    ):
        """Initialize parameters for KPConv.

        Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

        Deformable KPConv is not supported.

        Args:
             in_channels: dimension of input features.
             out_channels: dimension of output features.
             kernel_size: Number of kernel points.
             radius: radius used for kernel point init.
             sigma: influence radius of each kernel point.
             bias: use bias or not (default: False)
             dimension: dimension of the point space.
             inf: value of infinity to generate the padding point
             eps: epsilon for gaussian influence
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension

        self.inf = inf
        self.eps = eps

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(self.kernel_size, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer("kernel_points", kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self):
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed="center")
        return torch.from_numpy(kernel_points).float()

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        r"""KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)
        neighbors = index_select(s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences**2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)  # (N+1, C) -> (M, H, C)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        weighted_feats = weighted_feats.permute(1, 0, 2)  # (M, K, C) -> (K, M, C)
        kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, C_out) -> (K, M, C_out)
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)  # (K, M, C_out) -> (M, C_out)

        # normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "kernel_size: {}".format(self.kernel_size)
        format_string += ", in_channels: {}".format(self.in_channels)
        format_string += ", out_channels: {}".format(self.out_channels)
        format_string += ", radius: {:g}".format(self.radius)
        format_string += ", sigma: {:g}".format(self.sigma)
        format_string += ", bias: {}".format(self.bias is not None)
        format_string += ")"
        return format_string


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        negative_slope=0.1,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a KPConv block with ReLU and BatchNorm.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            negative_slope: leaky relu negative slope
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        strided=False,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm_conv = nn.LayerNorm(mid_channels)
        else:
            self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(
            mid_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(
                in_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.unary1(s_feats)

        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)

        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        return x
