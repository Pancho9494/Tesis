# import multiprocessing
import torch
import numpy as np
import math
import open3d as o3d

def npy(var):
    return var.data.cpu().numpy()

def write_ply(fn, point, normal=None, color=None):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)

    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)

    o3d.io.write_point_cloud(fn, ply)

    return


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ_true = np.asarray(occ1)
    occ_pred = np.asarray(occ2)


    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ_true.ndim >= 2:
        occ_true = occ_true.reshape(occ_true.shape[0], -1)
    if occ_pred.ndim >= 2:
        occ_pred = occ_pred.reshape(occ_pred.shape[0], -1)

    # Convert to boolean values
    occ_true = (occ_true >= 0.5)
    occ_pred = (occ_pred >= 0.5)

    # Compute IOU
    area_union = (occ_true | occ_pred).astype(np.float32).sum(axis=-1)
    area_intersect = (occ_true & occ_pred).astype(np.float32).sum(axis=-1)
    iou = (area_intersect / area_union)
    
    if area_union == 0:
        return area_union

    return iou

def trace_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    #area_union = (occ1 | occ2).astype(np.float32)
    #area_intersect = (occ1 & occ2).astype(np.float32)

    labels = (occ1 == occ2).squeeze()
    
    correct_pred = np.where(labels == True)[0]
    wrong_pred = np.where(labels == False)[0]
    return correct_pred, wrong_pred


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def normalize_coord(p, vol_range, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments

    Args:
        p (tensor): point
        vol_range (numpy array): volume boundary
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    p[:, 0] = (p[:, 0] - vol_range[0][0]) / (vol_range[1][0] - vol_range[0][0])
    p[:, 1] = (p[:, 1] - vol_range[0][1]) / (vol_range[1][1] - vol_range[0][1])
    p[:, 2] = (p[:, 2] - vol_range[0][2]) / (vol_range[1][2] - vol_range[0][2])
    
    if plane == 'xz':
        x = p[:, [0, 2]]
    elif plane =='xy':
        x = p[:, [0, 1]]
    elif plane =='yz':
        x = p[:, [1, 2]]
    else:
        x = p    
    return x

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def coord2index(p, vol_range, reso=None, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # normalize to [0, 1]
    x = normalize_coord(p, vol_range, plane=plane)
    
    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else: #* pytorch tensor
        x = (x * reso).long()

    if x.shape[1] == 2:
        index = x[:, 0] + reso * x[:, 1]
        index[index > reso**2] = reso**2
    elif x.shape[1] == 3:
        index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
        index[index > reso**3] = reso**3
    
    return index[None]

def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2**(int(depth) - 1)
    if ~(reso / base).is_integer(): # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break    
    return reso

def decide_total_volume_range(query_vol_metric, recep_field, unit_size, unet_depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        query_vol_metric (numpy array): query volume size
        recep_field (int): defined the receptive field for U-Net
        unit_size (float): the defined voxel size
        unet_depth (int): U-Net number of layers
    '''
    reso = query_vol_metric / unit_size + recep_field - 1
    reso = update_reso(int(reso), unet_depth) # make sure input reso can be processed by UNet
    input_vol_metric = reso * unit_size
    p_c = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    lb_input_vol, ub_input_vol = p_c - input_vol_metric/2, p_c + input_vol_metric/2
    lb_query_vol, ub_query_vol = p_c - query_vol_metric/2, p_c + query_vol_metric/2
    input_vol = [lb_input_vol, ub_input_vol]
    query_vol = [lb_query_vol, ub_query_vol]

    # handle the case when resolution is too large
    if reso > 10000:
        reso = 1
    
    return input_vol, query_vol, reso

def add_key(base, new, base_name, new_name, device=None):
    ''' Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    '''
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base,
                new_name: new}
    return base

class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

def rotate_pointcloud(pointcloud, points, points_iou=None):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    points[:,[0,2]] = points[:,[0,2]].dot(rotation_matrix)
    
    if points_iou is not None:
        points_iou[:,[0,2]] = points_iou[:,[0,2]].dot(rotation_matrix)

    return pointcloud, points, points_iou

def translate_pointcloud(pointcloud, points, points_iou=None):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

    points = np.add(np.multiply(points, xyz1), xyz2).astype('float32')

    if points_iou is not None:
        points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype('float32')
    
    return pointcloud, points, points_iou

def single_translate_pointcloud(pointcloud, points, points_iou=None, points_df=None, points_iou_df=None):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[1])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

    translated_points = np.add(np.multiply(points, xyz1), xyz2).astype('float32')
    
    translated_points_df = None
    if points_df is not None:
        translated_points_df = np.multiply(points_df, xyz1)
    
    translated_points_iou_df = None
    if points_iou_df is not None:
        translated_points_iou_df = np.multiply(points_iou_df, xyz1)

    if points_iou is not None:
        translated_points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype('float32')
        return translated_pointcloud, translated_points, translated_points_iou, translated_points_df, translated_points_iou_df
    else:
        return translated_pointcloud, translated_points, translated_points_df


