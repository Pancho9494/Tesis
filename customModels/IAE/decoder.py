import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Dict
from enum import Enum
from dataclasses import dataclass, field

@dataclass
class Plane:
    """
    This class constructs a list of indices that correspond to plane axes, so we can select them like this:
        axes = Plane().X.Z
        
    Instead of like this:
        axes = 'xz'
        if axes == 'xy':
            ...
        if axes == 'xz':
            ...
            
    """
    axes: List = field(default_factory=list)
    
    @property
    def X(self) -> Union["Plane", List[int]]:
        self.axes.append(0)
        
        if len(self.axes) == 2:
            return self.axes
        return self
    
    @property
    def Y(self) -> Union["Plane", List[int]]:
        self.axes.append(1)
        
        if len(self.axes) == 2:
            return self.axes
        return self
        
    @property
    def Z(self) -> Union["Plane", List[int]]:
        self.axes.append(2)
        
        if len(self.axes) == 2:
            return self.axes
        return self
    

def normalize_coordinate(coordinates: torch.Tensor, padding: float = 0.1, axes: Optional[List[int]] = None):
    ''' 
    Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        coordinates: point
        padding: Conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        axes: The indices of the planes to be used
    '''
    xy = coordinates if (axes is None) else coordinates[:, :, axes]
    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)
    torch.clamp(xy_new, min = 0.0, max = 1.0 - 10e-6)
    return xy_new


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        x_s = self.shortcut(x) if (self.shortcut is not None) else x
        return x_s + dx
    
class LocalDecoder(nn.Module):
    class SampleModes(Enum):
        BILINEAR = 0

        def __str__(self) -> str:
            return f"{self.name.lower()}"
    
    def __init__(self, latent_dim=128, hidden_size=256, n_blocks=5, 
                 leaky=False, sample_mode=SampleModes.BILINEAR, padding=0.1, d_dim=None):
        super().__init__()
        self.latent_dim = d_dim if (d_dim is not None) else latent_dim
        self.n_blocks = n_blocks

        if self.latent_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(self.latent_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(3, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        self.actvn = F.relu if (not leaky) else lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        
        self.th = nn.Tanh()

    def sample_features(self, points, c, axes: Optional[List[int]] = None):
        p_nor = normalize_coordinate(points.clone(), padding=self.padding + 10e-6, axes=axes)
        p_nor = p_nor[:, :, None].float() if (axes is not None) else p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        output_c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=str(self.sample_mode))
        output_c = output_c.squeeze(-1) if (axes is not None) else output_c.squeeze(-1).squeeze(-1)
        return output_c
    
    def forward(self, points: torch.Tensor, c_plane: Dict):
        if self.latent_dim != 0:
            c = 0
            axes = {'grid': None, 'xz': Plane().X.Z, 'xy': Plane().X.Y, 'yz': Plane().Y.Z}
            for plane in list(c_plane.keys()):
                c += self.sample_features(points, c_plane[plane], axes[plane])
            c = c.transpose(1, 2)
        
        points = points.float()
        net = self.fc_p(points)
        for i in range(self.n_blocks):
            if self.latent_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        out = self.th(out)
        return out

