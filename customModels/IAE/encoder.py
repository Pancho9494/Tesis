import torch
from submodules.GeoTransformer.geotransformer.modules.kpconv.modules import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock
from submodules.GeoTransformer.geotransformer.modules.kpconv.functional import nearest_upsample

class KPConvFPN(torch.nn.Module):
    def __init__(self, inDim: int, outDim: int, iniDim: int, kerSize: int,
                 iniRadius: float, iniSigma: float, groupNorm: int):
        super(KPConvFPN, self).__init__()
        self.blocks = [
            [
            ConvBlock(inDim, iniDim, kerSize, iniRadius, iniSigma, groupNorm),
            ResidualBlock(iniDim, 2 * iniDim, kerSize, iniRadius, iniSigma, groupNorm),
            ],
            
            [
            ResidualBlock(2 * iniDim, 2 * iniDim, kerSize, 1 * iniRadius, 1 * iniSigma, groupNorm, strided = True),
            ResidualBlock(2 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm),
            ResidualBlock(4 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm),
            ],
            
            [
            ResidualBlock(4 * iniDim, 4 * iniDim, kerSize, 2 * iniRadius, 2 * iniSigma, groupNorm, strided = True),
            ResidualBlock(4 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm),
            ResidualBlock(8 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm),
            ],
            
            [
            ResidualBlock(8 * iniDim, 8 * iniDim, kerSize, 4 * iniRadius, 4 * iniSigma, groupNorm, strided = True),
            ResidualBlock(8 * iniDim, 16 * iniDim, kerSize, 8 * iniRadius, 8 * iniSigma, groupNorm),
            ResidualBlock(16 * iniDim, 16 * iniDim, kerSize, 8 * iniRadius, 8 * iniSigma, groupNorm),
            ]
        ]
        
        self.last_layers = [
            UnaryBlock(24 * iniDim, 8 * iniDim, groupNorm),
            LastUnaryBlock(12 * iniDim, outDim)
        ]

    def forward(self, feats, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        residuals = []
        for i in range(len(self.blocks)):
            for j, layer in enumerate(self.blocks[i]):
                feats = layer(
                            feats, 
                            points_list[i], 
                            points_list[i] if (i == 0 or j != 0) else points_list[i - 1], 
                            neighbors_list[i] if (i == 0 or j != 0) else subsampling_list[i - 1]
                )
            residuals.append(feats)
        
        features = []
        features.append(residuals[3])
        
        latent_s3 = nearest_upsample(residuals[3], upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, residuals[2]], dim = 1)
        latent_s3 = self.last_layers[0](latent_s3)
        features.append(latent_s3)
        
        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, residuals[1]], dim = 1)
        latent_s2 = self.last_layers[1](latent_s2)
        features.append(latent_s2)
        
        features.reverse()
        return features
