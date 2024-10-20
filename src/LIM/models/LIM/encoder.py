import torch
from typing import List


from submodules.GeoTransformer.geotransformer.modules.kpconv.modules import (
    ConvBlock,
    ResidualBlock,
    UnaryBlock,
    LastUnaryBlock,
)
from submodules.GeoTransformer.geotransformer.modules.kpconv.functional import nearest_upsample


class KPConvFPN(torch.nn.Module):
    def __init__(
        self, inDim: int, outDim: int, iniDim: int, kerSize: int, iniRadius: float, iniSigma: float, groupNorm: int
    ):
        super(KPConvFPN, self).__init__()
        self.blocks = [
            [
                ConvBlock(inDim, iniDim, kerSize, iniRadius, iniSigma, groupNorm),
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
        ]

        self.last_layers = [UnaryBlock(24 * iniDim, 8 * iniDim, groupNorm), LastUnaryBlock(12 * iniDim, outDim)]

    def forward(self, feats, data_dict) -> List:
        """
        They do a bunch of stuff in GeoTransformer to generate the data_dict. What I've gathered so far is:

        For the feats values, which are just data_dict['features']
        In the Trainer class
        (submodules/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/trainval.py)
        they load the dataset with:
            train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
            self.register_loader(train_loader, val_loader)

        The train_loader is constructed in the dataset.py file
        (submodules/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/dataset.py)
        like this:
            train_dataset = ThreeDMatchPairDataset(
                cfg.data.dataset_root,
                'train',
                point_limit=cfg.train.point_limit,
                use_augmentation=cfg.train.use_augmentation,
                augmentation_noise=cfg.train.augmentation_noise,
                augmentation_rotation=cfg.train.augmentation_rotation,
            )
            train_loader = build_dataloader_stack_mode(
                train_dataset,
                registration_collate_fn_stack_mode,
                cfg.backbone.num_stages,
                cfg.backbone.init_voxel_size,
                cfg.backbone.init_radius,
                neighbor_limits,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                shuffle=True,
                distributed=distributed,
            )

        So the class that handles the __getitem__ calls is actually ThreeDMatchPairDataset
        (submodules/GeoTransformer/geotransformer/datasets/registration/threedmatch/dataset.py),
        which loads the reference pcd, augmentates them and saves them as 'ref_points'. The features then are
        initialized as an array of ones with the same shape as the points array:
            ref_points = self._load_point_cloud(metadata['pcd0'])
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
            data_dict['ref_points'] = ref_points.astype(np.float32)
            data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)


        But, the collate_fn they use, defined in registration_collate_fn_stack_mode()
        (submodules/GeoTransformer/geotransformer/utils/data.py)
        stacks the points and features of both the reference and the source, and **that** is what gets padded as the
        'features' key:
            feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
            points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
            points = torch.cat(points_list, dim=0)
            collated_dict['features'] = feats

        The rest of the keys here (points, neighbors, subsampling and upsampling) are all defined in the
        precompute_data_stack_mode function (submodules/GeoTransformer/geotransformer/utils/data.py) which is called
        right there in the registration_collate_fn_stack_mode function:
            input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
            collated_dict.update(input_dict)


        So basically, given the src and ref point clouds, they concatenate them, and compute the values for the keys
        using the precompute_data_stack_mode() function (submodules/GeoTransformer/geotransformer/utils/data.py)


        """

        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]
        upsampling_list = data_dict["upsampling"]

        residuals = []
        for i in range(len(self.blocks)):
            for j, layer in enumerate(self.blocks[i]):
                feats = layer(
                    feats,
                    points_list[i],
                    points_list[i] if (i == 0 or j != 0) else points_list[i - 1],
                    neighbors_list[i] if (i == 0 or j != 0) else subsampling_list[i - 1],
                )
            residuals.append(feats)

        features = []
        features.append(residuals[3])

        latent_s3 = nearest_upsample(residuals[3], upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, residuals[2]], dim=1)
        latent_s3 = self.last_layers[0](latent_s3)
        features.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, residuals[1]], dim=1)
        latent_s2 = self.last_layers[1](latent_s2)
        features.append(latent_s2)

        features.reverse()
        return features
