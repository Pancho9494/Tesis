from __future__ import annotations

import functools
from copy import deepcopy
from functools import partial
from typing import Callable

import numpy as np
import torchvision

from config.config import settings
from debug.decorators import only_kwargs
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures import Bunny
from LIM.data.structures.pair import Pair
from LIM.data.structures.pcloud import collate_cloud
from LIM.data.structures.transforms import transform_factory
from LIM.log import log


class RegistrationBunny(CloudDatasetsI):
    n_samples: int
    mother_bunny: Bunny
    split: CloudDatasetsI.SPLITS
    _transformations: list[partial]

    def __init__(self, n_samples: int, bunny_seed: int | None = None) -> None:
        self.n_samples = n_samples
        self.mother_bunny = Bunny(seed=bunny_seed) if bunny_seed is not None else Bunny()
        self._transformations = []

    @only_kwargs(["R", "center"])
    def with_rotation(self, *args, **kwargs) -> RegistrationBunny:
        self._transformations.append(partial(Bunny.rotate, **kwargs))
        return self

    @only_kwargs(["t", "frac_of_bbox"])
    def with_translation(self, *args, **kwargs) -> RegistrationBunny:
        self._transformations.append(partial(Bunny.translate, **kwargs))
        return self

    @only_kwargs(["mu", "sigma"])
    def with_noise(self, *args, **kwargs) -> RegistrationBunny:
        self._transformations.append(partial(Bunny.noise, **kwargs))
        return self

    @only_kwargs(["overlap"])
    def with_overlap(self, *args, **kwargs) -> RegistrationBunny:
        self._transformations = [partial(Bunny.split, **kwargs)] + self._transformations
        return self

    def __repr__(self) -> str:
        return "RegistrationBunny()"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Pair:
        transforms = self._transformations

        source = deepcopy(self.mother_bunny)
        if "split" == transforms[0].func.__name__:
            target = transforms[0](self=source, idx=idx)
            transforms = transforms[1:]
        else:
            target = source

        for transform in transforms:
            target = transform(self=target, idx=idx)

        pair = Pair(source.cloud, target.cloud, GT_tf_matrix=target.T)
        pair.correspondences
        return pair

    @classmethod
    def new_instance(
        cls, split: CloudDatasetsI.SPLITS, n_samples: int = 300, bunny_seed: int = 1234
    ) -> RegistrationBunny:
        instance = cls(n_samples, bunny_seed)
        instance.split = split
        return instance

    @classmethod
    def build(cls, config_fn: Callable[[RegistrationBunny], RegistrationBunny]) -> type[RegistrationBunny]:
        template_instance = cls(n_samples=1)
        configured_template = config_fn(template_instance)
        transformations_to_apply = configured_template._transformations
        log.info(f"{[v.func.__name__ for v in transformations_to_apply]=}")

        class ConfiguredRegistrationBunny(cls):
            @classmethod
            def new_instance(
                cls, split: CloudDatasetsI.SPLITS, n_samples: int = 300, bunny_seed: int = 1234
            ) -> RegistrationBunny:
                instance = super().new_instance(split, n_samples, bunny_seed)
                instance._transformations = transformations_to_apply
                return instance

        return ConfiguredRegistrationBunny

    @property
    def collate_fn(self) -> Callable:
        return functools.partial(collate_bunnies, split=self.split)


def collate_bunnies(batch: list[Pair], split: CloudDatasetsI.SPLITS) -> Pair:
    if settings is None:
        raise RuntimeError("settings has not been initialized")
    sources, targets, GT_TFs = [], [], []
    for pair in batch:
        sources.append(pair.source)
        targets.append(pair.target)
        GT_TFs.append(pair.GT_tf_matrix)

    source_batch, target_batch = collate_cloud(sources), collate_cloud(targets)
    GT_tf_batch = np.concatenate([np.expand_dims(arr, axis=0) for arr in GT_TFs], axis=0)

    pair = Pair(
        id=batch[0].id,
        source=source_batch,
        target=target_batch,
        GT_tf_matrix=np.squeeze(GT_tf_batch, axis=0),
    )

    if not hasattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()):
        return pair

    tf = torchvision.transforms.Compose(
        transform_factory(
            getattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()),
        )
    )
    (pair.source, source_idxs), (pair.target, tgt_idxs) = tf(pair.source), tf(pair.target)
    pair.source.points = pair.source.points.reshape(-1, 3)
    pair.source.features = pair.source.features.reshape(-1, 1)
    pair.target.points = pair.target.points.reshape(-1, 3)
    pair.target.features = pair.target.features.reshape(-1, 1)
    return pair
