from __future__ import annotations

from copy import deepcopy
from functools import partial
import functools
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
        # self._transformations.append(partial(Bunny.split, **kwargs))
        return self

    def __repr__(self) -> str:
        return "RegistrationBunny()"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Pair:
        source = deepcopy(self.mother_bunny)
        target = source
        for transform in self._transformations:
            target = transform(self=target, idx=idx)

        return Pair(source.cloud, target.cloud, GT_tf_matrix=target.T)

    @classmethod
    def new_instance(
        cls, split: CloudDatasetsI.SPLITS, n_samples: int = 100, bunny_seed: int = 1234
    ) -> RegistrationBunny:
        instance = cls(n_samples, bunny_seed)
        instance.split = split
        return instance

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

    tf = torchvision.transforms.Compose(
        transform_factory(
            getattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()),
        )
    )
    source_batch, target_batch = tf(source_batch), tf(target_batch)
    source_batch.points = source_batch.points.reshape(-1, 3)
    source_batch.features = source_batch.features.reshape(-1, 1)
    target_batch.points = target_batch.points.reshape(-1, 3)
    target_batch.features = target_batch.features.reshape(-1, 1)
    return Pair(
        id=batch[0].id,
        source=source_batch,
        target=target_batch,
        GT_tf_matrix=np.squeeze(GT_tf_batch, axis=0),
    )
