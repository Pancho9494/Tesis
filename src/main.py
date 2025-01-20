import torch
from functools import partial
from LIM.models.PREDATOR.predator import Predator
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch, collate_3dmatch
from LIM.data.structures.transforms import transform_factory

import builtins
from rich import traceback, pretty, print

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def train_predator() -> None:
    dataset = ThreeDLoMatch()
    model = Predator()
    # print(model)

    loader = iter(
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(
                collate_3dmatch,
                tf_pipeline=transform_factory(
                    {
                        # "DOWNSAMPLE": {"N_POINTS": 4096},
                    }
                ),
            ),
            num_workers=1,
            multiprocessing_context=None,
            pin_memory=True,
        )
    )

    sample = next(loader)
    
    RUN = True
    if not RUN:
        source, target = sample.src, sample.target
        print(f"loaded\nsource: {source}\ntarget: {target}")
        source.tensor, target.tensor = source.tensor.reshape(-1, 3), target.tensor.reshape(-1, 3)
        source.features, target.features = source.features.reshape(-1, 1), target.features.reshape(-1, 1)

        print("")

        BLOCK_1_DL, BLOCK_1_RADIUS = 0.05, 0.0625
        BLOCK_2_DL, BLOCK_2_RADIUS = 0.1, 0.125
        BLOCK_3_DL, BLOCK_3_RADIUS = 0.2, 0.25
        BLOCK_4_DL, BLOCK_4_RADIUS = None, 0.5
        source.layers.within(BLOCK_1_RADIUS, BLOCK_1_DL)
        source.layers.within(BLOCK_2_RADIUS, BLOCK_2_DL)
        source.layers.within(BLOCK_3_RADIUS, BLOCK_3_DL)
        source.layers.within(BLOCK_4_RADIUS, BLOCK_4_DL)

        print()
        print("points:\n", end="")
        print(
            {
                key: f"{value.shape} -> ({value.min():2.04f}, {value.max():2.04f})"
                for key, value in source.layers.points.items()
            }
        )
        print("neighbors:\n", end="")
        print(
            {key: f"{value.shape} -> ({value.min()}, {value.max()})" for key, value in source.layers.neighbors.items()}
        )
        print("pools:\n", end="")
        print(
            {
                key: (
                    f"{value.shape} -> ({value.min()}, {value.max()})"
                    if value.shape[0] != 0
                    else f"{value.shape} -> (0, 0)"
                )
                for key, value in source.layers.pools.items()
            }
        )
        print()
    else:
        sample = model(sample)


if __name__ == "__main__":
    train_predator()
