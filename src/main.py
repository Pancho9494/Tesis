import torch
from functools import partial
from LIM.models.PREDATOR.predator import Predator
from LIM.data.sets.threeDLoMatch import ThreeDLoMatch, collate_3dmatch
from LIM.data.structures.transforms import transform_factory


def train_predator() -> None:
    dataset = ThreeDLoMatch()
    model = Predator()
    print(model)

    loader = iter(
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(
                collate_3dmatch,
                tf_pipeline=transform_factory(
                    {
                        "DOWNSAMPLE": {"N_POINTS": 4096},
                    }
                ),
            ),
            num_workers=1,
            multiprocessing_context=None,
            pin_memory=True,
        )
    )

    sample = next(loader)
    print(sample)

    sample = model(sample)

    return


if __name__ == "__main__":
    train_predator()
