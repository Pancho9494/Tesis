import atexit

import config.config as config


def testing_bunny() -> None:
    from LIM.data.sets.datasetI import CloudDatasetsI
    from LIM.data.sets.registration_bunny import RegistrationBunny
    from LIM.data.structures import Pair

    N_SAMPLES = 10
    dataset = iter(
        RegistrationBunny.new_instance(CloudDatasetsI.SPLITS.TRAIN, N_SAMPLES, 1234)
        .with_rotation()
        .with_translation()
        .with_overlap(overlap=0.1)
    )
    for _ in range(N_SAMPLES):
        sample: Pair = next(dataset)
        sample.show()
    return


def train_with_bunny() -> None:
    from LIM.data.sets.registration_bunny import RegistrationBunny
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer
    from LIM.training.trainer import BaseTrainer

    trainer = PredatorTrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=RegistrationBunny,
    )
    atexit.register(trainer.cleanup)
    trainer.train()


if __name__ == "__main__":
    config.settings = config.Settings.from_yaml("src/config/cpu.yaml")
    # testing_bunny()
    train_with_bunny()
