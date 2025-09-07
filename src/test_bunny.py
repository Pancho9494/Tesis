import atexit
from LIM.log import log
import config.config as config


def test_registration_errors() -> None:
    from LIM.data.structures import Bunny
    from LIM.metrics.losses import RRE, RTE
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    rotations = [[1.57, 0.0, 0.0], [0.0, 1.57, 0.0], [0.0, 0.0, 1.57]]
    translations = [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]

    rre = RRE()
    rte = RTE()
    OVERLAP = 0.1
    source = Bunny(seed=1234)
    for rot in rotations:
        target = source.split(overlap=OVERLAP).rotate(R.from_euler("xyz", rot, degrees=False).as_matrix())
        log.info(f"Expected: {rot} RRE={rre((np.eye(4, 4), target.T)):2.3f}")

    for trans in translations:
        target = source.split(overlap=OVERLAP).translate(trans)
        log.info(f"Expected: {trans} RTE={rte((np.eye(4, 4), target.T)):2.3f}")


def testing_bunny() -> None:
    from LIM.data.sets.datasetI import CloudDatasetsI
    from LIM.data.sets.registration_bunny import RegistrationBunny
    from LIM.data.structures import Pair

    N_SAMPLES = 10
    # dataset = iter(
    #    RegistrationBunny.new_instance(CloudDatasetsI.SPLITS.TRAIN, N_SAMPLES, 1234)
    #    .with_overlap(overlap=0.1)
    #    .with_rotation()
    #    .with_translation()
    # )
    dataset = iter(
        RegistrationBunny.build(
            lambda db: db.with_rotation().with_translation(frac_of_bbox=1.5).with_overlap(overlap=0.1)
        ).new_instance(CloudDatasetsI.SPLITS.TRAIN, N_SAMPLES, 1234)
    )
    for _ in range(N_SAMPLES):
        sample: Pair = next(dataset)
        sample.show(sample.GT_tf_matrix)
    return


def train_with_bunny() -> None:
    from LIM.data.sets.registration_bunny import RegistrationBunny
    from LIM.models.PREDATOR import PREDATOR, PredatorTrainer
    from LIM.training.trainer import BaseTrainer

    trainer = PredatorTrainer(
        mode=BaseTrainer.Mode.NEW,
        model=PREDATOR,
        dataset=RegistrationBunny.build(lambda db: db.with_rotation().with_translation().with_overlap()),
    )
    atexit.register(trainer.cleanup)
    trainer.train()


if __name__ == "__main__":
    config.settings = config.Settings.from_yaml("src/config/gpu.yaml")
    log.info(f"{config.settings=}")
    # test_registration_errors()
    # testing_bunny()
    train_with_bunny()
