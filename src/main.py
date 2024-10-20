from rich.traceback import install

install(show_locals=False)
from os import putenv
from LIM.models.IAE.iae import IAE
from LIM.models.trainer import Trainer
from LIM.data.datasets.scanNet import ScanNet

# pytorch with rocm won't run without this
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("PYTORCH_ROCM_ARCH", "gfx1031")


def trainIAE():
    model = IAE()
    data = ScanNet()
    trainer = Trainer(model, data)
    trainer.train(plot=False)


if __name__ == "__main__":
    trainIAE()
