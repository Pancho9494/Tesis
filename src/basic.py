import builtins
from rich import traceback, pretty, print
import types
import sys

# from LIM.data.sets.bunny import Bunny
from pathlib import Path
import importlib

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def import_without_init(path: str | Path, module_name: str) -> types.ModuleType:
    path = path if isinstance(path, str) else Path(path).resolve()
    if str(path.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    #    dataset = Bunny()
    root = Path(__file__).parent
    # dgcnn = import_without_init(
    #     path=root / "submodules/IAE/src/encoder/dgcnn_semseg. py",
    #     module_name="DGCNN_semseg",
    # )
    # decoder = import_without_init(
    #     path=root / "submoddules/IAE/src/dnef/models/decoder.py",
    #     module_name="LocalDecoder",
    # )
    model = getattr(
        import_without_init(
            path=root / "submodules/IAE/src/shapenet_dfnet/models/__init__.py",
            module_name="shapenet_dfnet.models",
        ),
        "ConvolutionalDFNetwork",
    )


if __name__ == "__main__":
    main()
