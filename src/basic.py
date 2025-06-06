import builtins
import importlib.util
from os import PathLike
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any
import sys
from rich import pretty, print, traceback
import torch
from LIM.data.sets.bunny import Bunny

traceback.install(show_locals=False)
pretty.install()
builtins.print = print


def import_without_init(
    path_to_file: str | Path | PathLike, class_name: str, dependencies: list[str | Path | PathLike] | None = None
) -> Any:
    """
    Import a class from a module without running parent package __init__.py
    and with temporary sys.path dependencies.

    Args:
        module_path: Path to the Python file
        class_name: Name of the class to import
        dependencies: Optional list of directories to add to sys.path temporarily
    """
    dependencies = [] if dependencies is None else dependencies
    og_sys_path = sys.path.copy()
    try:
        for dependency in dependencies:
            sys.path.insert(0, str(Path(dependency).resolve()))
        path_to_file = path_to_file if isinstance(path_to_file, Path) else Path(path_to_file)
        spec: ModuleSpec = importlib.util.spec_from_file_location(path_to_file.stem, Path(path_to_file))  # pyright: ignore[reportAssignmentType]
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
        dynamic_class = getattr(module, class_name)
    finally:
        sys.path = og_sys_path
    return dynamic_class()


def main():
    dataset = Bunny()
    encoder = import_without_init(
        path_to_file=(IAE_PATH := Path("src/submodules/IAE/")) / "src/encoder/dgcnn_semseg.py",
        class_name="DGCNN_semseg",
        dependencies=[IAE_PATH],
    )
    decoder = import_without_init(
        path_to_file=IAE_PATH / "src/dfnet/models/decoder.py",
        class_name="LocalDecoder",
        dependencies=[IAE_PATH],
    )
    model = import_without_init(
        path_to_file=IAE_PATH / "src/dfnet/config.py",
        class_name="ConvolutionalDFNetwork",
    )(
        decoder,
        encoder,
        device=torch.device("cuda"),
    )
    print(dataset)
    print(model)
    print(encoder)
    print(decoder)


if __name__ == "__main__":
    main()
