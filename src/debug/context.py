import torch
from typing import Optional
from LIM.data.structures import PCloud


def inspect_tensor(target: torch.Tensor, name: Optional[str] = None) -> str:
    if name is None:
        name = "Tensor"

    out = f"\t|\t {name} \t|\t {tuple(target.shape)} \t|\t {target.min():2.4f} \t|\t {target.to(torch.float32).mean():2.4f} \t|\t {target.max():2.4f} \t|"
    print(out)
    return out


def inspect_cloud(target: PCloud, name: Optional[str] = None) -> str:
    LINES = "-" * 23
    print(f"\t|{LINES}+{LINES}+{LINES}+{LINES}+{LINES}|")
    name = name if name is not None else "Features"
    inspect_tensor(target.features, name=f"{name}")
    print(f"\t|{LINES}+{LINES}+{LINES}+{LINES}+{LINES}|")
    return target
