from dataclasses import dataclass, field

@dataclass
class Point:
    x: float = field(default = 0.0)
    y: float = field(default = 0.0)