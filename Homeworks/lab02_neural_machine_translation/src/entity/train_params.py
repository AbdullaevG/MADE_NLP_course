from dataclasses import dataclass, field

@dataclass
class TrainParams:
    """Structure for data parameters"""
    clip: int = field(default=1)
    num_epochs: int = field(default=30)