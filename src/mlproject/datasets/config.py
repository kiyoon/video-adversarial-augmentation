from dataclasses import dataclass


@dataclass
class DatasetConfig:
    num_classes: int = 10
    class_keys: list[str] | None = None
