from hydra_zen import builds, store

from .config import DatasetConfig

store(
    builds(DatasetConfig, num_classes=51, populate_full_signature=True),
    group="dataset",
    name="hmdb51",
)
