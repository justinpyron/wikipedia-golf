"""Dataset registry."""

from .smoke import dataset as smoke_dataset

DATASETS = {
    "smoke": smoke_dataset,
}
