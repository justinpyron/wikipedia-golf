"""Dataset registry."""

from .easy import dataset as easy_dataset
from .hard import dataset as hard_dataset
from .moderate import dataset as moderate_dataset
from .smoke import dataset as smoke_dataset

DATASETS = {
    "smoke": smoke_dataset,
    "easy": easy_dataset,
    "moderate": moderate_dataset,
    "hard": hard_dataset,
}
