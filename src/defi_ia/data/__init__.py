"""Data access layer: load raw competition files into tidy DataFrames."""

from defi_ia.data.load import (
    load_categories,
    load_test,
    load_train,
)

__all__ = ["load_categories", "load_train", "load_test"]
