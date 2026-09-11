import pytest

from sqe_analysis.example_data import (
    get_dataset_names,
    open_dataset,
    validate_metadata,
)


def test_open_dataset_unknown_name():
    with pytest.raises(ValueError, match="not found"):
        open_dataset("nonexistent")


def test_all_metadata_valid():
    invalid = []
    for ds_name in get_dataset_names():
        try:
            validate_metadata(open_dataset(ds_name))
        except ValueError as e:
            invalid.append((ds_name, e))

    if invalid:
        raise ValueError(f"Errors validating metadata:\n" + "\n".join(f"{ds_name}: {e}" for ds_name, e in invalid))
