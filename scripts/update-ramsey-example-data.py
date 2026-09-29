"""Update the bundled Ramsey metadata without changing the measured signals.

Run from the repository root with its Python environment:
    python scripts/update-ramsey-example-data.py

References were computed on 2026-09-29 with Qubex 1.5.0rc2, commit
2bf4160feee692f9ec2ff81b41f9eb440d3ba659 (NumPy 2.5.3, SciPy 1.18.1,
Xarray 2026.7.0). To reproduce in an environment with that Qubex checkout:

    from qubex import fit
    from sqe_analysis.signal_processing import project_complex

    projected = project_complex(ds[qubit], dim="idle_time")
    result = fit.fit_ramsey(
        target=qubit,
        times=projected.idle_time.to_numpy(),
        data=projected.to_numpy(),
        plot=False,
    )
    frequency_hz = float(result["f"]) * 1e9
    t2_star_seconds = float(result["tau"]) * 1e-9

Use the entire stored trace, starting at 0 ns, and Qubex's default fit options.
These are new fits with shared IQ projection, not the original measurement
calibration. Projected signals are used only for fitting, never saved here.
The full-window normalized RMS residuals are 0.1321 (Q22) and 0.0730 (Q06).
Q06's 10 us window is short compared with its fitted 59.9 us decay time, so
only its frequency is retained. Distorted/glitch traces have model mismatch
(residuals 0.6834/0.9117), so no expected fit values are assigned to them.

Task IDs were supplied by the measurement author. Original export filenames
retain the raw record index; no Ramsey X/Y interpretation is assigned to it.
The exports do not identify the signal unit or earlier processing history.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
from xarray.testing import assert_identical

from sqe_analysis.example_data import validate_metadata

DESCRIPTION = (
    "Complex-valued readout signal as a function of idle time in a Ramsey experiment. "
    "One qubit. Uniformly spaced idle times."
)

EXECUTION_ID = "20260915-022"
DATASETS = {
    "good_snr": {
        "task_id": "409a5b2b-b7e5-455d-952c-858845584e63",
        "original_file": "CheckRamsey_22_raw_0.nc",
        "quality_notes": (
            "Good SNR with clear damped oscillations. "
            "The decay does not fully saturate."
        ),
        "expected_fit_result": {
            "Q22": {
                "ramsey_frequency": 999798.989850514,
                "t2_star": 3.314007312873191e-05,
            }
        },
    },
    "good_snr_cut_off": {
        "task_id": "856d58d5-e07a-437b-8925-00800c76133d",
        "original_file": "CheckRamsey_6_raw_0.nc",
        "quality_notes": (
            "Good SNR. The observation window is too short to resolve the full decay."
        ),
        "expected_fit_result": {"Q06": {"ramsey_frequency": 1280533.4004875328}},
    },
    "distorted": {
        "task_id": "c5333c83-c459-404a-97ed-69acb7186b51",
        "original_file": "CheckRamsey_2_raw_0.nc",
        "quality_notes": "Clear oscillations with beating-like envelope modulation.",
    },
    "glitch": {
        "task_id": "940615b9-a5a5-4fb7-9670-938a47e1cd7a",
        "original_file": "CheckRamsey_35_raw_1.nc",
        "quality_notes": (
            "Large step-like changes and an early transient distort the oscillations. No proper signal to speak of."
        ),
    },
}


def main():
    data_dir = Path(__file__).resolve().parents[1] / "src/sqe_analysis/example_data"
    for qualifier, metadata in DATASETS.items():
        label = EXECUTION_ID.replace("-", "")
        path = data_dir / f"ramsey-{qualifier}-RX4_QD{label}.nc"
        with xr.open_dataset(path, engine="h5netcdf") as original:
            original.load()

        updated = original.assign_attrs(
            description=DESCRIPTION,
            quality_notes=metadata["quality_notes"],
            source=(
                "RIKEN, SQERT XLD4, chip FY2023 2nd 64Q No3 (1, 0), "
                f"QDash execution {EXECUTION_ID}, task {metadata['task_id']}; {metadata['original_file']}."
            ),
        )
        if "expected_fit_result" in metadata:
            updated.attrs["expected_fit_result"] = json.dumps(
                metadata["expected_fit_result"], allow_nan=False
            )
        validate_metadata(updated)
        if updated.attrs == original.attrs:
            continue

        # Verify a separate file before replacing a dataset open in a notebook.
        with TemporaryDirectory(dir=data_dir) as temporary_dir:
            temporary_path = Path(temporary_dir) / path.name
            updated.to_netcdf(temporary_path, engine="h5netcdf")
            with xr.open_dataset(temporary_path, engine="h5netcdf") as saved:
                validate_metadata(saved)
                assert_identical(saved, updated)
                assert_identical(
                    saved.drop_attrs(deep=False), original.drop_attrs(deep=False)
                )
            temporary_path.replace(path)
        print(f"Updated {path.name}")


if __name__ == "__main__":
    main()
