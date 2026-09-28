"""
Tests for DampedOscillationAnalysis
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from sqe_analysis.analysis import DampedOscillationAnalysis
from sqe_analysis.example_data import open_dataset
from sqe_analysis.signal_processing import project_complex


@pytest.mark.parametrize("phase", [-0.2, 0.125, 0.8])
@pytest.mark.parametrize("first_time", [0.0, 3.0])
def test_damped_oscillation_phase_guess(phase, first_time):
    """Estimate phase in turns relative to time zero."""
    sample_count = 256
    time_step = 0.25

    # Use exactly seven periods over the DFT window.
    frequency = 7 / (sample_count * time_step)
    time = first_time + np.arange(sample_count) * time_step

    data = xr.DataArray(
        0.2 + 0.8 * np.cos(2 * np.pi * (frequency * time + phase)),
        coords={"time": time},
    )

    guess = DampedOscillationAnalysis.guess(data, coords="time")

    assert guess is not None
    assert guess["f"].item() == pytest.approx(frequency, rel=1e-12, abs=0)

    # Phases differing by an integer number of turns are equivalent.
    phase_error = (guess["phi"].item() - phase + 0.5) % 1.0 - 0.5
    assert phase_error == pytest.approx(0.0, abs=1e-10)


def test_damped_oscillation_guess_nonuniform():
    """The FFT initializer does not support nonuniform spacing"""
    time = np.array([0.0, 1.0, 2.2, 3.0, 4.0])
    data = xr.DataArray(
        np.cos(2 * np.pi * 0.2 * time),
        coords={"time": time},
    )

    assert DampedOscillationAnalysis.guess(data, coords="time") is None


def test_damped_oscillation_analysis_ramsey_good_snr():
    """Fit real Ramsey data with both zero and nonzero starting times."""
    ds = open_dataset("ramsey-good_snr-RX4_QD409a5b2bb7e5455d952c858845584e63")
    dim = "idle_time"
    assert ds[dim].attrs["units"] == "ns"
    data = ds.Q22.assign_attrs(dataset_id=ds.source)

    results = []
    for first_sample in (0, 3):
        trace = data.isel({dim: slice(first_sample, None)})
        result = DampedOscillationAnalysis.run(trace, coords=dim)

        assert result.success.item()
        projected = result.intermediate_results.preprocessed_data
        fitted = DampedOscillationAnalysis.func(projected[dim], **result.fit_params)
        residual = fitted - projected
        normalized_rms = np.sqrt((residual**2).mean() / projected.var())

        # Both windows give about 0.13; allow some variation in the fit.
        assert normalized_rms.item() < 0.2
        results.append(result)

    full, cropped = results

    # Regression references from inspected fits, not experimental ground truth.
    # With time in ns, frequency is in cycle/ns and tau is in ns.
    assert full.params.f.item() == pytest.approx(1e-3, rel=0.01, abs=0)
    assert full.params.tau.item() == pytest.approx(33e3, rel=0.1, abs=0)
    assert cropped.params.f.item() == pytest.approx(
        full.params.f.item(), rel=1e-3, abs=0
    )
    assert cropped.params.tau.item() == pytest.approx(
        full.params.tau.item(), rel=0.05, abs=0
    )


@pytest.mark.parametrize("first_sample", [0, 3])
def test_damped_oscillation_analysis_ramsey_cut_off(first_sample):
    """Fit a short observation window using the default optimizer settings."""
    ds = open_dataset("ramsey-good_snr_cut_off-RX4_QD856d58d5e07a437b892500800c76133d")
    dim = "idle_time"
    assert ds[dim].attrs["units"] == "ns"
    data = ds.Q06.assign_attrs(dataset_id=ds.source)

    trace = data.isel({dim: slice(first_sample, None)})
    result = DampedOscillationAnalysis.run(trace, coords=dim)

    assert result.success.item()

    projected = result.intermediate_results.preprocessed_data
    fitted = DampedOscillationAnalysis.func(projected[dim], **result.fit_params)
    residual = fitted - projected
    normalized_rms = np.sqrt((residual**2).mean() / projected.var())
    assert normalized_rms.item() < 0.2

    # Regression reference in cycle/ns; the window does not constrain tau well.
    assert result.params.f.item() == pytest.approx(1.2805e-3, rel=0.01, abs=0)


def test_damped_oscillation_analysis_ramsey_batch_with_missing_trace():
    """Select FFT peaks per trace without an all-NaN neighbor aborting the fit."""
    good = open_dataset("ramsey-good_snr-RX4_QD409a5b2bb7e5455d952c858845584e63")
    cut_off = open_dataset(
        "ramsey-good_snr_cut_off-RX4_QD856d58d5e07a437b892500800c76133d"
    )
    data = (
        xr.concat(
            [
                good.Q22.isel(idle_time=slice(0, 101)),
                cut_off.Q06,
                xr.full_like(cut_off.Q06, np.nan),
            ],
            dim=xr.IndexVariable("trace", ["good", "cut_off", "missing"]),
        )
        .transpose("idle_time", "trace")
        .assign_attrs(dataset_id="test-ramsey-batch")
    )
    result = DampedOscillationAnalysis.run(data, coords="idle_time")

    assert result.success.dims == ("trace",)
    assert result.success.trace.values.tolist() == ["good", "cut_off", "missing"]
    assert result.success.values.tolist() == [True, True, False]
    assert result.params.sel(trace="missing").to_array().isnull().all()

    valid = result.params.sel(trace=["good", "cut_off"])
    assert valid.f.values.tolist() == pytest.approx([1e-3, 1.2805e-3], rel=0.01, abs=0)

    projected = result.intermediate_results.preprocessed_data.sel(
        trace=["good", "cut_off"]
    )
    fitted = DampedOscillationAnalysis.func(
        projected.idle_time,
        **result.fit_params.sel(trace=["good", "cut_off"]),
    )
    normalized_rms = np.sqrt(
        ((fitted - projected) ** 2).mean("idle_time") / projected.var("idle_time")
    )
    assert (normalized_rms < 0.2).all()


@pytest.mark.parametrize("coordinate_form", ["dataarray", "list", "iterator"])
def test_damped_oscillation_analysis_real_manual_coordinates(coordinate_form):
    """Preserve real-input manual fitting through the base coordinate API."""
    ds = open_dataset("ramsey-good_snr-RX4_QD409a5b2bb7e5455d952c858845584e63")
    data = (project_complex(ds.Q22, dim="idle_time") + 10).assign_attrs(
        dataset_id=ds.source
    )
    reference = DampedOscillationAnalysis.run(data, coords="idle_time")

    if coordinate_form == "dataarray":
        coords = data.idle_time
    elif coordinate_form == "list":
        coords = ["idle_time"]
    else:
        coords = iter(["idle_time"])

    result = DampedOscillationAnalysis.run(
        data,
        coords=coords,
        guess=reference.fit_params,
        curvefit_kwargs={"reduce_dims": "idle_time"},
    )

    assert result.success.item()
    assert result.intermediate_results is None
    assert_allclose(
        result.fit_params,
        reference.fit_params,
        rtol=1e-5,
        atol=1e-8,
    )
