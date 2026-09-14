"""
Tests for DampedOscillationAnalysis
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose, assert_identical

from sqe_analysis.analysis import DampedOscillationAnalysis


@pytest.mark.parametrize("automatic_f", [False, True])
@pytest.mark.parametrize("time_unit, time_scale", [("s", 1.0), ("us", 1e6)])
def test_damped_oscillation_analysis_basic(time_unit, time_scale, automatic_f):
    """Recover a noiseless oscillation with manual or automatic guesses."""
    time = np.linspace(0, 40e-6, 401) * time_scale
    tau = 12e-6 * time_scale
    f = 0.4e6 / time_scale

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / tau) * np.cos(2 * np.pi * f * time + 0.4),
        coords=[("time", time)],
        attrs={"dataset_id": "test"},
    )
    data.time.attrs["units"] = time_unit
    original_data = data.copy(deep=True)

    guess = {"a": 0.7, "b": 0.1, "tau": tau * 0.8, "phi": 0.3}
    if not automatic_f:
        guess["f"] = f * 1.01

    result = DampedOscillationAnalysis.run(
        data,
        coords="time",
        guess=guess,
    )

    if automatic_f:
        frequency_step = 1 / (time.size * (time[1] - time[0]))
        assert result.fit_params_guess.f.item() == pytest.approx(f, abs=frequency_step)
    else:
        assert result.fit_params_guess.f.item() == pytest.approx(guess["f"])

    assert result.success.all()
    assert result.params.a.item() == pytest.approx(0.8)
    assert result.params.b.item() == pytest.approx(0.2)
    assert result.params.tau.item() == pytest.approx(tau, rel=1e-6, abs=0)
    assert result.params.f.item() == pytest.approx(f)
    assert result.params.phi.item() == pytest.approx(0.4)

    assert_allclose(
        DampedOscillationAnalysis.func(data.time, **result.fit_params),
        data,
        rtol=1e-6,
        atol=1e-8,
    )
    assert_identical(data, original_data)


def test_damped_oscillation_analysis_nonuniform_time_with_manual_guess():
    """Fit nonuniformly sampled data using explicitly supplied guesses."""
    time = np.linspace(0, 40e-6, 401)
    time[1::2] += 20e-9

    data = xr.DataArray(
        0.2 + 0.8 * np.exp(-time / 12e-6) * np.cos(2 * np.pi * 400e3 * time + 0.4),
        coords=[("time", time)],
        attrs={"dataset_id": "test"},
    )
    data.time.attrs["units"] = "s"
    original_data = data.copy(deep=True)

    assert DampedOscillationAnalysis.guess(data, coords="time") is None

    result = DampedOscillationAnalysis.run(
        data,
        coords="time",
        guess={"a": 0.7, "b": 0.1, "tau": 10e-6, "f": 404e3, "phi": 0.3},
    )

    assert result.success.all()
    assert result.params.tau.item() == pytest.approx(12e-6, rel=1e-6, abs=0)
    assert result.params.f.item() == pytest.approx(400e3)
    assert_allclose(
        DampedOscillationAnalysis.func(data.time, **result.fit_params),
        data,
        rtol=1e-6,
        atol=1e-8,
    )
    assert_identical(data, original_data)
