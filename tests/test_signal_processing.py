import numpy as np
import pytest
import xarray as xr
from xarray.testing import (  # pyright: ignore[reportUnknownVariableType]
    assert_allclose,
    assert_equal,
)

from sqe_analysis.example_data import get_dataset_names, open_dataset
from sqe_analysis.signal_processing import project_complex, simple_dft


def test_project_complex_dataarray_zero():
    da = xr.DataArray(np.array([0, 0]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([0, 0], dims=["x"]))


def test_project_complex_dataarray_real():
    da = xr.DataArray(np.array([0, 2]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-1, 1], dims=["x"]))


def test_project_complex_dataarray_imag():
    da = xr.DataArray(np.array([-1j, 1j]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-1, 1], dims=["x"]))


def test_project_complex_dataarray_mixed():
    da = xr.DataArray(np.array([-3 - 4j, 3 + 4j]), dims=["x"])
    projected = project_complex(da)
    assert_equal(projected, xr.DataArray([-5, 5], dims=["x"]))


def test_project_complex_dataset_zero():
    ds = xr.Dataset({"v": (["x"], np.array([0, 0]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [0, 0])}))


def test_project_complex_dataset_real():
    ds = xr.Dataset({"v": (["x"], np.array([0, 2]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-1, 1])}))


def test_project_complex_dataset_imag():
    ds = xr.Dataset({"v": (["x"], np.array([-1j, 1j]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-1, 1])}))


def test_project_complex_dataset_mixed():
    ds = xr.Dataset({"v": (["x"], np.array([-3 - 4j, 3 + 4j]))})
    projected = project_complex(ds)
    assert_equal(projected, xr.Dataset({"v": (["x"], [-5, 5])}))


def test_project_complex_2d_tuple_dim():
    """Project 2D complex data across both dimensions simultaneously."""
    da = xr.DataArray(
        # The data forms a triangle in the complex plane. Projecting across both
        # dimensions should yeld points at +-1. Projecting along a
        # single dimension would yield points at +- sqrt(5) / 2.
        np.array([[-1, 2j], [1, 2j]]),
        dims=["x", "y"],
    )
    projected = project_complex(da, dim=("x", "y"))
    assert_allclose(
        projected,
        xr.DataArray([[-1, 1], [-1, 1]], dims=["x", "y"]),
    )


def test_project_complex_idempotent():
    for ds_name in get_dataset_names():
        ds = open_dataset(ds_name)
        p = project_complex(ds)
        pp = project_complex(p)
        assert_allclose(p, pp)


def test_simple_dft_basic_sine():
    """A single sine wave produces two symmetric peaks in the spectrum."""
    t = np.linspace(0, 1, 128, endpoint=False)
    data = xr.DataArray(np.sin(2 * np.pi * 5 * t), coords=[("time", t)])
    spec = simple_dft(data, "time")
    # The signal is real, so the spectrum is symmetric.
    # The positive-frequency peak should be at f=5.
    expected_freq = spec.frequency.sel(frequency=5, method="nearest")
    # The magnitude at f=5 should be the dominant non-DC component.
    assert abs(spec).sel(frequency=slice(0, None)).idxmax("frequency") == expected_freq


def test_simple_dft_constant_signal():
    """A constant signal has only a DC component."""
    t = np.linspace(0, 1, 64, endpoint=False)
    data = xr.DataArray(np.ones_like(t) * 3, coords=[("time", t)])
    spec = simple_dft(data, "time")
    # DC component is at frequency 0
    assert spec.sel(frequency=0) == pytest.approx(3 * len(t))
    # All other bins should be (near) zero
    cond = spec.frequency != 0
    assert_allclose(
        spec.where(cond),
        xr.zeros_like(spec).where(cond),
        atol=1e-10,
    )


def test_simple_dft_frequency_coordinates():
    """The frequency dimension has correct values matching np.fft.fftfreq."""
    t = np.arange(16, dtype=float) * 0.1
    data = xr.DataArray(np.random.randn(16), coords=[("time", t)])
    spec = simple_dft(data, "time")
    raw_freq = np.fft.fftfreq(16, d=0.1)
    expected_freq = xr.DataArray(
        raw_freq,
        coords=[("frequency", raw_freq)],
    ).sortby("frequency")
    assert_allclose(spec["frequency"], expected_freq)


def test_simple_dft_custom_frequency_dim_name():
    """The output frequency dimension can be renamed."""
    t = np.linspace(0, 1, 32, endpoint=False)
    data = xr.DataArray(np.random.randn(32), coords=[("t", t)])
    spec = simple_dft(data, "t", frequency_dim_name="freq")
    assert "freq" in spec.dims
    assert "t" not in spec.dims


def test_simple_dft_norm_backward():
    """norm='backward' is the default (no normalization)."""
    t = np.linspace(0, 1, 32, endpoint=False)
    data = xr.DataArray(np.ones(32), coords=[("time", t)])
    spec = simple_dft(data, "time", norm="backward")
    # backward: no normalization, so DC = sum of samples = 32
    assert spec.sel(frequency=0).real == 32.0


def test_simple_dft_norm_forward():
    """norm='forward' divides by N."""
    t = np.linspace(0, 1, 32, endpoint=False)
    data = xr.DataArray(np.ones(32), coords=[("time", t)])
    spec = simple_dft(data, "time", norm="forward")
    # forward: divides by N, so DC = 32 / 32 = 1
    assert spec.sel(frequency=0).real == 1.0


def test_simple_dft_norm_ortho():
    """norm='ortho' applies orthonormal normalization (divides by sqrt(N))."""
    t = np.linspace(0, 1, 32, endpoint=False)
    data = xr.DataArray(np.ones(32), coords=[("time", t)])
    spec = simple_dft(data, "time", norm="ortho")
    # ortho: divides by sqrt(N), so DC = 32 / sqrt(32) = sqrt(32)
    assert spec.sel(frequency=0).real == pytest.approx(float(np.sqrt(32)))


def test_simple_dft_preserves_non_transform_dimensions():
    """Non-transform dimensions are preserved and the transform applies along the target dim."""
    t = np.linspace(0, 1, 32, endpoint=False)
    channels = np.array(["a", "b"])
    data = xr.DataArray(
        np.random.randn(2, 32),
        coords=[("channel", channels), ("time", t)],
    )
    spec = simple_dft(data, "time")
    assert spec.dims == ("channel", "frequency")
    assert spec.sizes["channel"] == 2
    assert spec.sizes["frequency"] == 32


def test_simple_dft_single_point_raises():
    """A dimension with only one point raises a clear error."""
    data = xr.DataArray([1.0], coords=[("time", [0.0])])
    with pytest.raises(ValueError, match="at least 2 points"):
        simple_dft(data, "time")


def test_simple_dft_complex_input():
    """The DFT handles complex-valued input correctly."""
    t = np.linspace(0, 1, 64, endpoint=False)
    data = xr.DataArray(np.exp(1j * 2 * np.pi * 3 * t), coords=[("time", t)])
    spec = simple_dft(data, "time")
    # A complex exponential at +3 Hz puts all energy at that bin.
    peak_freq = abs(spec).idxmax("frequency")
    assert peak_freq == 3


def test_simple_dft_matches_numpy_fft():
    """The result matches calling np.fft.fft directly on the values."""
    t = np.linspace(0, 1, 48, endpoint=False)
    vals = np.random.randn(48) + 1j * np.random.randn(48)
    data = xr.DataArray(vals, coords=[("time", t)])
    spec = simple_dft(data, "time")
    # Build the expected result: sort the raw FFT output by frequency.
    dt = float(data["time"].diff("time").mean())
    raw_freq = np.fft.fftfreq(48, d=dt)
    expected = xr.DataArray(
        np.fft.fft(vals),
        coords=[("frequency", raw_freq)],
    ).sortby("frequency")
    assert_allclose(spec, expected)


def test_simple_dft_frequency_sorted():
    """The frequency coordinates are always returned in ascending order."""
    t = np.linspace(0, 1, 64, endpoint=False)
    data = xr.DataArray(np.random.randn(64), coords=[("time", t)])
    spec = simple_dft(data, "time")
    diff = spec["frequency"].diff("frequency")
    assert (diff >= 0).all()
