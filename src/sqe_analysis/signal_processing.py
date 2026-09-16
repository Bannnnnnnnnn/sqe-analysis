"""
Functions for digital signal processing with Xarray

These should complement functions available in the base Xarray library and in
other external libraries such as `xr-scipy <https://xr-scipy.readthedocs.io>`_
or `xrft <https://xrft.readthedocs.io/>`_. Those libraries should be preferred
whenever the desired functionality is available.
"""

from typing import Literal, cast, overload

import numpy as np
import xarray as xr
from xarray.core.types import Dims

from sqe_analysis.xarray_util import longest_dim


@overload
def project_complex(
    data: xr.DataArray, discard_imag: bool = True, dim: Dims | None = None
) -> xr.DataArray: ...


@overload
def project_complex(
    data: xr.Dataset, discard_imag: bool = True, dim: Dims | None = None
) -> xr.Dataset: ...


def project_complex(
    data: xr.DataArray | xr.Dataset,
    discard_imag: bool = True,
    dim: Dims | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    A simple method to project complex-valued data to the real axis in a way that maximizes the signal

    Note that the sign of the signal is not guaranteed. The same input data
    rotated slightly in the complex plane can result in output with the opposite
    sign.
    """

    if dim is None:
        dim = longest_dim(data)

    centered = data - data.mean(dim)
    # When the complex-valued points lie on a line that crosses the origin,
    # squaring them puts them all in the same quadrature. We can then take their
    # mean and then use *half* of the angle of that point (since squaring
    # doubles all angles) as the projection angle.
    angle = cast(
        xr.DataArray | xr.Dataset,
        xr.apply_ufunc(np.angle, (centered**2).mean(dim)) * 0.5,
    )
    result = cast(
        xr.DataArray | xr.Dataset,
        np.exp(-1j * angle) * centered,
    )

    if discard_imag:
        result = result.real

    return result


def simple_dft(
    data: xr.DataArray,
    dim: str,
    frequency_dim_name: str = "frequency",
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> xr.DataArray:
    """
    Simple discrete Fourier transform of a DataArray

    This is a simple wrapper around the `NumPy fft function
    <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_. For
    more advanced use, see the `xrft <https://xrft.readthedocs.io/>`_ package.

    Assumes that the transform dimension is evenly spaced.

    Args:
        data: Data array to transform
        dim: Dimension along which to apply transform, usually time
        frequency_dim_name: Name of the transformed dimension in the output data.
        norm: Normalization mode, see the `numpy documentation <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_ for details.
    """
    t = data[dim]
    if t.size < 2:
        raise ValueError(
            f"Dimension '{dim}' must have at least 2 points for DFT, got {t.size}"
        )
    dt = (t[1] - t[0]).item()
    f = np.fft.fftfreq(t.size, d=dt)

    spec = xr.apply_ufunc(
        np.fft.fft,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"norm": norm},
    )

    spec = spec.rename({dim: frequency_dim_name})
    spec = spec.assign_coords({frequency_dim_name: f})
    spec = spec.sortby(frequency_dim_name)
    return spec
