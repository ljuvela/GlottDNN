"""NumPy wrappers for GlottDNN signal-processing primitives."""

import numpy as np

from glottdnn_cpp import signal_processing as _native


def interpolate_linear(values, size):
    """Resample a one-dimensional array to ``size`` samples linearly."""
    return _native._interpolate_linear(np.asarray(values, dtype=np.float64), size)


def filter_signal(b, a, signal):
    """Apply the FIR/IIR filter defined by numerator ``b`` and denominator ``a``."""
    return _native._filter(np.asarray(b, dtype=np.float64),
                           np.asarray(a, dtype=np.float64),
                           np.asarray(signal, dtype=np.float64))


def convolve(first, second):
    """Return the convolution of two one-dimensional arrays."""
    return _native._conv(np.asarray(first, dtype=np.float64),
                         np.asarray(second, dtype=np.float64))


def autocorrelation(frame, order):
    """Return autocorrelation coefficients through the requested order."""
    return _native._autocorrelation(np.asarray(frame, dtype=np.float64), order)


def lsf_to_polynomial(lsf):
    """Convert one LSF vector to its polynomial coefficients."""
    return _native._lsf_to_poly(np.asarray(lsf, dtype=np.float64))


def polynomial_to_lsf(polynomial):
    """Convert one polynomial coefficient vector to LSFs."""
    return _native._poly_to_lsf(np.asarray(polynomial, dtype=np.float64))


def window(window_name, frame):
    """Apply a named window (``hann``, ``hamming``, ``blackman``, ``cosine``, or ``rect``)."""
    return _native._window(window_name, np.asarray(frame, dtype=np.float64))


def mean(values):
    """Return the arithmetic mean of a one-dimensional array."""
    return _native._mean(np.asarray(values, dtype=np.float64))


def energy(values):
    """Return the signal energy used by the vocoder."""
    return _native._energy(np.asarray(values, dtype=np.float64))


def next_power_of_two(value):
    """Return the smallest power of two greater than or equal to ``value``."""
    return _native._next_pow2(value)


def linear_to_erb(values, sample_rate):
    """Convert linear frequencies to the ERB scale."""
    return _native._linear_to_erb(np.asarray(values, dtype=np.float64), sample_rate)


def erb_to_linear(values, sample_rate):
    """Convert ERB frequencies to linear frequencies."""
    return _native._erb_to_linear(np.asarray(values, dtype=np.float64), sample_rate)


def median_filter(values, length):
    """Apply a median filter with the specified window length."""
    return _native._median_filter(np.asarray(values, dtype=np.float64), length)


def moving_average_filter(values, length):
    """Apply a moving-average filter with the specified window length."""
    return _native._moving_average_filter(np.asarray(values, dtype=np.float64), length)
