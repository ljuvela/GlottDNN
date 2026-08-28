import numpy as np

import signal_processing


def test_convolve_and_statistics():
    np.testing.assert_allclose(
        signal_processing.convolve([1.0, 2.0], [3.0, 4.0]),
        [3.0, 10.0, 8.0],
    )
    assert signal_processing.mean([1.0, 2.0, 3.0]) == 2.0
    assert signal_processing.next_power_of_two(17) == 32


def test_filter_and_window_return_new_arrays():
    signal = np.ones(5)
    filtered = signal_processing.filter_signal([1.0], [1.0], signal)
    windowed = signal_processing.window("rect", signal)

    np.testing.assert_allclose(filtered, signal)
    np.testing.assert_allclose(windowed, signal)
    assert filtered is not signal
    assert windowed is not signal


def test_autocorrelation_and_interpolation_shapes():
    np.testing.assert_allclose(
        signal_processing.autocorrelation([1.0, 2.0, 3.0], 2),
        [14.0, 8.0, 3.0],
    )
    interpolated = signal_processing.interpolate_linear([0.0, 1.0], 5)
    assert interpolated.shape == (5,)
    np.testing.assert_allclose(interpolated, [0.0, 0.25, 0.5, 0.75, 1.0])
