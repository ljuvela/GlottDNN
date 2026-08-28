import numpy as np
import pytest
from pathlib import Path
from urllib.request import urlopen

import glottdnn_cpp


@pytest.fixture(scope="session")
def audio_file(tmp_path_factory):
    destination = tmp_path_factory.mktemp("audio") / "sample.wav"
    with urlopen(
        "http://festvox.org/cmu_arctic/cmu_arctic/"
        "cmu_us_slt_arctic/wav/arctic_a0001.wav",
        timeout=30,
    ) as response:
        destination.write_bytes(response.read())
    return destination


@pytest.fixture
def config_file():
    return Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"


def test_high_pass_filter_binding(config_file):
    signal = np.linspace(-1.0, 1.0, 256)
    filtered = glottdnn_cpp.analysis.high_pass_filter(signal, str(config_file))
    assert filtered.shape == signal.shape
    assert not np.shares_memory(filtered, signal)


def test_spectral_analysis_binding(audio_file, config_file):
    from scipy.io import wavfile

    _, values = wavfile.read(audio_file)
    signal = values.astype(np.float64) / 32768.0
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    polynomial = glottdnn_cpp.analysis.spectral_analysis(
        analyzed["signal"],
        analyzed["fundf"],
        analyzed["gci_inds"],
        str(config_file),
    )
    assert polynomial.shape == analyzed["poly_vocal_tract"].shape


def test_qmf_spectral_analysis_binding(audio_file, config_file):
    from scipy.io import wavfile

    _, values = wavfile.read(audio_file)
    signal = values.astype(np.float64) / 32768.0
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    polynomial = glottdnn_cpp.analysis.spectral_analysis_qmf(
        analyzed["signal"], analyzed["fundf"], analyzed["gci_inds"],
        str(config_file),
    )
    assert polynomial.shape == analyzed["poly_vocal_tract"].shape


def test_params_object_can_be_loaded_and_reused(config_file):
    params = glottdnn_cpp.analysis.load_params(str(config_file))
    assert params.fs == 16000
    assert params.frame_shift > 0
    assert params.default_windowing_function == glottdnn_cpp.WindowingFunctionType.HANN
    signal = np.zeros(256)
    filtered = glottdnn_cpp.analysis.high_pass_filter(signal, str(config_file))
    assert filtered.shape == signal.shape


def test_inverse_filter_binding(audio_file, config_file):
    from scipy.io import wavfile

    _, values = wavfile.read(audio_file)
    signal = values.astype(np.float64) / 32768.0
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    result = glottdnn_cpp.analysis.inverse_filter(
        analyzed["signal"],
        analyzed["gci_inds"],
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["poly_vocal_tract"],
        str(config_file),
    )
    assert result["source_signal"].shape == analyzed["signal"].shape
    assert result["poly_glot"].shape[1] == analyzed["fundf"].shape[0]
