from pathlib import Path
from urllib.request import urlopen

import pytest

import glottdnn_cpp


AUDIO_URL = (
    "http://festvox.org/cmu_arctic/cmu_arctic/"
    "cmu_us_slt_arctic/wav/arctic_a0001.wav"
)


@pytest.fixture(scope="session")
def audio_file(tmp_path_factory):
    destination = tmp_path_factory.mktemp("audio") / "sample.wav"
    try:
        with urlopen(AUDIO_URL, timeout=30) as response:
            destination.write_bytes(response.read())
    except OSError as error:
        pytest.fail("Could not download the test audio sample: {}".format(error))
    return destination


@pytest.fixture
def config_file():
    return Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"


def test_binding_namespaces():
    assert callable(glottdnn_cpp.analysis.run)
    assert callable(glottdnn_cpp.synthesis.run)
    assert hasattr(glottdnn_cpp, "signal_processing")
    assert callable(glottdnn_cpp.analysis.high_pass_filter)
    assert callable(glottdnn_cpp.analysis.spectral_analysis)


def test_in_memory_analysis_and_synthesis(audio_file, config_file):
    import soundfile as sf
    import vocoder

    signal, sample_rate = sf.read(audio_file, dtype="float64")
    analyzed = vocoder.analyze(signal, sample_rate, str(config_file))
    synthesized = vocoder.synthesize(analyzed, str(config_file))
    assert synthesized["sample_rate"] == sample_rate
    assert synthesized["signal"].ndim == 1
    assert synthesized["signal"].size == synthesized["excitation_signal"].size


def test_analysis_and_synthesis_share_params(audio_file, config_file):
    import soundfile as sf
    import vocoder

    signal, sample_rate = sf.read(audio_file, dtype="float64")
    params = vocoder.load_config(str(config_file))
    analyzed = vocoder.analyze(signal, sample_rate, params)
    params.excitation_method = glottdnn_cpp.ExcitationMethod.SINGLE_PULSE
    result = vocoder.synthesize(analyzed, params)
    assert result["sample_rate"] == sample_rate


def test_verbose_toggle_is_default_quiet_and_can_be_enabled(audio_file, config_file, capfd):
    import soundfile as sf
    import vocoder

    signal, sample_rate = sf.read(audio_file, dtype="float64")
    params = vocoder.load_config(str(config_file))
    assert params.verbose is False

    vocoder.analyze(signal, sample_rate, params, verbose=False)
    captured = capfd.readouterr()
    assert "F0 analysis" not in captured.out

    vocoder.analyze(signal, sample_rate, params, verbose=True)
    captured = capfd.readouterr()
    assert "F0 analysis" in captured.out


def test_params_object_can_toggle_verbose_mode(config_file):
    import vocoder

    params = vocoder.load_config(str(config_file))
    params.verbose = True
    assert params.verbose is True
    params.verbose = False
    assert params.verbose is False


def test_param_string_repr_is_readable(config_file):
    import vocoder

    params = vocoder.load_config(str(config_file))
    text = str(params)
    assert text.startswith("Param(")
    assert "fs=" in text
    assert "verbose=" in text
    assert "excitation_method=" in text
    assert repr(params) == text


def test_param_wrapper_exposes_safe_accessors(config_file):
    import vocoder

    params = vocoder.load_config(str(config_file))
    assert isinstance(params, vocoder.ParamWrapper)
    assert params.native is not None
    assert params["fs"] == params.native.fs
    params["speed_scale"] = 0.75
    assert params.speed_scale == 0.75
    assert "fs" in params
    assert "speed_scale" in params.members()
    assert set(params.keys()) >= {"fs", "speed_scale", "verbose"}


def test_in_memory_single_pulse_synthesis(audio_file, config_file):
    import soundfile as sf
    import vocoder

    signal, sample_rate = sf.read(audio_file, dtype="float64")
    data = vocoder.analyze(signal, sample_rate, str(config_file))
    params = vocoder.load_config(str(config_file))
    params.excitation_method = glottdnn_cpp.ExcitationMethod.SINGLE_PULSE
    result = vocoder.synthesize(data, params)
    assert result["sample_rate"] == sample_rate
    assert result["signal"].size > 0


def test_synthesis_rejects_malformed_data(config_file):
    import numpy as np
    import vocoder

    params = glottdnn_cpp.analysis.load_params(str(config_file))
    data = {
        "fundf": np.ones(3),
        "frame_energy": np.ones(2),
        "excitation_pulses": np.zeros((params.paf_pulse_length, 3)),
        "lsf_vocal_tract": np.zeros((params.lpc_order_vt, 3)),
        "lsf_glot": np.zeros((params.lpc_order_glot, 3)),
        "hnr_glot": np.zeros((params.hnr_order, 3)),
    }
    with pytest.raises(ValueError, match="frame_energy"):
        vocoder.synthesize(data, str(config_file))


def test_analysis_and_synthesis_bindings(audio_file, config_file, tmp_path):
    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(audio_file.read_bytes())

    assert glottdnn_cpp.analysis.run(str(wav_file), str(config_file)) == 0

    for extension in (".f0", ".gain", ".hnr", ".lsf", ".slsf", ".pls"):
        assert wav_file.with_suffix(extension).is_file()

    assert glottdnn_cpp.synthesis.run(
        str(wav_file.with_suffix("")),
        str(config_file),
    ) == 0
    assert wav_file.with_name("sample.syn.wav").is_file()
