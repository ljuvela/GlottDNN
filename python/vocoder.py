"""Array-based Python API for GlottDNN analysis and synthesis."""

import numpy as np
import soundfile as sf

import glottdnn_cpp


def analyze(signal, sample_rate, default_config, user_config=""):
    """Analyze a mono signal and return its vocoder parameters as a dictionary."""
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    result = dict(glottdnn_cpp.analysis.run_array(
        samples, default_config, user_config))
    result["sample_rate"] = int(sample_rate)
    return result


def analyze_file(filename, default_config, user_config=""):
    """Read a WAV file and return its vocoder parameters as a dictionary."""
    signal, sample_rate = sf.read(filename, dtype="float64")
    return analyze(signal, sample_rate, default_config, user_config)


def synthesize(data, default_config, user_config=""):
    """Synthesize an analyzed parameter dictionary and return waveform arrays."""
    result = dict(glottdnn_cpp.synthesis.run_data(
        data, default_config, user_config))
    result["sample_rate"] = data["sample_rate"]
    return result


def synthesize_file(data, filename, default_config, user_config=""):
    """Synthesize data and write its waveform to a WAV file."""
    result = synthesize(data, default_config, user_config)
    sf.write(filename, result["signal"], result["sample_rate"])
    return result
