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
    """Synthesize an analyzed parameter dictionary entirely in memory."""
    required = (
        "fundf", "frame_energy", "excitation_pulses", "lsf_vocal_tract",
        "lsf_glot", "hnr_glot",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("missing synthesis fields: {}".format(", ".join(missing)))

    params = glottdnn_cpp.analysis.load_params(default_config, user_config)
    frames = np.asarray(data["fundf"], dtype=np.float64)
    if frames.ndim != 1 or frames.size == 0:
        raise ValueError("fundf must be a non-empty one-dimensional array")

    excitation = glottdnn_cpp.synthesis.create_excitation(
        data["fundf"], data["frame_energy"], data["excitation_pulses"],
        data["lsf_vocal_tract"], data["lsf_glot"], data["hnr_glot"],
        default_config, user_config,
    )
    if params.noise_gain_voiced > 0.0:
        excitation = glottdnn_cpp.synthesis.harmonic_modification(
            data["fundf"], data["hnr_glot"], excitation, default_config,
            user_config)
    if params.use_spectral_matching:
        excitation = glottdnn_cpp.synthesis.spectral_match_excitation(
            data["fundf"], data["frame_energy"], data["lsf_glot"],
            excitation, default_config, user_config)

    signal = np.zeros_like(excitation)
    spectrum = data.get("spectrum")
    if spectrum is None:
        if params.use_generic_envelope:
            raise ValueError("spectrum is required when use_generic_envelope is enabled")
        spectrum = np.zeros((2049, frames.size), dtype=np.float64)
    signal = glottdnn_cpp.synthesis.fft_filter_excitation(
        data["fundf"], data["frame_energy"], spectrum,
        data["lsf_vocal_tract"], data["lsf_glot"], excitation, signal,
        default_config, user_config,
    )
    signal = glottdnn_cpp.synthesis.generate_unvoiced_signal(
        data["fundf"], spectrum, data["lsf_vocal_tract"], data["lsf_glot"],
        data["frame_energy"], excitation, signal, default_config, user_config,
    )
    return {
        "signal": signal,
        "excitation_signal": excitation,
        "sample_rate": int(data.get("sample_rate", params.fs)),
    }


def synthesize_file(data, filename, default_config, user_config=""):
    """Synthesize data and write its waveform to a WAV file."""
    result = synthesize(data, default_config, user_config)
    sf.write(filename, result["signal"], result["sample_rate"])
    return result
