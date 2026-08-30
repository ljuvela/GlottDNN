"""Array-based Python API for GlottDNN analysis and synthesis."""

import numpy as np
import soundfile as sf

import glottdnn_cpp
from .params import ParamWrapper, _resolve_params, _unwrap_params, load_config

ParamWrapper = ParamWrapper
load_config = load_config

__all__ = [
    "ParamWrapper",
    "load_config",
    "analyze",
    "analyze_file",
    "synthesize",
    "synthesize_file",
    "single_pulse_excitation",
    "validate_synthesis_data",
]


def single_pulse_excitation(data, params, verbose=None):
    """Create excitation using the fixed GlottDNN single-pulse model."""
    params = _resolve_params(params)
    if verbose is not None:
        params.verbose = bool(verbose)
    validated = validate_synthesis_data(data, params)
    params_native = params.as_native()
    params_native.excitation_method = glottdnn_cpp.ExcitationMethod.SINGLE_PULSE
    return glottdnn_cpp.synthesis.create_excitation_with_params(
        validated["fundf"], validated["frame_energy"],
        validated["excitation_pulses"], validated["lsf_vocal_tract"],
        validated["lsf_glot"], validated["hnr_glot"], params_native,
    )


def analyze(signal, sample_rate, default_config, user_config="", verbose=None):
    """Analyze a mono signal and return its vocoder parameters as a dictionary."""
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    params = _resolve_params(default_config, user_config, verbose)
    result = dict(glottdnn_cpp.analysis.run_array_with_params(samples, params.as_native()))
    result["sample_rate"] = int(sample_rate)
    return result


def analyze_file(filename, default_config, user_config="", verbose=None):
    """Read a WAV file and return its vocoder parameters as a dictionary."""
    signal, sample_rate = sf.read(filename, dtype="float64")
    return analyze(signal, sample_rate, default_config, user_config, verbose=verbose)


def synthesize(data, default_config, user_config="", verbose=None):
    """Synthesize an analyzed parameter dictionary entirely in memory."""
    params = _resolve_params(default_config, user_config, verbose)
    params_native = params.as_native()
    use_params = True
    data = validate_synthesis_data(data, params)

    frames = data["fundf"].shape[0]
    if use_params:
        excitation = glottdnn_cpp.synthesis.create_excitation_with_params(
            data["fundf"], data["frame_energy"], data["excitation_pulses"],
            data["lsf_vocal_tract"], data["lsf_glot"], data["hnr_glot"], params_native)
    else:
        excitation = glottdnn_cpp.synthesis.create_excitation(
            data["fundf"], data["frame_energy"], data["excitation_pulses"],
            data["lsf_vocal_tract"], data["lsf_glot"], data["hnr_glot"],
            default_config, user_config,
        )
    if params.noise_gain_voiced > 0.0:
        if use_params:
            excitation = glottdnn_cpp.synthesis.harmonic_modification_with_params(
                data["fundf"], data["hnr_glot"], excitation, params_native)
        else:
            excitation = glottdnn_cpp.synthesis.harmonic_modification(
                data["fundf"], data["hnr_glot"], excitation, default_config,
                user_config)
    if params.use_spectral_matching:
        if use_params:
            excitation = glottdnn_cpp.synthesis.spectral_match_excitation_with_params(
                data["fundf"], data["frame_energy"], data["lsf_glot"],
                excitation, params_native)
        else:
            excitation = glottdnn_cpp.synthesis.spectral_match_excitation(
                data["fundf"], data["frame_energy"], data["lsf_glot"],
                excitation, default_config, user_config)

    spectrum = data.get("spectrum")
    if spectrum is None:
        spectrum = np.zeros((2049, frames), dtype=np.float64)
    if use_params:
        signal = glottdnn_cpp.synthesis.fft_filter_excitation_with_params(
            data["fundf"], data["frame_energy"], spectrum,
            data["lsf_vocal_tract"], data["lsf_glot"], excitation,
            np.zeros_like(excitation), params_native)
        signal = glottdnn_cpp.synthesis.generate_unvoiced_signal_with_params(
            data["fundf"], spectrum, data["lsf_vocal_tract"], data["lsf_glot"],
            data["frame_energy"], excitation, signal, params_native)
    else:
        signal = glottdnn_cpp.synthesis.fft_filter_excitation(
            data["fundf"], data["frame_energy"], spectrum,
            data["lsf_vocal_tract"], data["lsf_glot"], excitation,
            np.zeros_like(excitation), default_config, user_config)
        signal = glottdnn_cpp.synthesis.generate_unvoiced_signal(
            data["fundf"], spectrum, data["lsf_vocal_tract"], data["lsf_glot"],
            data["frame_energy"], excitation, signal, default_config,
            user_config)
    return {
        "signal": signal,
        "excitation_signal": excitation,
        "sample_rate": int(data.get("sample_rate", params.fs)),
    }


def validate_synthesis_data(data, params):
    """Validate and normalize arrays before passing them to native synthesis."""
    if not hasattr(data, "keys"):
        raise ValueError("synthesis data must be a mapping")
    required = (
        "fundf", "frame_energy", "excitation_pulses", "lsf_vocal_tract",
        "lsf_glot", "hnr_glot",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("missing synthesis fields: {}".format(", ".join(missing)))

    arrays = {}
    for name in required:
        arrays[name] = np.asarray(data[name], dtype=np.float64)

    frames = arrays["fundf"].shape[0] if arrays["fundf"].ndim == 1 else 0
    if frames == 0:
        raise ValueError("fundf must be a non-empty one-dimensional array")
    for name in ("frame_energy",):
        if arrays[name].shape != (frames,):
            raise ValueError("{} must have shape ({},)".format(name, frames))
    expected_rows = {
        "excitation_pulses": params.paf_pulse_length,
        "lsf_vocal_tract": params.lpc_order_vt,
        "lsf_glot": params.lpc_order_glot,
        "hnr_glot": params.hnr_order,
    }
    for name, rows in expected_rows.items():
        if arrays[name].ndim != 2 or arrays[name].shape != (rows, frames):
            raise ValueError(
                "{} must have shape ({}, {})".format(name, rows, frames)
            )
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("synthesis arrays must contain only finite values")
    if params.use_generic_envelope:
        if "spectrum" not in data:
            raise ValueError("spectrum is required when use_generic_envelope is enabled")
    if "spectrum" in data:
        arrays["spectrum"] = np.asarray(data["spectrum"], dtype=np.float64)
        if arrays["spectrum"].shape != (2049, frames):
            raise ValueError(
                "spectrum must have shape (2049, {})".format(frames)
            )
        if not np.all(np.isfinite(arrays["spectrum"])):
            raise ValueError("spectrum must contain only finite values")
    if "sample_rate" in data:
        sample_rate = data["sample_rate"]
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive number")
    arrays["sample_rate"] = data.get("sample_rate", params.fs)
    return arrays


def synthesize_file(data, filename, default_config, user_config="", verbose=None):
    """Synthesize data and write its waveform to a WAV file."""
    result = synthesize(data, default_config, user_config, verbose=verbose)
    sf.write(filename, result["signal"], result["sample_rate"])
    return result
