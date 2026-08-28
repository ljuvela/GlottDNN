import numpy as np
import soundfile as sf
from pathlib import Path

import glottdnn_cpp


def test_create_excitation_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    excitation = glottdnn_cpp.synthesis.create_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["excitation_pulses"],
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["hnr_glot"],
        str(config_file),
    )
    assert excitation.ndim == 1
    assert excitation.size > 0


def test_harmonic_modification_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    excitation = glottdnn_cpp.synthesis.create_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["excitation_pulses"],
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["hnr_glot"],
        str(config_file),
    )
    modified = glottdnn_cpp.synthesis.harmonic_modification(
        analyzed["fundf"],
        analyzed["hnr_glot"],
        excitation,
        str(config_file),
    )
    assert modified.shape == excitation.shape
    assert np.all(np.isfinite(modified))


def test_spectral_match_excitation_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    excitation = glottdnn_cpp.synthesis.create_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["excitation_pulses"],
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["hnr_glot"],
        str(config_file),
    )
    modified = glottdnn_cpp.synthesis.spectral_match_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["lsf_glot"],
        excitation,
        str(config_file),
    )
    assert modified.shape == excitation.shape
    assert np.all(np.isfinite(modified))


def test_generate_unvoiced_signal_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    spectrum = np.zeros((2049, analyzed["fundf"].size))
    generated = glottdnn_cpp.synthesis.generate_unvoiced_signal(
        analyzed["fundf"],
        spectrum,
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["frame_energy"],
        np.zeros_like(analyzed["signal"]),
        np.zeros_like(analyzed["signal"]),
        str(config_file),
    )
    assert generated.shape == analyzed["signal"].shape
    assert np.all(np.isfinite(generated))


def test_filter_excitation_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    excitation = glottdnn_cpp.synthesis.create_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["excitation_pulses"],
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["hnr_glot"],
        str(config_file),
    )
    filtered = glottdnn_cpp.synthesis.filter_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["lsf_vocal_tract"],
        excitation,
        np.zeros_like(excitation),
        str(config_file),
    )
    assert filtered.shape == excitation.shape
    assert np.all(np.isfinite(filtered))


def test_fft_filter_excitation_binding():
    config_file = Path(__file__).resolve().parents[1] / "config" / "config_default_16k.cfg"
    signal, _ = sf.read("data/tmp/slt_arctic_a0001.wav", dtype="float64")
    analyzed = glottdnn_cpp.analysis.run_array(signal, str(config_file))
    excitation = glottdnn_cpp.synthesis.create_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        analyzed["excitation_pulses"],
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        analyzed["hnr_glot"],
        str(config_file),
    )
    filtered = glottdnn_cpp.synthesis.fft_filter_excitation(
        analyzed["fundf"],
        analyzed["frame_energy"],
        np.zeros((2049, analyzed["fundf"].size)),
        analyzed["lsf_vocal_tract"],
        analyzed["lsf_glot"],
        excitation,
        np.zeros_like(excitation),
        str(config_file),
    )
    assert filtered.shape == excitation.shape
    assert np.all(np.isfinite(filtered))
