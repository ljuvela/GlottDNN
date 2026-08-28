"""Launch the native GlottDNN command-line programs installed by CMake."""

from pathlib import Path
import subprocess
import sys

import glottdnn_cpp


def _run(program):
    executable = Path(glottdnn_cpp.__file__).resolve().parent / "bin" / program
    return subprocess.call([str(executable), *sys.argv[1:]])


def analysis():
    return _run("Analysis")


def synthesis():
    return _run("Synthesis")


def lsf_post_filter():
    return _run("LsfPostFilter")
