import os
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build-cmake"
sys.path.insert(0, str(BUILD_DIR))

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    library_path = str(Path(conda_prefix) / "lib")
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    os.environ[variable] = library_path + os.pathsep + os.environ.get(variable, "")


@pytest.fixture(scope="session", autouse=True)
def arctic_sample():
    sample = ROOT / "data" / "tmp" / "slt_arctic_a0001.wav"
    if not sample.exists():
        sample.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "http://festvox.org/cmu_arctic/cmu_arctic/"
            "cmu_us_slt_arctic/wav/arctic_a0001.wav"
        )
        with urlopen(url) as response, sample.open("wb") as output:
            output.write(response.read())
