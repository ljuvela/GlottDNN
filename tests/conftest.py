import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build-cmake"
sys.path.insert(0, str(BUILD_DIR))

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    library_path = str(Path(conda_prefix) / "lib")
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    os.environ[variable] = library_path + os.pathsep + os.environ.get(variable, "")
