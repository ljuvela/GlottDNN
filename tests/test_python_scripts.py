import importlib
import sys
from pathlib import Path


def test_python_package_import_smoke():
    package_names = [
        "glottdnn",
        "glottdnn.params",
        "glottdnn.vocoder",
        "glottdnn.data_utils",
        "glottdnn.signal_processing",
    ]
    for name in package_names:
        module = importlib.import_module(name)
        assert module is not None


def test_python_helper_script_import_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    sys.path.insert(0, str(python_dir))
    try:
        script_names = [
            "glott_dnn_script",
            "train_dnn",
            "dnn_classes",
            "reaper_pitch_analysis",
            "native_cli",
        ]
        for name in script_names:
            module = importlib.import_module(name)
            assert module is not None
    finally:
        sys.path.remove(str(python_dir))
