from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

DEVKIT_COMMIT = "776102861a4fc587c7ca2141647d4562b93b7f15"
DEFAULT_DEVKIT = Path(__file__).resolve().parents[1] / "vendor" / "cascade-devkit"


def load_module(name: str, devkit_root: Path = DEFAULT_DEVKIT):
    path = devkit_root / "self-evaluation-kit" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official devkit file {path}; follow README setup")
    # Refuse silent devkit version drift. The checked-out source remains inspectable.
    actual = subprocess.check_output(
        ["git", "-C", str(devkit_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != DEVKIT_COMMIT:
        raise ValueError(f"Expected devkit {DEVKIT_COMMIT}, found {actual}")
    module_name = f"_cascade_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
