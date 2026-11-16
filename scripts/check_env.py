#!/usr/bin/env python
"""Print the versions that matter for reproducibility (paste the output into an experiment-log row)."""
import importlib
import platform
import subprocess

print("python", platform.python_version(), platform.platform())
for name in ["numpy", "scipy", "pandas", "cv2", "PIL", "sklearn", "yaml", "tensorflow", "keras", "matplotlib"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:12s}", getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(f"{name:12s} MISSING ({type(e).__name__})")
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    print("git commit", commit, "(uncommitted changes!)" if dirty else "(clean)")
except Exception:  # noqa: BLE001
    print("git: not a repository")
