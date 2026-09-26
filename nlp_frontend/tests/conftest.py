"""
Shared fixtures for the NLP frontend test suite.

PR #9 review item 4 (round 2): the end-to-end solver tests must not be
allowed to silently vanish from a CI run. This fixture tries, in order:
  1. OPTIMSOLVER_BIN environment variable -- CI should set this to the
     path produced by its own build step.
  2. A binary already present under <repo_root>/build/.
  3. A best-effort on-the-fly CMake configure + build.

If none of these produce a usable binary, the fixture calls
pytest.fail() -- a hard failure, not pytest.skip() -- so a missing
solver shows up as a red test in CI output, not a silently absent one.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"


def _candidate_paths():
    yield os.environ.get("OPTIMSOLVER_BIN")
    yield str(BUILD_DIR / "optimsolver")
    yield str(BUILD_DIR / "Release" / "optimsolver")
    yield shutil.which("optimsolver")


def _find_existing_binary():
    for c in _candidate_paths():
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _try_build():
    """Best-effort CMake configure + build. Returns the binary path on
    success, or None on any failure -- never raises."""
    if not (REPO_ROOT / "CMakeLists.txt").exists():
        return None
    try:
        subprocess.run(
            ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "-j"],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return _find_existing_binary()


@pytest.fixture(scope="session")
def optimsolver_path():
    binary = _find_existing_binary()
    if binary is None:
        binary = _try_build()
    if binary is None:
        pytest.fail(
            "optimsolver binary not found and could not be built automatically.\n"
            "Fix this by either:\n"
            f"  - building it yourself: cmake -S {REPO_ROOT} -B {BUILD_DIR} && "
            f"cmake --build {BUILD_DIR}\n"
            "  - or setting OPTIMSOLVER_BIN to an existing binary's path.\n"
            "This is a hard failure, not a skip -- the end-to-end tests are "
            "load-bearing for this frontend's correctness and must not "
            "silently disappear from CI."
        )
    return binary