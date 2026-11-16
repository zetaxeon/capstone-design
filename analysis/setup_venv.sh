#!/bin/bash
# Create the project venv (.venv, git-ignored) for the CARLA experiments and install the pinned packages.
# The venv does not see the system or user site-packages, so `pip3 install` outside it cannot change
# the numpy / torch the experiments run on. PYTHONPATH (CARLA PythonAPI for `agents`, ROS) still applies.
# Usage: analysis/setup_venv.sh        (safe to rerun; CARLA_ROOT overrides the CARLA location)
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
CARLA_ROOT=${CARLA_ROOT:-$HOME/carla_ws/CARLA_0.9.16}
VENV=$REPO/.venv
[ -x "$VENV/bin/python" ] || /usr/bin/python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$REPO/requirements-carla.txt"
"$VENV/bin/python" -m pip install --quiet "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
"$VENV/bin/python" -m pip install --quiet --no-deps scipy==1.8.0   # --no-deps: keep numpy 1.26.4 (see requirements-carla.txt)
"$VENV/bin/python" - <<'PY'
import sys, numpy, torch, carla
assert numpy.__version__ == "1.26.4", numpy.__version__
print("venv ok:", sys.prefix, "| numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
