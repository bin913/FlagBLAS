#!/bin/bash


# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VENDOR=$1

SUPPORTED_VENDORS=(
  "nvidia"
  "iluvatar"
  "hygon"
)
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

[ "$#" -eq 1 ] || { echo "Usage: source tools/setup_vendor.sh <vendor>"; exit 1; }
valid_vendor "$VENDOR" || { echo "Invalid vendor: $VENDOR"; exit 1; }

# Source environment variables if not already set
if [ -z "$BLAS_VENDOR" ]; then
  source tools/set-env.sh "$VENDOR"
fi

echo "Installing FlagBLAS for ${VENDOR} ..."

case $VENDOR in
  nvidia)
    # Install PyTorch and Triton with CUDA support
    uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128
    # Install FlagBLAS in editable mode

    uv pip uninstall triton
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
    python3.12 -m pip install flagtree===0.5.0 $RES
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  iluvatar)
    # Install PyTorch with Corex support
    uv pip install \
      "torch>=2.6.0"

    # Install FlagBLAS in editable mode
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  hygon)
    # Install PyTorch for Hygon DCU (ROCm/HIP).
    # The flagos-pypi-hygon index only hosts vendor wheels, so add a general
    # PyPI mirror (same one FlagGems uses) to resolve torch's transitive deps.
    UV_INDEX_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/simple"
    UV_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

    uv pip install torch==2.9.0+das.opt1.dtk2604 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL}

    # Install FlagTree compiler for Hygon DCU
    uv pip uninstall triton || true
    uv pip install flagtree==0.5.1+hcu3.1 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL}

    # Install FlagBLAS without touching the DTK-patched torch. pyproject.toml
    # declares `torch>=2.6.0`; without --no-deps the dependency resolver
    # replaces the DTK build with the newest CUDA torch from the extra index.
    uv pip install -e . --no-deps --no-build-isolation --index-url ${UV_EXTRA_INDEX_URL}

    # Test deps. `cupy-cuda12x` is excluded: it is NVIDIA-only and would pull
    # a CUDA runtime that conflicts with the DTK stack.
    uv pip install pytest numpy scipy distro gitpython pyyaml coverage pytest-md-report \
        --index-url ${UV_EXTRA_INDEX_URL}

    # Sanity check: make sure the DTK-patched torch survived the installs above.
    python -c "import torch; print('hygon torch:', torch.__version__)" || exit 1

    # Mirror FlagGems' env_source: bake the DTK environment into the venv so
    # that every `source .venv/bin/activate` also loads the DTK runtime.
    # Otherwise torch.cuda init fails at import time ("Found no NVIDIA driver")
    # because the DTK libs are missing from LD_LIBRARY_PATH.
    if [ -n "$DTK_ENV" ]; then
      printf '\n# Source Hygon DTK environment (required by DTK-patched PyTorch)\n[ -f "%s" ] && source "%s" || true\n' "$DTK_ENV" "$DTK_ENV" >> .venv/bin/activate
      echo "Baked DTK environment into .venv/bin/activate: $DTK_ENV"
    fi
    ;;
esac

echo "FlagBLAS installation for ${VENDOR} completed."
