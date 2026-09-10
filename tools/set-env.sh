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

# GitHub Actions runs steps with `bash -e -o pipefail`, so an unexpected
# nonzero return aborts the whole job. Surface the exact failing command via
# a workflow annotation so CI failures can be diagnosed from annotations.
if [ -n "${GITHUB_ACTIONS:-}" ]; then
  trap 'echo "::error title=set-env ::line ${LINENO} rc=$? cmd: ${BASH_COMMAND}"' ERR
fi

SUPPORTED_VENDORS=(
  "nvidia"
  "iluvatar"
  "ascend"
  "hygon"
)

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

# Validate argument count
[ "$#" -eq 1 ] || { echo "Please specify <VENDOR>"; exit 1; }

VENDOR=${1}
valid_vendor "$VENDOR"
if [ "$?" != 0 ]; then
    echo "Invalid vendor '${VENDOR}' specified ..."
    echo "Please specify one of: ${SUPPORTED_VENDORS[@]}"
    exit 1
fi

export BLAS_VENDOR=$VENDOR

case $VENDOR in
  nvidia)
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    ;;
  iluvatar)
    # Locate the real CoreX install. FlagGems' runners use the canonical
    # unversioned /usr/local/corex, but bare runners may only ship the
    # versioned dir (/usr/local/corex-4.4.0) without a symlink, so glob.
    # The corex PyTorch wheels link against the CUDA-10.2 runtime that CoreX
    # ships (libcudart still exports cudaProfilerInitialize there); that lib
    # dir MUST be on LD_LIBRARY_PATH before any `import torch`, otherwise the
    # loader picks the system CUDA 12 libcudart and dies with
    # "undefined symbol: cudaProfilerInitialize, version CUDART".
    export COREX_ROOT=${COREX_ROOT:-}
    if [ -z "$COREX_ROOT" ]; then
      # torch==2.7.1+corex.4.4.0 needs the corex 4.4.0 runtime: some runners
      # ship several corex versions (e.g. corex-4.4.0 and corex-4.5.0) with
      # /usr/local/corex symlinked to the newest, so prefer the explicit
      # 4.4.0 install over the symlink.
      for _cr in /usr/local/corex-4.4.0 /usr/local/corex /usr/local/corex-* /usr/local/CoreX-* /opt/corex-*; do
        if [ -d "$_cr" ] && { [ -d "$_cr/bin" ] || [ -d "$_cr/lib" ] || [ -d "$_cr/lib64" ]; }; then
          export COREX_ROOT="$_cr"
          break
        fi
      done
    fi
    # Fall back to the prefix of a PATH-resolved ixsmi, if any.
    if [ -z "$COREX_ROOT" ]; then
      _ixsmi=$(command -v ixsmi 2>/dev/null || true)
      if [ -n "$_ixsmi" ]; then
        _ixroot="$(cd "$(dirname "$_ixsmi")/.." && pwd 2>/dev/null || true)"
        if [ -n "$_ixroot" ] && { [ -d "$_ixroot/lib" ] || [ -d "$_ixroot/lib64" ]; }; then
          export COREX_ROOT="$_ixroot"
        fi
      fi
    fi
    if [ -z "$COREX_ROOT" ]; then
      echo "WARNING: no corex install found under /usr/local/corex*; corex torch will not work"
    else
      export PATH="${COREX_ROOT}/bin:${PATH}"
      # Triton's iluvatar backend guesses the CUDA-10.2 toolkit through
      # `whereis ixsmi` (taking the parent of its parent) unless CUDA_HOME is
      # set, and it wants corex's own libdevice bitcode (nvvm/libdevice/
      # libdevice.compute_bi.10.bc, not the NVIDIA filename). On runners where
      # ixsmi is also installed under /usr/local/bin that guess lands on
      # /usr/local and every kernel compile dies with
      # "FileNotFoundError: /usr/local/nvvm/libdevice/libdevice.compute_bi.10.bc".
      # corex is the CUDA toolkit here, so point both at it explicitly.
      export CUDA_HOME="${COREX_ROOT}"
      if [ -f "${COREX_ROOT}/nvvm/libdevice/libdevice.compute_bi.10.bc" ]; then
        export TRITON_LIBDEVICE_PATH="${COREX_ROOT}/nvvm/libdevice/libdevice.compute_bi.10.bc"
      fi
    fi
    # Build LD_LIBRARY_PATH with two priorities:
    #   1) dirs whose libcudart.so.10 exports cudaProfilerInitialize (the corex
    #      torch build needs it; the system CUDA 12 cudart lacks it) - these
    #      must win libcudart resolution;
    #   2) every other corex / CUDA-10.2 lib dir, which still provides runtime
    #      libraries such as libixthunk.so. Dropping those dirs would make the
    #      torch import fail with "cannot open shared object file".
    _good=""
    _late=""
    for _cr in "${COREX_ROOT}" /usr/local/cuda-10.2 /usr/local/corex /usr/local/corex-* /opt/corex-*; do
      [ -n "$_cr" ] && [ -d "$_cr" ] || continue
      for _cd in "$_cr/lib64" "$_cr/lib"; do
        if [ -d "$_cd" ] && ls "$_cd"/libcudart.so.10* >/dev/null 2>&1; then
          case ":$_good:" in *":$_cd:"*) ;; *) _good="$_good$_cd:" ;; esac
        fi
      done
    done
    # Sweep /usr/local and /opt for any other dir shipping a CUDA-10.2
    # libcudart (SONAME libcudart.so.10) or the corex runtime loader lib
    # (libixthunk.so, a NEEDED entry of libtorch_python.so), in case corex
    # lives somewhere other than the handful of dirs probed above.
    # `pipefail` + `set -e` would abort here when find hits an unreadable dir.
    _found="$(find /usr/local /opt -maxdepth 5 \
                \( -name 'libcudart.so.10*' -o -name 'libixthunk.so*' \) \( -type f -o -type l \) 2>/dev/null \
              | sed 's#/[^/]*$##' | sort -u || true)"
    while IFS= read -r _cd; do
      [ -n "$_cd" ] || continue
      case ":$_good:" in *":$_cd:"*) ;; *) _good="$_good$_cd:" ;; esac
    done <<< "$_found"
    # Demote (not drop) dirs whose libcudart.so.10 misses the symbol: some
    # corex versions ship a trimmed cudart that keeps the SONAME but drops the
    # deprecated profiler entry point.
    _checked=""
    _demoted=""
    _rest="$_good"
    while [ -n "$_rest" ]; do
      case "$_rest" in
        *:*) _cd="${_rest%%:*}"; _rest="${_rest#*:}" ;;
        *) _cd="$_rest"; _rest="" ;;
      esac
      [ -n "$_cd" ] || continue
      _keep=1
      _lib=$(ls "$_cd"/libcudart.so.10* 2>/dev/null | head -1 || true)
      if [ -n "$_lib" ]; then
        if command -v nm >/dev/null 2>&1; then
          nm -D "$_lib" 2>/dev/null | grep -q "cudaProfilerInitialize" || _keep=0
        elif command -v objdump >/dev/null 2>&1; then
          objdump -T "$_lib" 2>/dev/null | grep -q "cudaProfilerInitialize" || _keep=0
        fi
      fi
      if [ "$_keep" = 1 ]; then
        _checked="$_checked$_cd:"
      else
        case ":$_late:" in *":$_cd:"*) ;; *) _late="$_late$_cd:" ;; esac
        _demoted="${_demoted}${_cd} "
      fi
    done
    _good="$_checked"
    # Every other corex / CUDA-10.2 lib dir, so runtime libs such as
    # libixthunk.so resolve even when the dir ships no usable cudart.
    for _cr in "${COREX_ROOT}" /usr/local/cuda-10.2 /usr/local/corex /usr/local/corex-* /opt/corex-*; do
      [ -n "$_cr" ] && [ -d "$_cr" ] || continue
      for _cd in "$_cr/lib64" "$_cr/lib"; do
        if [ -d "$_cd" ]; then
          case ":$_good:" in
            *":$_cd:"*) ;;
            *) case ":$_late:" in *":$_cd:"*) ;; *) _late="$_late$_cd:" ;; esac ;;
          esac
        fi
      done
    done
    _libdirs="${_good}${_late}"
    _libdirs="${_libdirs%:}"
    if [ -n "$_libdirs" ]; then
      export LD_LIBRARY_PATH="${_libdirs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # The stock CUDA-10.2 wheel cupy asks for its runtime symbols under the
    # `libcudart.so.10.2` version node, which the corex runtime does not
    # provide (it exports them under `CUDART`); setup_vendor.sh builds a
    # forwarding shim for it. The shim has to win the lookup, so it goes in
    # front of everything above -- this is what also makes it effective in the
    # test step, which sources this script *after* .venv/bin/activate.
    if [ -f .venv/lib/cudart-shim/libcudart.so.10.2 ]; then
      export LD_LIBRARY_PATH="$(cd .venv/lib/cudart-shim && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # FlagGems backends.yaml sets CPATH for iluvatar as well.
    if [ -d /usr/local/cuda-10.2/include ]; then
      export CPATH=/usr/local/cuda-10.2/include
    fi
    # Emit the resolved env as a single workflow annotation so a CI failure can
    # be diagnosed from the annotations alone (plain stdout of self-hosted runs
    # is not fetchable without authentication). Keep it to ONE line: GitHub
    # keeps at most 10 warning annotations per step.
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::warning title=iluvatar env::COREX_ROOT=${COREX_ROOT:-<none>}; demoted=[${_demoted:-none}]; libdirs=${_libdirs:-<empty>}; cuda_home=${CUDA_HOME:-<none>}; libdevice=${TRITON_LIBDEVICE_PATH:-<none>}"
    fi
    echo "COREX_ROOT=${COREX_ROOT:-<none>}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
    ;;
  ascend)
    if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
      source /usr/local/Ascend/cann/set_env.sh || true
    fi
    ;;
  hygon)
    # Locate and source the Hygon DTK environment. The DTK-patched PyTorch
    # needs the DTK runtime libs (LD_LIBRARY_PATH) to detect the DCUs at
    # torch import time, so this must run before any python/torch invocation.
    export DTK_ENV=""
    for f in /opt/dtk-26.04/env.sh /opt/dtk/env.sh /usr/local/dtk/env.sh /opt/dtk-*/env.sh /usr/local/dtk-*/env.sh; do
      if [ -f "$f" ]; then
        export DTK_ENV="$f"
        source "$f" || true
        echo "Sourced Hygon DTK environment: $f"
        break
      fi
    done
    if [ -z "$DTK_ENV" ]; then
      echo "WARNING: no DTK env.sh found under /opt/dtk-26.04, /opt/dtk*, /usr/local/dtk*. torch will not see the DCUs."
    fi
    # Explicitly ensure the DTK/hyhal library dirs are on LD_LIBRARY_PATH,
    # in case env.sh does not cover them (torch._C._cuda_init needs them to
    # find the DCU driver).
    if [ -n "$DTK_ENV" ]; then
      DTK_ROOT="${DTK_ENV%/env.sh}"
      for d in "${DTK_ROOT}/lib" "${DTK_ROOT}/lib64" /opt/hyhal/lib /opt/hyhal/lib64; do
        if [ -d "$d" ]; then
          case ":$LD_LIBRARY_PATH:" in
            *":$d:"*) ;;
            *) export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
          esac
        fi
      done
    fi
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    ;;
esac

echo "Environment configured for vendor: ${VENDOR} (BLAS_VENDOR=${BLAS_VENDOR})"
