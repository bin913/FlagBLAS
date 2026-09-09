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
      for _cr in /usr/local/corex /usr/local/corex-* /usr/local/CoreX-* /opt/corex-*; do
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
    fi
    # Build LD_LIBRARY_PATH: dirs containing a CUDA-10.2-era libcudart must
    # come first (corex lib dirs, then a local /usr/local/cuda-10.2 install).
    _libdirs=""
    for _cr in "${COREX_ROOT}" /usr/local/cuda-10.2; do
      [ -n "$_cr" ] || continue
      for _cd in "$_cr/lib64" "$_cr/lib"; do
        if [ -d "$_cd" ] && ls "$_cd"/libcudart.so.10* >/dev/null 2>&1; then
          case ":$_libdirs:" in *":$_cd:"*) ;; *) _libdirs="$_libdirs$_cd:" ;; esac
        fi
      done
    done
    # Sweep /usr/local and /opt for any other dir shipping a CUDA-10.2
    # libcudart (SONAME libcudart.so.10), in case corex lives elsewhere.
    while IFS= read -r _cd; do
      [ -n "$_cd" ] || continue
      case ":$_libdirs:" in
        *":$_cd:"*) ;;
        *) _libdirs="$_libdirs$_cd:" ;;
      esac
    done < <(find /usr/local /opt -maxdepth 5 -name 'libcudart.so.10*' -type f 2>/dev/null | sed 's#/[^/]*$##' | sort -u)
    for _cr in "${COREX_ROOT}" /usr/local/cuda-10.2; do
      [ -n "$_cr" ] || continue
      for _cd in "$_cr/lib64" "$_cr/lib"; do
        if [ -d "$_cd" ]; then
          case ":$_libdirs:" in *":$_cd:"*) ;; *) _libdirs="$_libdirs$_cd:" ;; esac
        fi
      done
    done
    _libdirs="${_libdirs%:}"
    while [ -n "$_libdirs" ]; do
      case "$_libdirs" in
        *:*) _cd="${_libdirs%%:*}"; _libdirs="${_libdirs#*:}" ;;
        *) _cd="$_libdirs"; _libdirs="" ;;
      esac
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$_cd:"*) ;;
        *) export LD_LIBRARY_PATH="$_cd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
      esac
    done
    # FlagGems backends.yaml sets CPATH for iluvatar as well.
    if [ -d /usr/local/cuda-10.2/include ]; then
      export CPATH=/usr/local/cuda-10.2/include
    fi
    # Emit the resolved env as a workflow annotation so a CI failure can be
    # diagnosed from the annotations alone (plain stdout of self-hosted runs
    # is not fetchable without authentication).
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::warning title=iluvatar env::COREX_ROOT=${COREX_ROOT:-<none>}; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
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
