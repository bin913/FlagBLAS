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
  "ascend"
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
    # Install FlagTree compiler (plain build, no CUDA). The flagtree wheel
    # bundles the `triton` package that flag_blas imports at runtime, so it
    # must actually be installed or import fails later.
    # Version aligned with FlagGems' nvidia backends (flagtree==0.6.1);
    # `===` pins the exact plain 0.6.1 build (the hosted index also serves
    # vendor-tagged 0.6.1+<backend>3.6 wheels).
    uv pip uninstall triton || true
    # Use `uv pip` (not `python3.12 -m pip`): the venv is created by `uv venv`,
    # which does not seed pip, so `-m pip` always fails with
    # "No module named pip" and flagtree is never installed.
    uv pip install flagtree===0.6.1 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    uv pip install -e .
    uv pip install ".[test,nvidia-cuda128]"
    ;;

  iluvatar)
    # --- CI diagnostic: which iluvatar toolchains already exist on this runner?
    # Temporary probe (until the runner provisioning is settled). It runs before
    # anything is installed, so it reports the pristine image, and it only uses
    # find_spec / subprocess, so it is immune to this script's PATH/PYTHONPATH.
    # For every torch/triton/cupy installation it finds it also runs a two-line
    # smoke test, because the presence of a package says nothing about it being
    # usable (the iluvatar IX Triton plugins for cp310 are known to fail at
    # load_dialects, and stock cupy cannot load against corex's libcudart).
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      python - <<'PYEOF' || true
import glob, os, re, subprocess, sys

VENV = sys.executable


def interp_version(exe):
    try:
        return subprocess.run(
            [exe, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception:
        return ""


# interpreter per python X.Y (the venv one first: that is what CI runs on)
interps = {}
for exe in [VENV] + sorted(
    glob.glob("/usr/bin/python3") + glob.glob("/usr/bin/python3.*")
    + glob.glob("/usr/local/bin/python3") + glob.glob("/usr/local/bin/python3.*")
    + glob.glob("/opt/*/bin/python3")
):
    if not os.path.isfile(exe) or not re.fullmatch(r"python3(\.\d+)?", os.path.basename(exe)):
        continue
    v = interp_version(exe)
    if v and v not in interps:
        interps[v] = exe

out = ["venv=%s(py%s)" % (VENV, interp_version(VENV))]
out.append("interps=" + ",".join("%s:%s" % (v, e) for v, e in sorted(interps.items())))

PKG_DIRS = (
    "/usr/local/lib/python3*/site-packages", "/usr/local/lib/python3*/dist-packages",
    "/usr/lib/python3/dist-packages", "/usr/lib/python3*/dist-packages",
    "/usr/local/corex-*/lib64/python3/dist-packages",
    "/usr/local/corex-*/lib64/python3.*/dist-packages",
    "/usr/local/corex-*/lib64/python3/site-packages",
    "/usr/local/corex-*/lib64/python3.*/site-packages",
    "/opt/*/lib/python3*/site-packages", "/usr/local/apps_whl",
)
found = {}
for pat in PKG_DIRS:
    for d in sorted(glob.glob(pat)):
        hits = [n for n in ("torch", "triton", "cupy", "flagtree") if os.path.isdir(os.path.join(d, n))]
        if hits:
            found[d] = hits


def pyver_of(path):
    m = re.search(r"python3\.(\d+)", path)
    return "3.%s" % m.group(1) if m else ""


def smoke(path, code):
    exe = interps.get(pyver_of(path), VENV)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(path) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = "/usr/local/corex-4.4.0/lib64:/usr/local/corex/lib64:" + env.get("LD_LIBRARY_PATH", "")
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    except Exception as exc:
        return "%s[%s] EXC:%s" % (path, exe, type(exc).__name__)
    if r.returncode == 0:
        return "%s[%s] OK:%s" % (path, exe, r.stdout.strip().splitlines()[-1][:60] if r.stdout.strip() else "?")
    tail = (r.stdout + r.stderr).strip().splitlines()
    return "%s[%s] FAIL:%s" % (path, exe, (tail[-1] if tail else "?")[:140])


TRITON_CODE = (
    "import triton;"
    "from triton._C.libtriton import ir, iluvatar;"
    "mk = getattr(ir, 'context', None) or getattr(ir, 'MLIRContext');"
    "iluvatar.load_dialects(mk());"
    "print('triton', triton.__version__, 'IX load_dialects ok')"
)
CUPY_CODE = "import cupy;print('cupy', cupy.__version__)"

for d, hits in sorted(found.items()):
    if "triton" in hits:
        out.append(smoke(os.path.join(d, "triton"), TRITON_CODE))
    if "cupy" in hits:
        out.append(smoke(os.path.join(d, "cupy"), CUPY_CODE))
    if "torch" in hits:
        out.append("%s[t] = %s" % (os.path.join(d, "torch"), ",".join(sorted(glob.glob(os.path.join(d, "torch-*.dist-info"))))))

for d in sorted(found):
    c = os.path.join(d, "triton", "_C")
    if os.path.isdir(c):
        out.append("_C %s = %s" % (c, ",".join(sorted(os.listdir(c)))))

out.append("corex=" + ",".join(sorted(glob.glob("/usr/local/corex*") + glob.glob("/usr/local/cuda*"))))
out.append("corexpy=" + ",".join(sorted(glob.glob("/usr/local/corex-*/lib*/python3*"))))
out.append("pythons=" + ",".join(sorted(glob.glob("/usr/local/lib/python3*") + glob.glob("/usr/lib/python3*"))))

msg = "\n".join(out)
print("::warning title=iluvatar runner probe::" + msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))
PYEOF
    fi

    # Iluvatar PyTorch/CuPy are bundled with the corex driver (cp310 wheels
    # under /usr/local/corex-*/lib64/python3/dist-packages, reachable through
    # the global PYTHONPATH or by baking it into the venv). Reusing that env is
    # the only way to get a working cupy (no corex cupy wheel is hosted on the
    # flagos mirror), so detect it first and skip any vanilla-torch install.
    if [ -z "$ILUVATAR_COREX_PYDIR" ]; then
      for _cpd in /usr/local/corex-*/lib64/python3/dist-packages /usr/local/corex-*/lib/python3/dist-packages \
                  /usr/local/corex-*/lib64/python3/site-packages /usr/local/corex-*/lib/python3/site-packages \
                  /usr/local/corex-*/lib64/python3.*/dist-packages /usr/local/corex-*/lib/python3.*/dist-packages \
                  /usr/local/corex-*/lib64/python3.*/site-packages /usr/local/corex-*/lib/python3.*/site-packages; do
        if [ -d "$_cpd/torch" ] && [ -d "$_cpd/cupy" ]; then
          export ILUVATAR_COREX_PYDIR="$_cpd"
          break
        fi
      done
    fi

    if [ -n "$ILUVATAR_COREX_PYDIR" ]; then
      echo "Using bundled corex python env: ${ILUVATAR_COREX_PYDIR}"
      # Make the bundled torch/cupy reachable from the venv in every later
      # step (the CI test step only does `source .venv/bin/activate`).
      export PYTHONPATH="${ILUVATAR_COREX_PYDIR}${PYTHONPATH:+:$PYTHONPATH}"
      printf '\n# Source bundled corex python env (required by corex PyTorch/CuPy)\nexport PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}"\n' "$ILUVATAR_COREX_PYDIR" >> .venv/bin/activate
      echo "Baked corex python env into .venv/bin/activate: ${ILUVATAR_COREX_PYDIR}"

      # The bundled env carries torch/cupy but no Triton compiler, and
      # flag_blas imports triton while loading. FlagTree bundles the triton
      # package for the IX backend (same as FlagGems uses for iluvatar); take
      # the iluvatar build, which is also published for the cp310 the bundled
      # env runs on (the corex triton wheel on the iluvatar index is cp312).
      uv pip uninstall triton || true
      uv pip install "flagtree==0.5.1+iluvatar3.1" \
          --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
          --extra-index-url https://mirrors.aliyun.com/pypi/simple \
          --index-strategy unsafe-best-match || {
            echo "::error title=iluvatar flagtree install failed::uv pip install flagtree==0.5.1+iluvatar3.1"
            exit 1
          }
      echo "::warning title=iluvatar setup::flagtree installed"
    else
      echo "::warning title=iluvatar setup::no bundled corex python env; installing corex torch + cupy from the flagos/aliyun mirrors (cp312)."
      # Mirrors FlagGems backends.yaml (iluvatar): python 3.12 + pinned corex
      # torch/torchaudio/torchvision/triton + numpy<2 from the
      # flagos-pypi-iluvatar index (aliyun is only a transitive-deps mirror).
      uv pip install torch==2.7.1+corex.4.4.0 torchaudio==2.7.1+corex.4.4.0 torchvision==0.22.1+corex.4.4.0 triton==3.1.0+corex.4.4.0 "numpy<2" \
          --index-url https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple \
          --extra-index-url https://mirrors.aliyun.com/pypi/simple \
          --index-strategy unsafe-best-match || {
            echo "::error title=iluvatar torch install failed::uv pip install corex torch from flagos-pypi-iluvatar"
            exit 1
          }
      # CuPy: the tests build their GPU reference through cupy, so it must be
      # importable. No corex cupy wheel is hosted on the flagos index, so take
      # the stock one (via the aliyun mirror). The corex driver reports CUDA
      # 10.2, which rules out cupy-cuda11x/cuda12x (both need a newer driver
      # and die with "cudaErrorInsufficientDriver"); cupy-cuda102 is the only
      # build whose runtime matches, and 12.3.0 is its last release.
      # cupy dlopens the CUDA runtime from LD_LIBRARY_PATH (it bundles no CUDA
      # libraries), which is why corex's lib64 must be on the path -- that is
      # already arranged by set-env.sh / .venv/bin/activate.
      uv pip install "cupy-cuda102==12.3.0" "numpy<2" \
          --index-url https://mirrors.aliyun.com/pypi/simple || {
            echo "::error title=iluvatar cupy install failed::uv pip install cupy-cuda102==12.3.0"
            exit 1
          }
    fi

    # Install FlagBLAS editable. Base deps do not include torch, so a normal
    # install cannot replace the bundled/corex torch.
    if ! uv pip install -e . 2>&1 | tee /tmp/iluvatar-flagblas-install.log; then
      echo "::error title=iluvatar flagblas install failed::$(tail -8 /tmp/iluvatar-flagblas-install.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi

    # Test deps. numpy is pinned <2 explicitly: the corex-bundled numpy is 1.x
    # and the corex torch/cupy are built against the numpy 1.x C API.
    if ! uv pip install pytest "numpy<2" scipy distro gitpython pyyaml coverage pytest-md-report \
         --index-url https://mirrors.aliyun.com/pypi/simple 2>&1 | tee /tmp/iluvatar-testdeps.log; then
      echo "::error title=iluvatar test deps install failed::$(tail -8 /tmp/iluvatar-testdeps.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi
    echo "::warning title=iluvatar setup::testdeps installed"

    # Sanity check: make sure torch comes from the corex build, that torch.cuda
    # can actually initialize against the corex driver, and that cupy imports
    # (the tests build their GPU reference through cupy, so a broken cupy must
    # fail here with the real traceback instead of as a collection error).
    # One consolidated annotation (GitHub caps warnings per step) with the
    # facts needed to debug a missing corex runtime library.
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      # `|| true` everywhere: pipefail + set -e would abort on a non-matching
      # grep or an unreadable dir.
      _diag="corex=$(ls -d /usr/local/corex* /opt/corex* 2>/dev/null | tr '\n' ',' || true)"
      # Locate the corex python packages, if any: they carry the only cupy that
      # can load against corex's libcudart, so their absence is the first thing
      # to check when the cupy import below fails.
      _diag="${_diag} pydirs=$(ls -d /usr/local/corex-*/lib*/python3*/{dist,site}-packages 2>/dev/null | tr '\n' ',' || true)"
      _diag="${_diag} cupyenvs=$(find /usr/local /opt -maxdepth 8 -type d -name cupy 2>/dev/null | head -4 | tr '\n' ',' || true)"
      _diag="${_diag} ixthunk=$(find /usr/local /opt -maxdepth 6 -name 'libixthunk.so*' 2>/dev/null | tr '\n' ',' || true)"
      _diag="${_diag} unresolved=$(ldd .venv/lib/python*/site-packages/torch/lib/libtorch_python.so 2>/dev/null | grep 'not found' | tr '\n' ',' || true)"
      _diag="${_diag} ldpath=$(printf '%s' "${LD_LIBRARY_PATH:-}" | head -c 1200 || true)"
      echo "::warning title=iluvatar diag::${_diag}"
    fi
    set +e
    python - <<'PYEOF'
import sys, os, importlib.metadata, traceback
try:
    import torch
    tdir = os.path.dirname(torch.__file__)
    dist = importlib.metadata.version("torch")
    # Corex builds are tagged +corex (mirror install lands in the plain venv
    # site-packages, so a path check alone would falsely flag it as vanilla;
    # the bundled env under /usr/local/corex-* is covered by the dist tag too).
    is_corex = "+corex" in dist or "/corex" in tdir
    print("torch dir:", tdir, "| corex:", is_corex)
    print("torch dist:", dist)
    if not is_corex:
        raise RuntimeError(f"vanilla torch in venv, expected corex build (dist={dist})")
    torch.cuda.init()
    print("torch.cuda available:", torch.cuda.is_available(),
          "| count:", torch.cuda.device_count(),
          "| name:", torch.cuda.get_device_name(0))
    if torch.cuda.device_count() == 0:
        raise RuntimeError("no iluvatar device visible to torch")
    try:
        import cupy
    except ImportError as e:
        if "libcudart.so.10.2" in str(e):
            raise RuntimeError(
                "cupy cannot load against this corex runtime: corex's "
                "libcudart.so.10.2 exports its symbols under the CUDART "
                "version node, while wheels built against stock CUDA 10.2 "
                "need them under the libcudart.so.10.2 node. Only the "
                "corex-built cupy loads here, so install the corex python "
                "packages on this runner (e.g. "
                "corex-<ver>/lib64/python3/dist-packages, which ship "
                "torch/cupy built for corex); setup reuses that env "
                "automatically."
            ) from e
        raise
    print("cupy:", cupy.__version__)
    from cupy_backends.cuda.libs import cublas
    print("cublas wrapper OK:", cublas is not None)
except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("::error title=iluvatar torch/cupy sanity check failed::" + tb.replace("%", "%25").replace("\n", "%0A"))
    sys.exit(1)
PYEOF
    SANITY_RC=$?
    set -e
    if [ $SANITY_RC -ne 0 ]; then
      echo "::error title=iluvatar setup::torch/cupy sanity check failed with rc=${SANITY_RC}"
      exit 1
    fi
    echo "::warning title=iluvatar setup::torch/cupy sanity check passed"

    # Mirror FlagGems: bake the corex/CUDA-10.2 runtime env into
    # .venv/bin/activate so any later `source .venv/bin/activate` (the CI
    # test step) resolves the CUDA-10.2 runtime shipped with corex
    # (otherwise torch import fails with "undefined symbol:
    # cudaProfilerInitialize").
    if { [ -n "$COREX_ROOT" ] && [ -d "$COREX_ROOT" ]; } || [ -d /usr/local/cuda-10.2 ]; then
      {
        echo ""
        echo "# --- FlagBLAS corex runtime env (iluvatar) ---"
        if [ -n "$COREX_ROOT" ] && [ -d "$COREX_ROOT" ]; then
          echo "export COREX_ROOT=\"${COREX_ROOT}\""
          for _cd in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib"; do
            if [ -d "$_cd" ]; then
              echo "case \":\${LD_LIBRARY_PATH:-}:\" in *\":${_cd}:\"*) ;; *) export LD_LIBRARY_PATH=\"${_cd}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\" ;; esac"
            fi
          done
          echo "export PATH=\"${COREX_ROOT}/bin\${PATH:+:\$PATH}\""
        fi
        for _cd in /usr/local/cuda-10.2/lib64 /usr/local/cuda-10.2/lib; do
          if [ -d "$_cd" ]; then
            echo "case \":\${LD_LIBRARY_PATH:-}:\" in *\":${_cd}:\"*) ;; *) export LD_LIBRARY_PATH=\"${_cd}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\" ;; esac"
          fi
        done
        if [ -d /usr/local/cuda-10.2/include ]; then
          echo "export CPATH=/usr/local/cuda-10.2/include"
        fi
        echo "# --- end FlagBLAS corex runtime env ---"
      } >> .venv/bin/activate
      echo "Baked corex runtime env into .venv/bin/activate: ${COREX_ROOT:-/usr/local/cuda-10.2}"
    fi
    ;;

  ascend)
    # Install PyTorch (CPU build) and torch-npu for Ascend NPU
    uv pip install torch==2.10.0+cpu torch-npu==2.10.0 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple

    # Install FlagTree compiler for Ascend
    uv pip uninstall triton || true
    uv pip install flagtree==0.6.0+ascend3.5 \
        --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple

    # Install FlagBLAS in editable mode
    uv pip install -e .
    uv pip install ".[test]"
    ;;
  hygon)
    # Install PyTorch for Hygon DCU (ROCm/HIP).
    # The flagos-pypi-hygon index only hosts vendor wheels, so add a general
    # PyPI mirror (same one FlagGems uses) to resolve torch's transitive deps.
    # --index-strategy unsafe-best-match is required: under uv's default
    # first-index strategy, torch is found on the aliyun mirror and the DTK
    # build from the flagos index is never considered.
    UV_INDEX_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/simple"
    UV_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

    uv pip install torch==2.9.0+das.opt1.dtk2604 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL} \
        --index-strategy unsafe-best-match || {
          echo "::error title=hygon torch install failed::uv pip install torch==2.9.0+das.opt1.dtk2604 (indexes: ${UV_INDEX_URL}, ${UV_EXTRA_INDEX_URL})"
          exit 1
        }
    echo "::warning title=hygon setup::torch installed"

    # Install FlagTree compiler for Hygon DCU
    uv pip uninstall triton || true
    uv pip install flagtree==0.5.1+hcu3.1 \
        --index-url ${UV_INDEX_URL} \
        --extra-index-url ${UV_EXTRA_INDEX_URL} \
        --index-strategy unsafe-best-match || {
          echo "::error title=hygon flagtree install failed::uv pip install flagtree==0.5.1+hcu3.1"
          exit 1
        }
    echo "::warning title=hygon setup::flagtree installed"

    # Install FlagBLAS without touching the DTK-patched torch. pyproject.toml
    # declares `torch>=2.6.0`; without --no-deps the dependency resolver
    # replaces the DTK build with the newest CUDA torch from the extra index.
    if ! uv pip install -e . --no-deps --no-build-isolation \
         --index-url ${UV_EXTRA_INDEX_URL} 2>&1 | tee /tmp/flagblas-install.log; then
      echo "::error title=hygon flagblas install failed::$(tail -8 /tmp/flagblas-install.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi
    echo "::warning title=hygon setup::flagblas installed"

    # Test deps. `cupy-cuda12x` is excluded: it is NVIDIA-only and would pull
    # a CUDA runtime that conflicts with the DTK stack.
    # sqlalchemy/packaging/pybind11 are FlagBLAS runtime deps that were skipped
    # by the --no-deps install above (sqlalchemy is imported at module load time
    # via flag_blas.utils.models).
    # numpy must stay on 1.x: the DTK-patched torch 2.9.0 is built against the
    # numpy 1.x C API ("_ARRAY_API not found" under numpy 2.x).
    if ! uv pip install pytest numpy\<2 scipy distro gitpython pyyaml coverage pytest-md-report \
         sqlalchemy packaging pybind11 \
         --index-url ${UV_EXTRA_INDEX_URL} 2>&1 | tee /tmp/hygon-testdeps.log; then
      echo "::error title=hygon test deps install failed::$(tail -8 /tmp/hygon-testdeps.log | tr '\n' ' ' | head -c 1500)"
      exit 1
    fi
    echo "::warning title=hygon setup::testdeps installed"

    # Sanity check: make sure the DTK-patched torch survived the installs above.
    # NOTE: torch.__version__ drops the +das.opt1.dtk2604 local tag (it reports
    # "2.9.0"), so check the installed distribution version instead.
    set +e
    python - <<'PYEOF'
import sys, importlib.metadata, traceback
try:
    dist = importlib.metadata.version("torch")
    print("hygon torch dist:", dist)
    assert dist.startswith("2.9.0+das.opt1.dtk2604"), \
        f"unexpected torch distribution: {dist}"
    import torch
    print("torch.__version__:", torch.__version__)
    print("torch.version.hip:", getattr(torch.version, "hip", None))
except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("::error title=hygon torch sanity check failed::" + tb.replace("%", "%25").replace("\n", "%0A"))
    sys.exit(1)
PYEOF
    SANITY_RC=$?
    set -e
    if [ $SANITY_RC -ne 0 ]; then
      echo "::error title=hygon torch sanity check failed::sanity check failed with rc=${SANITY_RC}"
      exit 1
    fi
    echo "::warning title=hygon setup::sanity check passed"

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
