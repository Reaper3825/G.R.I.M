#!/usr/bin/env bash
# Default to CUDA 12 when 11.x is detected (CUTLASS/flash-attention requires 12+).
# Source from run_train_on_anvil.sh or run_train_on_bridges2.sh before build:
#   source "$(dirname "${BASH_SOURCE[0]}")/ensure_cuda12_for_training.sh" || true
# Optional: set GRIM_CUDA12_ROOT to your CUDA 12 toolkit path, or GRIM_PROJECT_DIR to repo root (script looks for $GRIM_PROJECT_DIR/cuda-12.0).

_ensure_cuda12_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
_ensure_cuda12_project="${GRIM_PROJECT_DIR:-$GRIM_BRIDGES2_DIR}"

# If GRIM_CUDA_ROOT already set and we're not forcing, use it and export PATH/LD_LIBRARY_PATH
if [[ -n "${GRIM_CUDA_ROOT:-}" ]] && [[ -z "${GRIM_CUDA12_FORCE:-}" ]]; then
  export GRIM_CUDA_ROOT
  export PATH="${GRIM_CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  return 0 2>/dev/null || exit 0
fi

NVCC="$(which nvcc 2>/dev/null)" || true
if [[ -z "$NVCC" ]]; then
  # No nvcc in PATH; if user set GRIM_CUDA12_ROOT, use it
  if [[ -n "${GRIM_CUDA12_ROOT:-}" ]] && [[ -x "${GRIM_CUDA12_ROOT}/bin/nvcc" ]]; then
    export GRIM_CUDA_ROOT="$GRIM_CUDA12_ROOT"
    export PATH="${GRIM_CUDA_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
    return 0 2>/dev/null || exit 0
  fi
  echo "ERROR: nvcc not found. module load cuda or set GRIM_CUDA12_ROOT." >&2
  return 1 2>/dev/null || exit 1
fi

# Parse version from "Cuda compilation tools, release 11.2, V11.2.152" or "release 12.0"
CUDA_VER="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)" || true
CUDA_MAJOR="${CUDA_VER%%.*}"

if [[ -n "$CUDA_MAJOR" ]] && [[ "$CUDA_MAJOR" -ge 12 ]]; then
  # Already 12+; set GRIM_CUDA_ROOT from current nvcc
  CUDAToolkit_ROOT="$(dirname "$(dirname "$NVCC")")"
  export GRIM_CUDA_ROOT="$CUDAToolkit_ROOT"
  export PATH="${GRIM_CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  return 0 2>/dev/null || exit 0
fi

# CUDA 11.x (or unknown) detected — default to project CUDA 12 if present
CANDIDATE=""
if [[ -n "${GRIM_CUDA12_ROOT:-}" ]] && [[ -x "${GRIM_CUDA12_ROOT}/bin/nvcc" ]]; then
  CANDIDATE="$GRIM_CUDA12_ROOT"
elif [[ -n "$_ensure_cuda12_project" ]] && [[ -x "${_ensure_cuda12_project}/cuda-12.0/bin/nvcc" ]]; then
  CANDIDATE="${_ensure_cuda12_project}/cuda-12.0"
elif [[ -n "$_ensure_cuda12_project" ]] && [[ -x "${_ensure_cuda12_project}/cuda-12/bin/nvcc" ]]; then
  CANDIDATE="${_ensure_cuda12_project}/cuda-12"
elif [[ -x "${_ensure_cuda12_script_dir}/../cuda-12.0/bin/nvcc" ]]; then
  CANDIDATE="$(cd "${_ensure_cuda12_script_dir}/../cuda-12.0" && pwd)"
fi

if [[ -n "$CANDIDATE" ]]; then
  export GRIM_CUDA_ROOT="$CANDIDATE"
  export PATH="${GRIM_CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  echo "[ensure_cuda12] CUDA 11.x detected; using project CUDA 12: $GRIM_CUDA_ROOT" >&2
  return 0 2>/dev/null || exit 0
fi

echo "ERROR: CUDA 11.x detected. CUTLASS (flash-attention) requires CUDA 12+." >&2
[[ -n "$_ensure_cuda12_project" ]] && echo "  Looked for: ${_ensure_cuda12_project}/cuda-12.0 and ${_ensure_cuda12_project}/cuda-12" >&2
echo "  Install CUDA 12 to project space (e.g. toolkitpath=${_ensure_cuda12_project:-/path/to/project}/cuda-12.0), then:" >&2
echo "  - re-run without setting GRIM_CUDA_ROOT (script will auto-use it), or" >&2
echo "  - run: GRIM_CUDA_ROOT=/path/to/cuda-12.0 ./scripts/run_train_on_anvil.sh --build" >&2
return 1 2>/dev/null || exit 1
