#!/usr/bin/env bash
# Run on the Bridges-2 (Linux) repo root after git pull. Used by run_train_on_bridges2.sh when FAS sync is on.
# Skips forced submodule fetch/checkout when flash-attention matches the superproject gitlink and (pinned path)
# csrc/cutlass is already at CUTLASS_PIN — avoids clobbering a good tree. Always runs patch steps (idempotent).
# This script only owns flash-attention/cutlass. It must not deinitialize vcpkg.
#
# Env:
#   CUTLASS_PIN           Required for pinned path (default matches run_train_on_bridges2.sh).
#   GRIM_USE_LATEST_CUTLASS=1  Use upstream cutlass init only (no pin checkout).

set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

CUTLASS_PIN="${CUTLASS_PIN:-bbe579a9e3beb6ea6626d9227ec32d0dae119a49}"

apply_patches_pinned() {
  if [[ ! -d external/flash-attention/csrc/cutlass ]]; then
    return 0
  fi
  (cd external/flash-attention/csrc/cutlass && git apply -p1 < ../../../../scripts/patches/cutlass-stride-nvcc-fix.patch 2>/dev/null || true)
  if [[ ! -d external/flash-attention ]]; then
    return 0
  fi
  (cd external/flash-attention && (git apply -p1 < ../../scripts/patches/flash-attention-bwd-template-fix.patch 2>/dev/null || (sed -i.bak -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, true>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, true>/g' csrc/flash_attn/src/flash_bwd_kernel.h))) || true
}

if [[ "${GRIM_USE_LATEST_CUTLASS:-0}" == "1" ]]; then
  WANT_FA=$(git ls-tree HEAD external/flash-attention 2>/dev/null | awk '{print $3}' || true)
  HAVE_FA=$(git -C external/flash-attention rev-parse HEAD 2>/dev/null || true)
  if [[ -n "$WANT_FA" && "$WANT_FA" == "$HAVE_FA" ]]; then
    echo "[bridges2_ensure_flash_attention] flash-attention already at superproject gitlink ($WANT_FA); skipping submodule refresh (latest-cutlass mode)."
  else
    echo "[bridges2_ensure_flash_attention] refreshing flash-attention (latest-cutlass mode)..."
    git submodule update --init --force external/flash-attention
    (cd external/flash-attention && git submodule update --init csrc/cutlass)
  fi
  exit 0
fi

# Pinned CUTLASS (default)
WANT_FA=$(git ls-tree HEAD external/flash-attention 2>/dev/null | awk '{print $3}' || true)
HAVE_FA=$(git -C external/flash-attention rev-parse HEAD 2>/dev/null || true)
HAVE_CUT=$(git -C external/flash-attention/csrc/cutlass rev-parse HEAD 2>/dev/null || true)

if [[ -n "$WANT_FA" && -n "$HAVE_FA" && "$WANT_FA" == "$HAVE_FA" && -n "$HAVE_CUT" && "$HAVE_CUT" == "$CUTLASS_PIN" ]]; then
  echo "[bridges2_ensure_flash_attention] flash-attention=$WANT_FA cutlass=$CUTLASS_PIN already present; skipping submodule fetch/checkout (applying patches only)."
  apply_patches_pinned
else
  echo "[bridges2_ensure_flash_attention] refreshing flash-attention / cutlass (want_fa=${WANT_FA:-?} have_fa=${HAVE_FA:-?} have_cut=${HAVE_CUT:-?})..."
  git submodule update --init --force external/flash-attention
  (cd external/flash-attention && git submodule update --init --force csrc/cutlass)
  (cd external/flash-attention/csrc/cutlass && git fetch origin && git checkout -f "$CUTLASS_PIN")
  apply_patches_pinned
fi
