#!/bin/bash
#SBATCH --job-name=grim-train
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250124p
#SBATCH --gpus=h100-80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_bridges2_%j.out
#SBATCH --error=logs/train_bridges2_%j.err
#
# GRIM-text training on PSC Bridges-2. This is the authoritative train_gpu
# runtime path used by scripts/run_train_on_bridges2.sh.
#
# Defaults match run_train_on_bridges2.sh (override at submit: sbatch -p/-A/--gpus/-t).
# If your partition QOS caps walltime lower, submit with e.g. sbatch -t 12:00:00 ....
#
# Submit from repo root on Bridges-2:
#   cd /ocean/projects/<alloc>/<user>/G.R.I.M
#   mkdir -p logs
#   export GRIM_BRIDGES2_DIR=$PWD   # optional if you use the default path below
#   sbatch scripts/train_bridges2.sbatch
#
# From your machine (syncs + submits a real batch job via sbatch on Bridges-2):
#   ./scripts/run_train_on_bridges2.sh --sbatch
# Build train_gpu on the cluster first (submitted as a synchronous Slurm batch job):
#   ./scripts/run_train_on_bridges2.sh --build
#
# Do not run this .sbatch file with ./ on a laptop — it only runs on a compute node
# after sbatch (SLURM_JOB_ID will be set).
#
# Canonical config: train_gpu loads ai_config.json from the repo root. Do not pass a config override.

set -e
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script runs only on a Bridges-2 compute node after sbatch (SLURM_JOB_ID is unset here)." >&2
  echo "From your Mac — submit a Slurm batch training job:" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --sbatch" >&2
  echo "Build on the cluster first, then batch (or both in one go):" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --build" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --build --sbatch" >&2
  echo "On a Bridges-2 login node: sbatch scripts/train_bridges2.sbatch" >&2
  exit 1
fi

GRIM_ROOT="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis250124p/uwadkins/G.R.I.M}"
cd "$GRIM_ROOT" || {
  echo "ERROR: cannot cd to GRIM_ROOT=$GRIM_ROOT (set GRIM_BRIDGES2_DIR if your clone lives elsewhere)." >&2
  exit 1
}
mkdir -p logs

TRAINING_DIR="resources/models/GRIM-text/training"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$GRIM_ROOT/$BUILD_DIR/train_gpu"

if [[ -n "${GRIM_TRAIN_CONFIG:-}" ]]; then
  echo "ERROR: GRIM_TRAIN_CONFIG is no longer supported by scripts/train_bridges2.sbatch." >&2
  echo "  train_gpu now loads the canonical ai_config.json from the repo root." >&2
  echo "  Update $GRIM_ROOT/ai_config.json before submitting the job." >&2
  exit 1
fi

CONFIG_PATH="$GRIM_ROOT/ai_config.json"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: canonical ai_config.json not found at $CONFIG_PATH" >&2
  exit 1
fi

# Initialize the CUDA runtime environment inside the allocated batch node.
source /etc/profile.d/modules.sh 2>/dev/null || {
  echo "ERROR: Bridges-2 module initialization failed." >&2
  exit 1
}
module load cuda 2>/dev/null || true
module load gcc/13.3.1-p20240614
export GRIM_PROJECT_DIR="$GRIM_ROOT"
source "$GRIM_ROOT/scripts/ensure_cuda12_for_training.sh" 2>/dev/null || true
export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"

GXX_PATH=$(command -v g++)
if [[ -z "$GXX_PATH" ]]; then
  echo "ERROR: gcc/13.3.1-p20240614 did not provide g++ on $(hostname)." >&2
  exit 1
fi
LIBSTDCXX_PATH=$(g++ -print-file-name=libstdc++.so.6)
if [[ ! -f "$LIBSTDCXX_PATH" ]]; then
  echo "ERROR: GCC runtime libstdc++.so.6 not found at $LIBSTDCXX_PATH." >&2
  exit 1
fi
if ! strings "$LIBSTDCXX_PATH" | grep -qx 'GLIBCXX_3.4.32'; then
  echo "ERROR: $LIBSTDCXX_PATH does not provide required ABI GLIBCXX_3.4.32." >&2
  exit 1
fi
LIBSTDCXX_RESOLVED_PATH=$(ldd "$EXE" | awk '$1 == "libstdc++.so.6" && $2 == "=>" { print $3 }')
if [[ -z "$LIBSTDCXX_RESOLVED_PATH" || ! -f "$LIBSTDCXX_RESOLVED_PATH" ]]; then
  echo "ERROR: could not resolve libstdc++.so.6 for $EXE." >&2
  ldd "$EXE" >&2
  exit 1
fi
LIBSTDCXX_CANONICAL_PATH=$(readlink -f "$LIBSTDCXX_PATH")
LIBSTDCXX_RESOLVED_CANONICAL_PATH=$(readlink -f "$LIBSTDCXX_RESOLVED_PATH")
if [[ -z "$LIBSTDCXX_CANONICAL_PATH" || -z "$LIBSTDCXX_RESOLVED_CANONICAL_PATH" ]]; then
  echo "ERROR: failed to canonicalize the expected or resolved libstdc++.so.6 path." >&2
  exit 1
fi
if [[ "$LIBSTDCXX_RESOLVED_CANONICAL_PATH" != "$LIBSTDCXX_CANONICAL_PATH" ]]; then
  echo "ERROR: $EXE is not resolving libstdc++.so.6 from the pinned GCC runtime." >&2
  echo "  expected: $LIBSTDCXX_PATH -> $LIBSTDCXX_CANONICAL_PATH" >&2
  echo "  resolved: $LIBSTDCXX_RESOLVED_PATH -> $LIBSTDCXX_RESOLVED_CANONICAL_PATH" >&2
  ldd "$EXE" >&2
  exit 1
fi

echo "Running train_gpu on $(hostname) at $(date)"
echo "  GRIM_ROOT=$GRIM_ROOT"
echo "  EXE=$EXE"
echo "  config=$CONFIG_PATH"
echo "  C++ runtime=$LIBSTDCXX_RESOLVED_PATH"
"$EXE"
echo "Done at $(date)"
