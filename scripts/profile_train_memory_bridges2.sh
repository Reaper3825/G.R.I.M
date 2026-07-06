#!/usr/bin/env bash
# Profile GRIM-text train_gpu memory on Bridges-2 from inside a Slurm GPU allocation.
#
# Preferred use on Bridges-2 login node:
#   bash scripts/profile_train_memory_bridges2.sh --mode nsys --gpu-type h100-80 --time 2:00:00
#
# The script self-launches with srun unless --inside-allocation is supplied. It writes:
#   resources/models/GRIM-text/training/logs/memory_profiles/<timestamp>/
#     manifest.json
#     command.txt
#     nvidia_smi_samples.csv
#     summary.txt
#     train_gpu.stdout.log
#     train_gpu.stderr.log
#     train_gpu_memory.nsys-rep       when nsys is available
# When launched from a local machine without srun, it runs over SSH on Bridges-2
# and mirrors the remote text/CSV/log artifacts back to the matching local profile folder.

set -euo pipefail

MODE="auto"
PARTITION="${GRIM_BRIDGES2_PARTITION:-GPU-shared}"
ACCOUNT="${GRIM_BRIDGES2_ACCOUNT:-cis250124p}"
GPU_TYPE="${GRIM_BRIDGES2_GPU_TYPE:-h100-80}"
TIME_LIMIT="${GRIM_BRIDGES2_TIME_LIMIT:-2:00:00}"
CPUS_PER_TASK="${GRIM_BRIDGES2_PROFILE_CPUS:-8}"
SAMPLE_MS="${GRIM_BRIDGES2_PROFILE_SAMPLE_MS:-100}"
OUTPUT_ROOT=""
TRAIN_EXE=""
GPU_ID="auto"
INSIDE_ALLOCATION=0
TRAIN_ARGS=(--training)
ORIGINAL_ARGS=("$@")

shell_join() {
  local out=""
  local arg
  local quoted
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out+="$quoted "
  done
  printf '%s' "${out% }"
}

usage() {
  cat <<'EOF'
Profile GRIM-text train_gpu memory on Bridges-2 from inside a Slurm GPU allocation.

Preferred use on Bridges-2 login node:
  bash scripts/profile_train_memory_bridges2.sh --mode nsys --gpu-type h100-80 --time 2:00:00

The script self-launches with srun unless --inside-allocation is supplied. It writes:
  resources/models/GRIM-text/training/logs/memory_profiles/<timestamp>/
    manifest.json
    command.txt
    nvidia_smi_samples.csv
    summary.txt
    train_gpu.stdout.log
    train_gpu.stderr.log
    train_gpu_memory.nsys-rep       when nsys is available

  When launched from a local machine without srun, it runs over SSH on Bridges-2
  and mirrors the remote text/CSV/log artifacts back to the matching local profile folder.

Options:
  --mode auto|nsys|smi       auto prefers Nsight Systems, then nvidia-smi polling.
  --partition NAME           Slurm partition. Default: GPU-shared.
  --account ID               ACCESS allocation. Default: GRIM_BRIDGES2_ACCOUNT or cis250124p.
  --gpu-type TYPE            h100-80, v100-32, v100-16, l40s-48. Default: h100-80.
  --time LIMIT               Slurm time limit. Default: GRIM_BRIDGES2_TIME_LIMIT or 2:00:00.
  --cpus N                   CPUs per task for the profiler run. Default: 8.
  --sample-ms N              nvidia-smi sampling interval. Default: 100.
  --gpu-id ID|auto|all        nvidia-smi target. auto uses CUDA_VISIBLE_DEVICES inside Slurm.
  --train-exe PATH           Override train_gpu path.
  --output-root PATH         Override output root.
  --inside-allocation        Run directly; do not call srun. Useful inside salloc/sbatch.
  --                         Arguments after -- replace the default train_gpu args.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS_PER_TASK="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --train-exe) TRAIN_EXE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --inside-allocation) INSIDE_ALLOCATION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; TRAIN_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$MODE" in
  auto|nsys|smi) ;;
  *) echo "ERROR: --mode must be auto, nsys, or smi (got: $MODE)" >&2; exit 1 ;;
esac
[[ "$SAMPLE_MS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --sample-ms must be a positive integer" >&2; exit 1; }
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --cpus must be a positive integer" >&2; exit 1; }
[[ -n "$ACCOUNT" ]] || { echo "ERROR: --account or GRIM_BRIDGES2_ACCOUNT is required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_DIR="$REPO_ROOT/resources/models/GRIM-text/training"
DEFAULT_TRAIN_EXE="$TRAINING_DIR/TrainingLoop/build/train_gpu"
if [[ -z "$TRAIN_EXE" ]]; then
  TRAIN_EXE="$DEFAULT_TRAIN_EXE"
fi

pull_remote_profile_artifacts_to_local() {
  local remote_output_log="$1"
  local remote_profile_dir
  local local_profile_dir
  local artifacts=(manifest.json command.txt summary.txt nvidia_smi_samples.csv train_gpu.stdout.log train_gpu.stderr.log)
  local artifact
  local remote_artifact_path
  local local_artifact_path
  local q_remote_artifact_path
  local pulled_count=0

  remote_profile_dir="$(sed -n 's/^\[Bridges-2 profile\] output: //p' "$remote_output_log" | tail -n 1)"
  if [[ -z "$remote_profile_dir" ]]; then
    echo "[Bridges-2 profile] no remote profile output path was printed; artifact pull skipped." >&2
    return 0
  fi

  local_profile_dir="$TRAINING_DIR/logs/memory_profiles/$(basename "$remote_profile_dir")"
  mkdir -p "$local_profile_dir"

  for artifact in "${artifacts[@]}"; do
    remote_artifact_path="$remote_profile_dir/$artifact"
    local_artifact_path="$local_profile_dir/$artifact"
    q_remote_artifact_path="$(printf '%q' "$remote_artifact_path")"

    if ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        "$BRIDGES2_SSH" "test -f $q_remote_artifact_path && cat $q_remote_artifact_path" > "$local_artifact_path"; then
      echo "[Bridges-2 profile] pulled $artifact: $local_artifact_path"
      pulled_count=$((pulled_count + 1))
    else
      rm -f "$local_artifact_path"
      echo "[Bridges-2 profile] remote artifact missing or pull failed: $remote_artifact_path" >&2
    fi
  done

  echo "[Bridges-2 profile] pulled $pulled_count profile artifact(s) into $local_profile_dir"
}

if [[ "$INSIDE_ALLOCATION" != "1" && -z "${SLURM_JOB_ID:-}" && ! $(command -v srun 2>/dev/null || true) ]]; then
  BRIDGES2_USER="${GRIM_BRIDGES2_USER:-uwadkins}"
  BRIDGES2_SSH="${GRIM_BRIDGES2_SSH:-bridges2}"
  if [[ "$BRIDGES2_SSH" == "bridges2" ]] && ! grep -q "Host bridges2" ~/.ssh/config 2>/dev/null; then
    BRIDGES2_SSH="$BRIDGES2_USER@bridges2.psc.edu"
  fi
  BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/${ACCOUNT}/${BRIDGES2_USER}/G.R.I.M}"
  REMOTE_RUNTIME_DIR="$BRIDGES2_DIR/.grim_bridges2_runtime"
  REMOTE_SCRIPT="$REMOTE_RUNTIME_DIR/profile_train_memory_bridges2.$$.${RANDOM}.sh"
  REMOTE_RUNTIME_DIR_Q="$(printf '%q' "$REMOTE_RUNTIME_DIR")"
  REMOTE_SCRIPT_Q="$(printf '%q' "$REMOTE_SCRIPT")"
  REMOTE_ARG_STRING="$(shell_join "${ORIGINAL_ARGS[@]}")"

  if ! command -v ssh >/dev/null 2>&1; then
    echo "ERROR: srun is not available locally and ssh is not on PATH." >&2
    echo "Run this on Bridges-2, or install/configure ssh and set GRIM_BRIDGES2_SSH/GRIM_BRIDGES2_DIR." >&2
    exit 1
  fi

  echo "[Bridges-2 profile] srun is not available locally; launching on Bridges-2 via ssh: $BRIDGES2_SSH"
  echo "[Bridges-2 profile] remote repo: $BRIDGES2_DIR"
  LOCAL_LAUNCH_LOG="$(mktemp -t grim_bridges2_profile.XXXXXX.log)"
  set +e
  ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH" \
    "set -u; mkdir -p $REMOTE_RUNTIME_DIR_Q; cat > $REMOTE_SCRIPT_Q; chmod 700 $REMOTE_SCRIPT_Q; bash $REMOTE_SCRIPT_Q $REMOTE_ARG_STRING; rc=\$?; rm -f $REMOTE_SCRIPT_Q; exit \$rc" < "$0" 2>&1 | tee "$LOCAL_LAUNCH_LOG"
  REMOTE_EXIT=${PIPESTATUS[0]}
  set -e
  pull_remote_profile_artifacts_to_local "$LOCAL_LAUNCH_LOG"
  rm -f "$LOCAL_LAUNCH_LOG"
  exit "$REMOTE_EXIT"
fi

if [[ "$INSIDE_ALLOCATION" != "1" && -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[Bridges-2 profile] requesting Slurm GPU allocation: partition=$PARTITION gpu=$GPU_TYPE account=$ACCOUNT time=$TIME_LIMIT"
  REINVOKE_ARGS=(
    --inside-allocation
    --mode "$MODE"
    --partition "$PARTITION"
    --account "$ACCOUNT"
    --gpu-type "$GPU_TYPE"
    --time "$TIME_LIMIT"
    --cpus "$CPUS_PER_TASK"
    --sample-ms "$SAMPLE_MS"
    --gpu-id "$GPU_ID"
    --train-exe "$TRAIN_EXE"
  )
  if [[ -n "$OUTPUT_ROOT" ]]; then
    REINVOKE_ARGS+=(--output-root "$OUTPUT_ROOT")
  fi
  REINVOKE_ARGS+=(-- "${TRAIN_ARGS[@]}")

  exec srun \
    -p "$PARTITION" \
    -A "$ACCOUNT" \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres="gpu:$GPU_TYPE:1" \
    -t "$TIME_LIMIT" \
    bash "$0" \
      "${REINVOKE_ARGS[@]}"
fi

if [[ ! -x "$TRAIN_EXE" ]]; then
  echo "ERROR: train_gpu executable not found or not executable: $TRAIN_EXE" >&2
  echo "Build first, for example: ./scripts/run_train_on_bridges2.sh --build --gpu-type $GPU_TYPE" >&2
  exit 1
fi

load_runtime_modules() {
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true
  module load nsight-systems 2>/dev/null || module load nsight_systems 2>/dev/null || module load nvhpc 2>/dev/null || true
  export GRIM_PROJECT_DIR="$REPO_ROOT"
  source "$REPO_ROOT/scripts/ensure_cuda12_for_training.sh" 2>/dev/null || true
  export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
  export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"
}

select_mode() {
  if [[ "$MODE" != "auto" ]]; then
    printf '%s\n' "$MODE"
    return 0
  fi
  if command -v nsys >/dev/null 2>&1 && nsys profile --help 2>&1 | grep -q -- '--cuda-memory-usage'; then
    printf '%s\n' "nsys"
  else
    printf '%s\n' "smi"
  fi
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

sample_interval_seconds() {
  awk -v ms="$SAMPLE_MS" 'BEGIN { printf "%.3f", ms / 1000.0 }'
}

resolve_smi_args() {
  SMI_ARGS=()
  if [[ "$GPU_ID" == "all" ]]; then
    return 0
  fi
  local target="$GPU_ID"
  if [[ "$target" == "auto" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]]; then
      target="${CUDA_VISIBLE_DEVICES%%,*}"
    else
      target="0"
    fi
  fi
  SMI_ARGS=(-i "$target")
}

write_manifest() {
  local run_dir="$1"
  local selected_mode="$2"
  local gpu_info="$3"
  local git_rev="unknown"
  git_rev="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  cat > "$run_dir/manifest.json" <<EOF
{
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "mode": "$(json_escape "$selected_mode")",
  "repo_root": "$(json_escape "$REPO_ROOT")",
  "git_rev": "$(json_escape "$git_rev")",
  "train_exe": "$(json_escape "$TRAIN_EXE")",
  "train_args": "$(json_escape "$(shell_join "${TRAIN_ARGS[@]}")")",
  "sample_ms": $SAMPLE_MS,
  "cuda_visible_devices": "$(json_escape "${CUDA_VISIBLE_DEVICES:-}")",
  "gpu_info": "$(json_escape "$gpu_info")"
}
EOF
}

query_gpu_info() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "nvidia-smi unavailable"
    return 0
  fi
  resolve_smi_args
  nvidia-smi "${SMI_ARGS[@]}" --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || \
    printf '%s\n' "nvidia-smi query failed"
}

sample_once() {
  local csv_path="$1"
  local sample_utc
  sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  resolve_smi_args
  nvidia-smi "${SMI_ARGS[@]}" \
    --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      printf '%s,%s\n' "$sample_utc" "$line" >> "$csv_path"
    done
}

sample_until_exit() {
  local pid="$1"
  local csv_path="$2"
  local summary_path="$3"
  local interval
  interval="$(sample_interval_seconds)"

  echo "sample_utc,timestamp,index,uuid,name,memory_used_mib,memory_total_mib,utilization_gpu_pct" > "$csv_path"
  while kill -0 "$pid" 2>/dev/null; do
    sample_once "$csv_path"
    sleep "$interval"
  done
  sample_once "$csv_path"

  local peak_line peak_mib
  peak_line="$(awk -F',' 'NR > 1 { gsub(/^[ \t]+|[ \t]+$/, "", $6); if ($6 + 0 > max) { max = $6 + 0; line = $0 } } END { print line }' "$csv_path")"
  peak_mib="$(awk -F',' 'NR > 1 { gsub(/^[ \t]+|[ \t]+$/, "", $6); if ($6 + 0 > max) max = $6 + 0 } END { if (max == "") max = 0; print max }' "$csv_path")"
  {
    echo "sample_ms=$SAMPLE_MS"
    echo "peak_memory_used_mib=$peak_mib"
    echo "peak_sample=$peak_line"
    echo "samples_csv=$csv_path"
  } >> "$summary_path"
}

run_profiled_command() {
  local run_dir="$1"
  shift
  local cmd=("$@")
  local stdout_path="$run_dir/train_gpu.stdout.log"
  local stderr_path="$run_dir/train_gpu.stderr.log"
  local summary_path="$run_dir/summary.txt"
  local samples_path="$run_dir/nvidia_smi_samples.csv"

  shell_join "${cmd[@]}" > "$run_dir/command.txt"
  touch "$summary_path"
  echo "command=$(shell_join "${cmd[@]}")" >> "$summary_path"

  cd "$REPO_ROOT"
  "${cmd[@]}" > "$stdout_path" 2> "$stderr_path" &
  local train_pid=$!
  set +e
  sample_until_exit "$train_pid" "$samples_path" "$summary_path"
  wait "$train_pid"
  local rc=$?
  set -e
  echo "train_exit_code=$rc" >> "$summary_path"
  echo "stdout=$stdout_path" >> "$summary_path"
  echo "stderr=$stderr_path" >> "$summary_path"
  return "$rc"
}

load_runtime_modules
SELECTED_MODE="$(select_mode)"
if [[ "$SELECTED_MODE" == "nsys" && ! $(command -v nsys 2>/dev/null || true) ]]; then
  echo "ERROR: --mode nsys requested, but nsys is not on PATH after module load attempts." >&2
  echo "Try: module avail nsight" >&2
  exit 1
fi
if [[ "$SELECTED_MODE" == "smi" && ! $(command -v nvidia-smi 2>/dev/null || true) ]]; then
  echo "ERROR: --mode smi requested, but nvidia-smi is not available." >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$TRAINING_DIR/logs/memory_profiles"
fi
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
RUN_DIR="$OUTPUT_ROOT/$RUN_STAMP"
mkdir -p "$RUN_DIR"

GPU_INFO="$(query_gpu_info | tr '\n' ';' | sed 's/;$//')"
write_manifest "$RUN_DIR" "$SELECTED_MODE" "$GPU_INFO"

echo "[Bridges-2 profile] output: $RUN_DIR"
echo "[Bridges-2 profile] mode: $SELECTED_MODE"
if [[ "$MODE" == "auto" && "$SELECTED_MODE" == "smi" && $(command -v nsys 2>/dev/null || true) ]]; then
  echo "[Bridges-2 profile] auto selected smi because this Bridges-2 nsys does not advertise --cuda-memory-usage."
fi
echo "[Bridges-2 profile] GPU: $GPU_INFO"
echo "[Bridges-2 profile] train_gpu: $TRAIN_EXE ${TRAIN_ARGS[*]}"

if [[ "$SELECTED_MODE" == "nsys" ]]; then
  NSYS_REPORT_BASE="$RUN_DIR/train_gpu_memory"
  NSYS_ARGS=(profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt --output="$NSYS_REPORT_BASE")
  if nsys profile --help 2>&1 | grep -q -- '--cuda-memory-usage'; then
    NSYS_ARGS+=(--cuda-memory-usage=true)
    echo "memory_trace_source=nsight_systems_cuda_memory_usage" >> "$RUN_DIR/summary.txt"
  else
    echo "memory_trace_source=nvidia_smi_polling" >> "$RUN_DIR/summary.txt"
    echo "memory_trace_note=This nsys does not advertise --cuda-memory-usage, so the .nsys-rep is a CUDA timeline and peak memory comes from nvidia_smi_samples.csv." >> "$RUN_DIR/summary.txt"
    echo "[Bridges-2 profile] warning: this nsys does not advertise --cuda-memory-usage; collecting CUDA timeline without memory annotation. Peak memory will come from nvidia_smi_samples.csv."
  fi
  if run_profiled_command "$RUN_DIR" nsys "${NSYS_ARGS[@]}" "$TRAIN_EXE" "${TRAIN_ARGS[@]}"; then
    echo "nsys_report_base=$NSYS_REPORT_BASE" >> "$RUN_DIR/summary.txt"
  else
    rc=$?
    echo "ERROR: nsys training run failed with exit code $rc; see $RUN_DIR" >&2
    cat "$RUN_DIR/summary.txt" >&2
    exit "$rc"
  fi
else
  if run_profiled_command "$RUN_DIR" "$TRAIN_EXE" "${TRAIN_ARGS[@]}"; then
    echo "note=nvidia-smi polling is repeatable but can miss spikes shorter than the sample interval; prefer --mode nsys for allocation timelines." >> "$RUN_DIR/summary.txt"
  else
    rc=$?
    echo "ERROR: training run failed with exit code $rc; see $RUN_DIR" >&2
    cat "$RUN_DIR/summary.txt" >&2
    exit "$rc"
  fi
fi

echo "[Bridges-2 profile] complete"
cat "$RUN_DIR/summary.txt"