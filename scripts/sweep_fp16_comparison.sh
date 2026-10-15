#!/usr/bin/env bash
# Run selected H=40 methods sequentially from an isolated source snapshot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_ROOT="$PROJECT_ROOT/HeteroViT-MPI"
EPOCHS=40
SEED=42
METHOD_LIST=fp32,fp16_whole_allgather,fp16_whole_allreduce,fp16_delta_allgather,fp16_delta_allreduce
MAX_STEPS=""
DRY_RUN=0
FP16_DIAGNOSTIC=0
VERIFY_DELTA_STATE=0
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --fp16-diagnostic) FP16_DIAGNOSTIC=1; shift ;;
    --verify-delta-state) VERIFY_DELTA_STATE=1; shift ;;
    --seed)
      if (($# < 2)) || [[ ! "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "--seed requires a nonnegative integer" >&2; exit 2
      fi
      SEED="$2"; shift 2 ;;
    --methods)
      if (($# < 2)) || [[ ! "$2" =~ ^[a-z0-9_]+(,[a-z0-9_]+)*$ ]]; then
        echo "--methods requires comma-separated method names" >&2; exit 2
      fi
      METHOD_LIST="$2"; shift 2 ;;
    --epochs|--max-steps)
      option="$1"
      if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$option requires a positive integer" >&2; exit 2
      fi
      if [[ "$option" == --epochs ]]; then EPOCHS="$2"; else MAX_STEPS="$2"; fi
      shift 2 ;;
    -h|--help)
      echo "Usage: bash $0 [--dry-run] [--epochs N] [--max-steps N] [--seed N] [--methods NAME,NAME] [--fp16-diagnostic] [--verify-delta-state]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
IFS=',' read -r -a METHODS <<< "$METHOD_LIST"
CONFIGS=()
declare -A SEEN_METHODS=()
for method in "${METHODS[@]}"; do
  [[ -z "${SEEN_METHODS[$method]:-}" ]] || { echo "Duplicate method: $method" >&2; exit 2; }
  SEEN_METHODS[$method]=1
  case "$method" in
    fp32) CONFIGS+=(local_sgd_h40_fp32_40e.yaml) ;;
    fp16_whole_allgather) CONFIGS+=(local_sgd_h40_fp16_allgather_40e.yaml) ;;
    fp16_whole_allreduce) CONFIGS+=(local_sgd_h40_fp16_40e.yaml) ;;
    fp16_delta_allgather) CONFIGS+=(local_sgd_h40_fp16_delta_allgather_40e.yaml) ;;
    fp16_delta_allgather_ef) CONFIGS+=(local_sgd_h40_fp16_delta_allgather_ef_40e.yaml) ;;
    fp16_delta_allreduce) CONFIGS+=(local_sgd_h40_fp16_delta_allreduce_40e.yaml) ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
  esac
done
ENTRYPOINT=train_mpi.py
if ((VERIFY_DELTA_STATE)); then
  [[ -n "$MAX_STEPS" && "$FP16_DIAGNOSTIC" -eq 0 ]] || {
    echo "--verify-delta-state requires --max-steps and excludes --fp16-diagnostic" >&2; exit 2;
  }
  for method in "${METHODS[@]}"; do
    [[ "$method" == fp16_delta_* ]] || { echo "State smoke requires delta methods" >&2; exit 2; }
  done
  ENTRYPOINT=tools/smoke_delta_training.py
fi
if ((FP16_DIAGNOSTIC)); then
  [[ "$METHOD_LIST" == fp32 && -n "$MAX_STEPS" ]] || {
    echo "--fp16-diagnostic requires --methods fp32 and --max-steps N" >&2; exit 2;
  }
  CONFIGS=(fp16_same_inputs_seed43_smoke.yaml)
  ENTRYPOINT=tools/diagnose_fp16_same_inputs.py
fi
for config in "${CONFIGS[@]}"; do
  [[ -f "$SOURCE_ROOT/configs/mpi/$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
# Dry run prints the fixed experiment plan; performs no network access or training.
if ((DRY_RUN)); then
  for i in "${!METHODS[@]}"; do
    echo "${METHODS[$i]} | ${CONFIGS[$i]} | epochs=$EPOCHS H=40 seed=$SEED reference=false max_steps=${MAX_STEPS:-full}"
  done
  if ((FP16_DIAGNOSTIC)); then
    echo "diagnostic: same-input whole/delta FP16 AllGather shadows; only FP32 applied; timing excluded from performance comparison"
  fi
  if ((VERIFY_DELTA_STATE)); then echo "state smoke: verify FP32 model/optimizer, anchor consensus and EF residual"; fi
  exit 0
fi

mkdir -p "$SOURCE_ROOT/results"
SWEEP_DIR="$(mktemp -d "$SOURCE_ROOT/results/fp16_sweep_seed${SEED}_XXXXXXXX")"
STAGE_DIR="$(mktemp -d /tmp/heterovit_fp16_sweep_XXXXXXXX)"
export OUTPUT_DIR="$SWEEP_DIR/results"
mkdir -p "$STAGE_DIR/cluster" "$OUTPUT_DIR" "$SWEEP_DIR/ranks"
printf '%s\n' "$STAGE_DIR" > "$SWEEP_DIR/source_stage.txt"
printf 'running\n' > "$SWEEP_DIR/status.txt"
if ((FP16_DIAGNOSTIC)); then
  printf 'same_input_fp16_diagnostic\n' > "$SWEEP_DIR/run_kind.txt"
else
  printf 'training_comparison\n' > "$SWEEP_DIR/run_kind.txt"
fi
trap 'rc=$?; if ((rc != 0)); then printf "failed exit=%s\n" "$rc" > "$SWEEP_DIR/status.txt"; fi' EXIT
for folder in src configs tests tools; do
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SOURCE_ROOT/$folder" "$STAGE_DIR/"
done
cp "$SOURCE_ROOT/train_mpi.py" "$STAGE_DIR/train_mpi.py"
cp "$PROJECT_ROOT/mpi_cluster/python_by_host.sh" "$STAGE_DIR/cluster/python_by_host.sh"
cp "$PROJECT_ROOT/mpi_cluster/hostfile" "$STAGE_DIR/cluster/hostfile"
chmod +x "$STAGE_DIR/cluster/python_by_host.sh"
mapfile -t WORKERS < <(awk 'NF && $1 !~ /^#/ {print $1}' "$PROJECT_ROOT/mpi_cluster/slaves.txt")
[[ "${#WORKERS[@]}" -eq 4 ]] || { echo "Expected four workers" >&2; exit 1; }
for worker in "${WORKERS[@]}"; do
  [[ "$worker" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid worker hostname" >&2; exit 1; }
  printf -v QUOTED_STAGE '%q' "$STAGE_DIR"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "icip@$worker" "mkdir -p $QUOTED_STAGE"
  rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "$STAGE_DIR/" "icip@$worker:$STAGE_DIR/"
done
MPI_ARGS=(
  -x PYTHONNOUSERSITE=1 -x OUTPUT_DIR
  --mca btl_tcp_if_include 192.168.1.0/24
  --mca oob_tcp_if_include 192.168.1.0/24
  -np 5 --hostfile "$STAGE_DIR/cluster/hostfile" --map-by ppr:1:node
  -wdir "$STAGE_DIR" "$STAGE_DIR/cluster/python_by_host.sh"
)
echo "Sweep output: $SWEEP_DIR"
# Reference diagnostic/preflight is separate from timed training runs.
"$STAGE_DIR/cluster/python_by_host.sh" "$STAGE_DIR/tools/summarize_fp16_sweep.py" --check-configs "$STAGE_DIR" --methods "${METHODS[@]}"
mpirun "${MPI_ARGS[@]}" tools/preflight_delta_runtime.py configs/mpi/local_sgd_h40_fp16_delta_allgather_40e.yaml > "$SWEEP_DIR/runtime.json" 2> "$SWEEP_DIR/runtime.stderr"
EXTRA_ARGS=()
[[ -z "$MAX_STEPS" ]] || EXTRA_ARGS=(--max-steps "$MAX_STEPS")
for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"
  echo "Starting $method (epochs=$EPOCHS, H=40, seed=$SEED)"
  mpirun "${MPI_ARGS[@]}" "$ENTRYPOINT" \
    --config "configs/mpi/${CONFIGS[$i]}" --name "$method" \
    --epochs "$EPOCHS" --seed "$SEED" "${EXTRA_ARGS[@]}" 2>&1 | tee "$SWEEP_DIR/$method.log"
  # Each node stores its own run directory; collect only reports, not checkpoints.
  mkdir -p "$SWEEP_DIR/ranks/$method/rank0"
  shopt -s nullglob
  runs=("$OUTPUT_DIR/${method}_"*)
  shopt -u nullglob
  [[ "${#runs[@]}" -eq 1 ]] || { echo "Expected one rank0 run directory for $method" >&2; exit 1; }
  rsync -a --include='*/' --include='*.csv' --include='*.json' --include='fp16_ef_state_rank*.npz' --include='config.yaml' --include='run.log' --exclude='*' \
    "${runs[0]}/" "$SWEEP_DIR/ranks/$method/rank0/"
  for j in "${!WORKERS[@]}"; do
    rank=$((j + 1))
    mkdir -p "$SWEEP_DIR/ranks/$method/rank$rank"
    rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
      --include='*/' --include='*.csv' --include='*.json' --include='fp16_ef_state_rank*.npz' --include='config.yaml' --include='run.log' --exclude='*' \
      "icip@${WORKERS[$j]}:$OUTPUT_DIR/${method}_*/" "$SWEEP_DIR/ranks/$method/rank$rank/"
  done
  if ((FP16_DIAGNOSTIC)); then
    cp "$SWEEP_DIR/ranks/$method/rank0/fp16_same_inputs_summary.json" "$SWEEP_DIR/diagnostic_summary.json"
  else
    "$STAGE_DIR/cluster/python_by_host.sh" "$STAGE_DIR/tools/summarize_fp16_sweep.py" "$SWEEP_DIR" --methods "${METHODS[@]}"
  fi
  echo "Completed $method"
done
printf 'completed\n' > "$SWEEP_DIR/status.txt"
if ((FP16_DIAGNOSTIC)); then
  echo "Diagnostic complete: $SWEEP_DIR/diagnostic_summary.json"
else
  echo "Sweep complete: $SWEEP_DIR/summary.csv"
fi
