#!/usr/bin/env bash
# Run the five H=40 methods sequentially from an isolated source snapshot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_ROOT="$PROJECT_ROOT/HeteroViT-MPI"
EPOCHS=40
MAX_STEPS=""
DRY_RUN=0
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --epochs|--max-steps)
      option="$1"
      if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$option requires a positive integer" >&2; exit 2
      fi
      if [[ "$option" == --epochs ]]; then EPOCHS="$2"; else MAX_STEPS="$2"; fi
      shift 2 ;;
    -h|--help)
      echo "Usage: bash $0 [--dry-run] [--epochs N] [--max-steps N]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
METHODS=(fp32 fp16_whole_allgather fp16_whole_allreduce fp16_delta_allgather fp16_delta_allreduce)
CONFIGS=(
  local_sgd_h40_fp32_40e.yaml
  local_sgd_h40_fp16_allgather_40e.yaml
  local_sgd_h40_fp16_40e.yaml
  local_sgd_h40_fp16_delta_allgather_40e.yaml
  local_sgd_h40_fp16_delta_allreduce_40e.yaml
)
for config in "${CONFIGS[@]}"; do
  [[ -f "$SOURCE_ROOT/configs/mpi/$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
# Dry run prints the fixed experiment plan; performs no network access or training.
if ((DRY_RUN)); then
  for i in "${!METHODS[@]}"; do
    echo "${METHODS[$i]} | ${CONFIGS[$i]} | epochs=$EPOCHS H=40 seed=42 reference=false max_steps=${MAX_STEPS:-full}"
  done
  exit 0
fi

mkdir -p "$SOURCE_ROOT/results"
SWEEP_DIR="$(mktemp -d "$SOURCE_ROOT/results/fp16_sweep_XXXXXXXX")"
STAGE_DIR="$(mktemp -d /tmp/heterovit_fp16_sweep_XXXXXXXX)"
export OUTPUT_DIR="$SWEEP_DIR/results"
mkdir -p "$STAGE_DIR/cluster" "$OUTPUT_DIR" "$SWEEP_DIR/ranks"
printf '%s\n' "$STAGE_DIR" > "$SWEEP_DIR/source_stage.txt"
printf 'running\n' > "$SWEEP_DIR/status.txt"
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
"$STAGE_DIR/cluster/python_by_host.sh" "$STAGE_DIR/tools/summarize_fp16_sweep.py" --check-configs "$STAGE_DIR"
mpirun "${MPI_ARGS[@]}" tools/preflight_delta_runtime.py configs/mpi/local_sgd_h40_fp16_delta_allgather_40e.yaml > "$SWEEP_DIR/runtime.json" 2> "$SWEEP_DIR/runtime.stderr"
EXTRA_ARGS=()
[[ -z "$MAX_STEPS" ]] || EXTRA_ARGS=(--max-steps "$MAX_STEPS")
for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"
  echo "Starting $method (epochs=$EPOCHS, H=40, seed=42)"
  mpirun "${MPI_ARGS[@]}" train_mpi.py \
    --config "configs/mpi/${CONFIGS[$i]}" --name "$method" \
    --epochs "$EPOCHS" --seed 42 "${EXTRA_ARGS[@]}" 2>&1 | tee "$SWEEP_DIR/$method.log"
  # Each node stores its own run directory; collect only reports, not checkpoints.
  mkdir -p "$SWEEP_DIR/ranks/$method/rank0"
  shopt -s nullglob
  runs=("$OUTPUT_DIR/${method}_"*)
  shopt -u nullglob
  [[ "${#runs[@]}" -eq 1 ]] || { echo "Expected one rank0 run directory for $method" >&2; exit 1; }
  rsync -a --include='*/' --include='*.csv' --include='*.json' --include='config.yaml' --include='run.log' --exclude='*' \
    "${runs[0]}/" "$SWEEP_DIR/ranks/$method/rank0/"
  for j in "${!WORKERS[@]}"; do
    rank=$((j + 1))
    mkdir -p "$SWEEP_DIR/ranks/$method/rank$rank"
    rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
      --include='*/' --include='*.csv' --include='*.json' --include='config.yaml' --include='run.log' --exclude='*' \
      "icip@${WORKERS[$j]}:$OUTPUT_DIR/${method}_*/" "$SWEEP_DIR/ranks/$method/rank$rank/"
  done
  "$STAGE_DIR/cluster/python_by_host.sh" "$STAGE_DIR/tools/summarize_fp16_sweep.py" "$SWEEP_DIR"
  echo "Completed $method"
done
printf 'completed\n' > "$SWEEP_DIR/status.txt"
echo "Sweep complete: $SWEEP_DIR/summary.csv"
