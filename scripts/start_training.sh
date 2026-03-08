#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

MODE="all"
PROFILE="full"
PREPARE_DATA=0
SKIP_INSTALL=0
STAGE1_WEIGHTS=""
STAGE1_EPOCHS=""
STAGE1_IMGSZ=""
STAGE1_BATCH=""
STAGE1_DEVICE=""
STAGE1_WORKERS=""
STAGE1_FRACTION=""
STAGE1_SEED=""
STAGE2_EPOCHS=""
STAGE2_BATCH=""
STAGE2_IMGSZ=""
STAGE2_DEVICE=""
STAGE2_WORKERS=""
STAGE2_FRACTION=""
STAGE2_SEED=""

usage() {
  cat <<'EOF'
Usage: ./scripts/start_training.sh [options]

Runs training using the project .venv. Creates .venv if missing.

Options:
  --profile <fast|full>       Preset profile (default: full)
  --fast                      Shortcut for --profile fast
  --full                      Shortcut for --profile full
  --mode <stage1|stage2|all>  Training mode (default: all)
  --prepare-data              Run scripts/prepare_data_v2.py before training
  --skip-install              Skip pip install -r requirements.txt
  --stage1-weights <value>    Stage-1 init weights (profile default)
  --stage1-epochs <value>     Stage-1 epochs (profile default)
  --stage1-imgsz <value>      Stage-1 image size (profile default)
  --stage1-batch <value>      Stage-1 batch size (profile default)
  --stage1-device <value>     Stage-1 device: auto|mps|cpu|cuda (profile default)
  --stage1-workers <value>    Stage-1 dataloader workers (profile default)
  --stage1-fraction <value>   Stage-1 train split fraction in (0,1] (default: 1.0)
  --stage1-seed <value>       Stage-1 fraction sampling seed (default: 42)
  --stage2-epochs <value>     Stage-2 epochs (profile default)
  --stage2-batch <value>      Stage-2 batch size (profile default)
  --stage2-imgsz <value>      Stage-2 image size (profile default)
  --stage2-device <value>     Stage-2 device: auto|mps|cpu|cuda (profile default)
  --stage2-workers <value>    Stage-2 dataloader workers (profile default)
  --stage2-fraction <value>   Stage-2 train split fraction in (0,1] (default: 1.0)
  --stage2-seed <value>       Stage-2 fraction sampling seed (default: 42)
  -h, --help                  Show this help
EOF
}

apply_profile_defaults() {
  case "${PROFILE}" in
    full)
      STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-model/detector/yolov8n.pt}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
      STAGE1_IMGSZ="${STAGE1_IMGSZ:-640}"
      STAGE1_BATCH="${STAGE1_BATCH:-16}"
      STAGE1_DEVICE="${STAGE1_DEVICE:-auto}"
      STAGE1_WORKERS="${STAGE1_WORKERS:-4}"
      STAGE1_FRACTION="${STAGE1_FRACTION:-1.0}"
      STAGE1_SEED="${STAGE1_SEED:-42}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-8}"
      STAGE2_BATCH="${STAGE2_BATCH:-32}"
      STAGE2_IMGSZ="${STAGE2_IMGSZ:-224}"
      STAGE2_DEVICE="${STAGE2_DEVICE:-auto}"
      STAGE2_WORKERS="${STAGE2_WORKERS:-4}"
      STAGE2_FRACTION="${STAGE2_FRACTION:-1.0}"
      STAGE2_SEED="${STAGE2_SEED:-42}"
      ;;
    fast)
      STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-model/detector/yolov8n.pt}"
      STAGE1_EPOCHS="${STAGE1_EPOCHS:-15}"
      STAGE1_IMGSZ="${STAGE1_IMGSZ:-512}"
      STAGE1_BATCH="${STAGE1_BATCH:-16}"
      STAGE1_DEVICE="${STAGE1_DEVICE:-auto}"
      STAGE1_WORKERS="${STAGE1_WORKERS:-4}"
      STAGE1_FRACTION="${STAGE1_FRACTION:-1.0}"
      STAGE1_SEED="${STAGE1_SEED:-42}"
      STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
      STAGE2_BATCH="${STAGE2_BATCH:-32}"
      STAGE2_IMGSZ="${STAGE2_IMGSZ:-224}"
      STAGE2_DEVICE="${STAGE2_DEVICE:-auto}"
      STAGE2_WORKERS="${STAGE2_WORKERS:-4}"
      STAGE2_FRACTION="${STAGE2_FRACTION:-1.0}"
      STAGE2_SEED="${STAGE2_SEED:-42}"
      ;;
    *)
      echo "Error: invalid --profile '${PROFILE}'. Use fast or full." >&2
      exit 1
      ;;
  esac
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

ensure_venv() {
  require_cmd python3
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "[setup] Creating .venv..."
    python3 -m venv "${VENV_DIR}"
  fi

  if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
    echo "[setup] Installing training dependencies..."
    "${VENV_PIP}" install -r "${ROOT_DIR}/requirements.txt"
  else
    echo "[setup] Skipping dependency install."
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode)
        MODE="$2"
        shift 2
        ;;
      --profile)
        PROFILE="$2"
        shift 2
        ;;
      --fast)
        PROFILE="fast"
        shift
        ;;
      --full)
        PROFILE="full"
        shift
        ;;
      --prepare-data)
        PREPARE_DATA=1
        shift
        ;;
      --skip-install)
        SKIP_INSTALL=1
        shift
        ;;
      --stage1-weights)
        STAGE1_WEIGHTS="$2"
        shift 2
        ;;
      --stage1-epochs)
        STAGE1_EPOCHS="$2"
        shift 2
        ;;
      --stage1-imgsz)
        STAGE1_IMGSZ="$2"
        shift 2
        ;;
      --stage1-batch)
        STAGE1_BATCH="$2"
        shift 2
        ;;
      --stage1-device)
        STAGE1_DEVICE="$2"
        shift 2
        ;;
      --stage1-workers)
        STAGE1_WORKERS="$2"
        shift 2
        ;;
      --stage1-fraction)
        STAGE1_FRACTION="$2"
        shift 2
        ;;
      --stage1-seed)
        STAGE1_SEED="$2"
        shift 2
        ;;
      --stage2-epochs)
        STAGE2_EPOCHS="$2"
        shift 2
        ;;
      --stage2-batch)
        STAGE2_BATCH="$2"
        shift 2
        ;;
      --stage2-imgsz)
        STAGE2_IMGSZ="$2"
        shift 2
        ;;
      --stage2-device)
        STAGE2_DEVICE="$2"
        shift 2
        ;;
      --stage2-workers)
        STAGE2_WORKERS="$2"
        shift 2
        ;;
      --stage2-fraction)
        STAGE2_FRACTION="$2"
        shift 2
        ;;
      --stage2-seed)
        STAGE2_SEED="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Error: unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  case "${MODE}" in
    stage1|stage2|all)
      ;;
    *)
      echo "Error: invalid --mode '${MODE}'. Use stage1, stage2, or all." >&2
      exit 1
      ;;
  esac

  apply_profile_defaults

  if [[ "${STAGE1_DEVICE}" == "auto" ]]; then
    local_device="auto"
    if [[ -x "${VENV_PYTHON}" ]]; then
      local_device="$(${VENV_PYTHON} -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo auto)"
    fi
    STAGE1_DEVICE="${local_device}"
  fi

  if [[ "${STAGE2_DEVICE}" == "auto" ]]; then
    local_device="auto"
    if [[ -x "${VENV_PYTHON}" ]]; then
      local_device="$(${VENV_PYTHON} -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo auto)"
    fi
    STAGE2_DEVICE="${local_device}"
  fi

  echo "[config] profile=${PROFILE} mode=${MODE}"
  echo "[config] stage1: weights=${STAGE1_WEIGHTS} epochs=${STAGE1_EPOCHS} imgsz=${STAGE1_IMGSZ} batch=${STAGE1_BATCH} device=${STAGE1_DEVICE} workers=${STAGE1_WORKERS} fraction=${STAGE1_FRACTION} seed=${STAGE1_SEED}"
  echo "[config] stage2: epochs=${STAGE2_EPOCHS} imgsz=${STAGE2_IMGSZ} batch=${STAGE2_BATCH} device=${STAGE2_DEVICE} workers=${STAGE2_WORKERS} fraction=${STAGE2_FRACTION} seed=${STAGE2_SEED}"
}

run_stage1() {
  echo "[train] Running stage-1 detector training..."
  "${VENV_PYTHON}" "${ROOT_DIR}/train/train_stage1_detector_yolo.py" \
    --weights "${STAGE1_WEIGHTS}" \
    --epochs "${STAGE1_EPOCHS}" \
    --imgsz "${STAGE1_IMGSZ}" \
    --batch "${STAGE1_BATCH}" \
    --device "${STAGE1_DEVICE}" \
    --workers "${STAGE1_WORKERS}" \
    --train-fraction "${STAGE1_FRACTION}" \
    --seed "${STAGE1_SEED}"
}

run_stage2() {
  echo "[train] Running stage-2 freshness training..."
  STAGE2_EPOCHS="${STAGE2_EPOCHS}" \
  STAGE2_BATCH_SIZE="${STAGE2_BATCH}" \
  STAGE2_IMG_SIZE="${STAGE2_IMGSZ}" \
  STAGE2_DEVICE="${STAGE2_DEVICE}" \
  STAGE2_WORKERS="${STAGE2_WORKERS}" \
  STAGE2_TRAIN_FRACTION="${STAGE2_FRACTION}" \
  STAGE2_SEED="${STAGE2_SEED}" \
  "${VENV_PYTHON}" "${ROOT_DIR}/train/train_stage2_freshness.py"
}

parse_args "$@"
ensure_venv

if [[ "${PREPARE_DATA}" -eq 1 ]]; then
  echo "[data] Preparing data_v2..."
  "${VENV_PYTHON}" "${ROOT_DIR}/scripts/prepare_data_v2.py"
fi

if [[ "${MODE}" == "stage1" || "${MODE}" == "all" ]]; then
  run_stage1
fi

if [[ "${MODE}" == "stage2" || "${MODE}" == "all" ]]; then
  run_stage2
fi

echo "[done] Training script finished."
