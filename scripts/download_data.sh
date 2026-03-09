#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/data/external"

# Map format: <kaggle_dataset_slug>::<target_subdir_under_data_external>
DEFAULT_MAP=(
  "raghavrpotdar/fresh-and-stale::raghavrpotdar_fresh_and_stale"
  "muhriddinmuxiddinov/fruits-and-vegetables::muhriddinmuxiddinov_fruits_and_vegetables"
  "ulnnproject/food-freshness::ulnnproject_food_freshness"
  "filipemonteir/fresh-and-rotten::filipemonteir_fresh_and_rotten"
)

DATASET_MAP=()
SKIP_EXISTING=1

usage() {
  cat <<'EOF'
Usage: ./scripts/download_data.sh [options]

Downloads external source datasets from Kaggle into data/external/
for scripts/prepare_data_v2.py.

Options:
  --dataset <slug::target_subdir>
      Add/override dataset mapping. Example:
      --dataset "owner/dataset::my_target_folder"
      Can be passed multiple times.

  --no-skip-existing
      Re-download even if target folder already has files.

  -h, --help
      Show this help.

Examples:
  ./scripts/download_data.sh
  ./scripts/download_data.sh --dataset "owner/dataset::owner_dataset"
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

check_kaggle_auth() {
  local cfg_dir="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
  local cfg_file="${cfg_dir}/kaggle.json"
  if [[ ! -f "${cfg_file}" ]]; then
    echo "Error: Kaggle credentials not found at ${cfg_file}" >&2
    echo "Create an API token in Kaggle account settings and place kaggle.json there." >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        DATASET_MAP+=("$2")
        shift 2
        ;;
      --no-skip-existing)
        SKIP_EXISTING=0
        shift
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

  if [[ ${#DATASET_MAP[@]} -eq 0 ]]; then
    DATASET_MAP=("${DEFAULT_MAP[@]}")
  fi
}

parse_args "$@"
require_cmd kaggle
check_kaggle_auth

mkdir -p "${EXTERNAL_DIR}"

echo "[info] Root: ${ROOT_DIR}"
echo "[info] External data dir: ${EXTERNAL_DIR}"

ok=0
failed=0
skipped=0

for mapping in "${DATASET_MAP[@]}"; do
  slug="${mapping%%::*}"
  target_subdir="${mapping#*::}"

  if [[ -z "${slug}" || -z "${target_subdir}" || "${slug}" == "${target_subdir}" ]]; then
    echo "[error] Invalid mapping: ${mapping}" >&2
    ((failed+=1))
    continue
  fi

  target_dir="${EXTERNAL_DIR}/${target_subdir}"
  mkdir -p "${target_dir}"

  if [[ "${SKIP_EXISTING}" -eq 1 ]] && [[ -n "$(ls -A "${target_dir}" 2>/dev/null || true)" ]]; then
    echo "[skip] ${slug} -> ${target_dir} (already has files)"
    ((skipped+=1))
    continue
  fi

  echo "[download] ${slug} -> ${target_dir}"
  if kaggle datasets download -d "${slug}" -p "${target_dir}" --unzip; then
    ((ok+=1))
  else
    echo "[error] Failed to download: ${slug}" >&2
    ((failed+=1))
  fi

done

echo
echo "[summary] downloaded=${ok} skipped=${skipped} failed=${failed}"
echo "[next] Build merged dataset: ./.venv/bin/python scripts/prepare_data_v2.py"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
