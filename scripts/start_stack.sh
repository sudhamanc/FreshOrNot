#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
SKIP_PYTHON_INSTALL=0
SKIP_NODE_INSTALL=0

usage() {
  cat <<'EOF'
Usage: ./scripts/start_stack.sh [options]

Starts backend (FastAPI) and frontend (Vite) for local development.
Netlify/Render hosting remains separate in production.

Options:
  --backend-port <port>       Backend port (default: 8000)
  --frontend-port <port>      Frontend port (default: 5173)
  --skip-python-install       Skip pip install for backend requirements
  --skip-node-install         Skip npm install in frontend
  -h, --help                  Show this help
EOF
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

  if [[ "${SKIP_PYTHON_INSTALL}" -eq 0 ]]; then
    echo "[setup] Installing backend Python dependencies..."
    "${VENV_PIP}" install -r "${ROOT_DIR}/backend/requirements.txt"
  else
    echo "[setup] Skipping backend Python dependency install."
  fi
}

ensure_frontend_deps() {
  require_cmd npm
  if [[ "${SKIP_NODE_INSTALL}" -eq 0 ]]; then
    echo "[setup] Installing frontend Node dependencies..."
    (cd "${ROOT_DIR}/frontend" && npm install)
  else
    echo "[setup] Skipping frontend npm install."
  fi
}

load_backend_env() {
  local env_file="${ROOT_DIR}/backend/.env"
  if [[ -f "${env_file}" ]]; then
    echo "[setup] Loading backend env vars from backend/.env"
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend-port)
        BACKEND_PORT="$2"
        shift 2
        ;;
      --frontend-port)
        FRONTEND_PORT="$2"
        shift 2
        ;;
      --skip-python-install)
        SKIP_PYTHON_INSTALL=1
        shift
        ;;
      --skip-node-install)
        SKIP_NODE_INSTALL=1
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
}

cleanup() {
  echo
  echo "[shutdown] Stopping frontend/backend..."
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}

parse_args "$@"

require_cmd bash
ensure_venv
ensure_frontend_deps
load_backend_env

trap cleanup EXIT INT TERM

export VITE_API_BASE_URL="http://localhost:${BACKEND_PORT}/api"

echo "[start] Backend: http://localhost:${BACKEND_PORT}"
"${VENV_PYTHON}" -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "[start] Frontend: http://localhost:${FRONTEND_PORT}"
(
  cd "${ROOT_DIR}/frontend"
  npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
