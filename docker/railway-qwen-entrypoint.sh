#!/bin/sh
set -eu

MODEL_REPO="${MODEL_REPO:-HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf}"
MODEL_DIR="${MODEL_DIR:-/tmp/models}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
N_CTX="${N_CTX:-8192}"
N_PARALLEL="${N_PARALLEL:-1}"
N_THREADS="${N_THREADS:-$(nproc)}"

mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Downloading ${MODEL_REPO}/${MODEL_FILE} to ${MODEL_PATH}"
  URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}?download=true"
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -L --fail --retry 5 --retry-delay 5 -C - \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      -o "${MODEL_PATH}" \
      "${URL}"
  else
    curl -L --fail --retry 5 --retry-delay 5 -C - \
      -o "${MODEL_PATH}" \
      "${URL}"
  fi
fi

echo "Starting llama-server"
exec /usr/local/bin/llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  -c "${N_CTX}" \
  -np "${N_PARALLEL}" \
  -t "${N_THREADS}"
