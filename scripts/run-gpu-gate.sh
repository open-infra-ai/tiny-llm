#!/usr/bin/env bash
# Real-GPU correctness gate: every test must run (no skips) and pass.
#
# Requires both real-model env vars so the GGUF-gated tests execute:
#   TLLM_GGUF_TEST_MODEL    Qwen2.5-0.5B-Instruct GGUF (GQA 14->2)
#   TLLM_GGUF_TEST_MODEL_2  a second GGUF with a distinct GQA/MQA config
#
# Usage: scripts/run-gpu-gate.sh
# Optional: BUILD_DIR, CTEST_TIMEOUT
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
ctest_timeout="${CTEST_TIMEOUT:-300}"
model="${TLLM_GGUF_TEST_MODEL:-}"
model2="${TLLM_GGUF_TEST_MODEL_2:-}"

die() {
    echo "error: $*" >&2
    exit 1
}

require_model() {
    local var="$1" path="$2"
    [[ -n "${path}" ]] || die "${var} is not set; this gate refuses to run with model-gated tests skipping"
    [[ -f "${path}" ]] || die "${var} points to a missing file: ${path}"
}

require_model TLLM_GGUF_TEST_MODEL "${model}"
require_model TLLM_GGUF_TEST_MODEL_2 "${model2}"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; this gate requires a CUDA device"
nvidia-smi -L >/dev/null 2>&1 || die "no CUDA device visible to nvidia-smi"
command -v nvcc >/dev/null 2>&1 || die "nvcc not found"

log_dir="${repo_root}/docs/performance/results/data"
log_file="${log_dir}/$(date -u +%F)-gpu-gate.log"
mkdir -p "${log_dir}"

tracked_dirty="$(git -C "${repo_root}" status --porcelain --untracked-files=no | wc -l)"

{
    echo "# tiny-llm GPU correctness gate"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "tracked_dirty_paths: ${tracked_dirty}"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | head -1)"
    echo "nvcc: $(nvcc --version | tail -1)"
    for pair in "model:${model}" "model2:${model2}"; do
        label="${pair%%:*}"
        path="${pair#*:}"
        echo "${label}: $(basename "${path}") ($(stat -c%s "${path}") bytes, sha256 $(sha256sum "${path}" | cut -d' ' -f1))"
    done
    echo
} | tee "${log_file}"

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON 2>&1 |
    tee -a "${log_file}"
cmake --build "${build_dir}" -j"$(nproc)" 2>&1 | tee -a "${log_file}"

set +e
ctest --test-dir "${build_dir}" --output-on-failure --timeout "${ctest_timeout}" 2>&1 |
    tee -a "${log_file}"
ctest_status="${PIPESTATUS[0]}"
set -e

total="$(grep -oE 'tests failed out of [0-9]+' "${log_file}" | tail -1 | grep -oE '[0-9]+' || true)"
skipped="$(awk '/The following tests did not run:/{flag=1; next} flag && /\(Skipped\)/{n++} END{print n+0}' "${log_file}")"

if [[ "${ctest_status}" -ne 0 ]]; then
    echo "gate FAILED: ctest exited ${ctest_status}; see ${log_file}" >&2
    exit "${ctest_status}"
fi
if [[ "${skipped}" -ne 0 ]]; then
    echo "gate FAILED: ${skipped} test(s) skipped; see ${log_file}" >&2
    awk '/The following tests did not run:/{flag=1} flag' "${log_file}" >&2
    exit 1
fi

sed -i "s|${repo_root}|<repo>|g; s|${HOME}|~|g" "${log_file}"

echo "gate OK: ${total:-unknown} tests executed, 0 skipped, log ${log_file}"
