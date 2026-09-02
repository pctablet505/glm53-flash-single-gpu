#!/usr/bin/env bash
# Serve GLM-5.3-Flash (NVFP4) on a single RTX PRO 6000 (96 GB).
#
# HOW THIS FITS ON ONE CARD
# The checkpoint is 181.3 GiB, but 163.3 GiB of that (90%) is routed MoE expert
# weights. --cpu-offload-params experts keeps those in pinned host RAM and lets
# the GPU read them over PCIe via UVA; everything else -- attention, embeddings,
# the shared expert, norms -- is only 18.0 GiB and stays resident.
#
# Because it is a top-8-of-288 MoE, a decode step touches only 8/288 of each
# layer's expert bytes, so zero-copy UVA reads what is actually routed. That is
# why the *uva* backend is right here and the *prefetch* backend is not: prefetch
# moves whole layers ahead of time (~3.9 GiB/layer) regardless of routing.
#
# WHY THE OFFLOAD IS THIS LARGE
# It is not a preference. 95.6 GiB of VRAM minus 18.0 GiB of resident weights
# minus KV/activations leaves room for ~60-65 GiB of experts, so ~100 GiB must
# live on the host. CPU_OFFLOAD_GB below that will OOM at load; above it just
# costs speed. Tune downward, and expect the floor near 100.
#
# Needs Glm5Next support from vLLM PR #53906 -- not in any released vLLM.
#
# Unlike the Qwen3.8-Flash-Next launcher, this needs NO sudo: UVA offload is
# in-process pinned memory, with none of the CUDA-IPC/pidfd_getfd handoff that
# forced ptrace_scope=0 there.
set -uo pipefail

ROOT="/home/pctablet505/Projects/vllm-glm53"
VENV="$ROOT/.venv-glm53"
MODEL="${MODEL:-$HOME/Models/GLM-5.3-Flash-ABLITERATED-NVFP4}"
PORT="${PORT:-8002}"
SERVED_NAME="${SERVED_NAME:-glm53-flash}"

# Expert weights to keep in host RAM. See "WHY THE OFFLOAD IS THIS LARGE".
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-112}"

# MANDATORY on SM120. Both CUTLASS-family NVFP4 MoE kernels (flashinfer_cutlass
# -- which is what "auto" picks -- and vLLM's own cutlass) accept the work and
# return GARBAGE: the model loads, serves, and emits one token forever with no
# error or warning. flashinfer_trtllm and flashinfer_cutedsl refuse outright on
# this card; marlin is the only backend that is both accepted and correct.
# Verified 2026-08-29 by sweeping all five. See FINDINGS.md bug 7.
MOE_BACKEND="${MOE_BACKEND:-marlin}"

# --- cwd shadowing guard (required) ------------------------------------------
# $ROOT is ITSELF a second, uncompiled vLLM checkout: $ROOT/vllm/ exists next to
# the real $ROOT/src/vllm/. Python puts cwd first on sys.path, so launching from
# $ROOT imports the copy with no compiled extensions, and the failure names the
# wrong subsystem entirely:
#   "ImportError: vllm.vllm_flash_attn requires the CUDA flash attention
#    extensions (_vllm_fa2_C or _vllm_fa3_C)"
# which reads as a broken build. The build is fine; the cwd is not. Run from a
# directory containing no vllm/ so the editable install wins.
RUNDIR="$ROOT/run"
mkdir -p "$RUNDIR"
cd "$RUNDIR" || exit 1

resolved=$("$VENV/bin/python" -c "import vllm; print(vllm.__file__)" 2>/dev/null)
case "$resolved" in
    "$ROOT/src/vllm/"*) ;;
    *) echo "FATAL: vllm resolves to $resolved, expected $ROOT/src/vllm/..." >&2
       echo "       Something is shadowing the editable install." >&2
       exit 1 ;;
esac
echo "vllm:    $resolved"
# -----------------------------------------------------------------------------

export CUDA_HOME="$VENV/lib/python3.13/site-packages/nvidia/cu13"

# --- lib64 shim (required) ---------------------------------------------------
# FlashInfer's JIT hardcodes "-L$cuda_home/lib64" (flashinfer/jit/cpp_ext.py),
# but the pip CUDA wheels lay libraries out in lib/, not lib64/. Without this
# the fused-MoE kernel fails to LINK at first forward pass with
# "ld: cannot find -lcudart / -lnvrtc", surfacing as "Ninja build failed".
[ -e "$CUDA_HOME/lib64" ] || ln -sfn lib "$CUDA_HOME/lib64"
# -----------------------------------------------------------------------------
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
# pinned_use_cuda_host_register: allocate pinned host memory by malloc +
# cudaHostRegister rather than cudaHostAlloc. cudaHostAlloc hands out
# SHARED 4 GiB granules, so the ~2.4 GiB w13 and ~1.2 GiB w2 expert
# tensors each rounded up to a full 4 GiB block -- 33.8 GiB of RssShmem
# for 19.0 GiB of actual weights (1.85x). At full scale that turns a
# 100 GiB offload into ~185 GiB of host RSS, which is what kept killing
# the box. Registering exact-size allocations removes the rounding.
export PYTORCH_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
export TORCH_CUDA_ARCH_LIST=12.0f

# Do NOT pin the offloaded weights in Python. Counter-intuitive but measured:
# when the offloader hands get_cuda_view_from_cpu_tensor an ALREADY-pinned
# tensor, that tensor came from PyTorch's caching host allocator, which rounds
# to power-of-two blocks -- the 2.42 GiB w13 expert tensors took 4 GiB each and
# the 1.21 GiB w2 took 2 GiB, so 19.0 GiB of weights occupied 33.8 GiB of RAM.
# Handing it an UNPINNED tensor makes the C++ op pin the memory itself with an
# exact-size cudaHostAlloc (csrc/libtorch_stable/cuda_view.cu), which brought
# the same 19.0 GiB down to 21.5 GiB resident -- 1.94x -> 1.13x.
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1

# DeepGEMM's paged-MQA attention asserts arch_major == 10 (SM100); this card is
# SM120. Only the speculative-decode path reaches it, but disabling it costs
# nothing measurable on the plain path and keeps SPEC=1 usable.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# --- host RAM guard ----------------------------------------------------------
# There is no swap on this box, so overshooting host RAM is an OOM kill, not a
# slowdown -- one already happened on 2026-08-28. Pinned pages cannot be
# reclaimed, so refuse to start rather than take the machine down.
avail_gib=$(awk '/MemAvailable/{m=$2} /SwapFree/{s=$2} END{print int((m+s)/1048576)}' /proc/meminfo)
# +45, not +15. The 2026-08-29 attempt offloaded 110 GiB and peaked at
# ~168 GiB of host use -- the loader's transient copies and page cache cost
# far more than the pinned weights alone. Calibrate to the measurement.
need_gib=$(( ${CPU_OFFLOAD_GB%.*} + 45 ))
if [ "$avail_gib" -lt "$need_gib" ]; then
    echo "FATAL: need ~${need_gib} GiB host RAM (${CPU_OFFLOAD_GB} offload + 15 overhead)," >&2
    echo "       but only ${avail_gib} GiB is available. Free memory or lower CPU_OFFLOAD_GB." >&2
    exit 1
fi
echo "host RAM: ${avail_gib} GiB available, need ~${need_gib} GiB -- ok"
# -----------------------------------------------------------------------------

# --- duplicate-launch guard ---------------------------------------------------
# A vLLM that OOMs on the GPU HANGS instead of exiting -- it keeps its VRAM and
# its process tree. Launching a second one then stalls forever behind it with an
# empty log, which looks like a slow start rather than a collision (2026-08-29).
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "FATAL: port $PORT is already bound. Stop the running server first:" >&2
    echo "  pkill -f 'vllm-glm53.*vllm serve'   # or kill the pid holding it" >&2
    exit 1
fi
others=$(pgrep -f "vllm-glm53/.venv-glm53/bin/vllm serve" | grep -v "^$$\$" | head -3)
if [ -n "$others" ]; then
    echo "FATAL: a vllm from this tree is already running (pids: $(echo $others | tr '\n' ' '))." >&2
    echo "       It may be a hung GPU-OOM. Kill it before relaunching." >&2
    exit 1
fi
# -----------------------------------------------------------------------------

cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "stopping vllm (pid $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$VLLM_PID" 2>/dev/null && kill -KILL "$VLLM_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

echo "model:   $MODEL"
echo "offload: ${CPU_OFFLOAD_GB} GiB of experts -> host RAM (uva)"
echo "serving: $SERVED_NAME on :$PORT, max_len ${MAX_LEN:-262144}"

# --limit-mm-per-prompt zeroes the vision tower: this is wired up as a coding
# backend, and skipping it saves both VRAM and a slice of startup.
#
# SPEC=1 enables the MTP draft head (checkpoint layer 45). It matters more here
# than usual: decode is PCIe-bound on expert fetches, and verifying several
# tokens per forward reuses one fetch, so it converts saved bandwidth directly
# into tokens/s. Off by default only because the first boot should have the
# smallest possible failure surface.
SPEC_ARGS=()
if [ "${SPEC:-0}" = "1" ]; then
    # 2, not 1 or 3: SPEC_TOKENS=1 returns correct text but dies with
    # "CUDA error: an illegal memory access" partway through a longer
    # generation; 2 measured 10.1-11.2 tok/s against an 8.7 baseline.
    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-2}}")
fi

"$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${MAX_LEN:-524288}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.93}" \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --offload-backend uva \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --cpu-offload-params experts \
    --max-num-seqs "${MAX_SEQS:-1}" \
    --max-num-batched-tokens "${MAX_BATCHED:-8192}" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    ${ATTN_BACKEND:+--attention-backend "$ATTN_BACKEND"} \
    ${MOE_BACKEND:+--kernel-config "{\"moe_backend\":\"$MOE_BACKEND\"}"} \
    --load-format "${LOAD_FORMAT:-auto}" \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 &

VLLM_PID=$!
wait "$VLLM_PID"
