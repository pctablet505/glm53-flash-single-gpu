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
# live on the host: CPU_OFFLOAD_GB below that OOMs at load. Going ABOVE the
# floor is what we actually want -- see "the measured-best configuration".
#
# Needs Glm5Next support from vLLM PR #53906 -- not in any released vLLM.
#
# Unlike the Qwen3.8-Flash-Next launcher, this needs NO sudo: UVA offload is
# in-process pinned memory, with none of the CUDA-IPC/pidfd_getfd handoff that
# forced ptrace_scope=0 there.
set -uo pipefail

ROOT="/home/pctablet505/Projects/vllm-glm53"
VENV="$ROOT/.venv-glm53"
MODEL="${MODEL:-dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4}"
PORT="${PORT:-8002}"
SERVED_NAME="${SERVED_NAME:-glm53-flash}"

# --- the measured-best configuration (16.76 tok/s, 2026-08-31) ---------------
# These four knobs are a MATCHED SET, not independent preferences. Changing one
# without the others loses most of the gain:
#
#   offload 125 + 68 hot slots   16.76 tok/s   <- default below
#   offload 105, no pinning      13.73 tok/s   (the old default)
#   offload 125, no pinning      11.34 tok/s   (more traffic, nothing to absorb)
#
# Offloading MORE is faster only because the VRAM it frees funds the hot-expert
# cache, and the hottest 36% of a layer's experts serve 75% of its fetches. Take
# the cache away and 125 is the worst of the three. That is why the old header
# advice ("above ~100 just costs speed") no longer holds -- it was written
# before pinning existed.
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-125}"

# Per-layer VRAM cache of the hottest experts, gathered by the Triton kernel in
# offloader/dma_stage.py. The profile is per-layer frequency counts and is tied
# to the offload above: layer N in the profile means "the Nth offloaded layer",
# so a profile captured at a different offload describes different layers.
# Regenerate with VLLM_MOE_SKEW_PERLAYER=<out.json> at the SAME offload.
# 62 (was 68): at 68 slots the run OOMed mid-serving ("Tried to allocate 68.00
# MiB") after load succeeded. GPU_UTIL cannot buy the headroom back because
# KV_BYTES is explicit and weights+cache are fixed, so the only slack is here.
# The 2 dropped experts are the COLDEST of the pinned set: ~0.9 GiB freed for
# a fraction of a point of hit rate, and prefill speed is untouched.
# 54, down from 62. Long-prompt activation (attention + the sparse indexer) is
# allocated per request and scales with sequence length, so the usable context
# ceiling is set by FREE VRAM, not by --max-model-len. Measured:
#   60 slots / 3.4 GiB free -> a 180k prompt dies ("Tried to allocate 116 MiB")
#   54 slots / 6.2 GiB free -> 304,800 tokens prefill fine (205 s, 1487 tok/s)
# Going further down to 48 buys nothing: decode 14.92 vs 14.99 tok/s, inside the
# +-1 tok/s run-to-run spread, because 48->54 is only ~3 points of cache hit rate.
# Shrinking MAX_BATCHED would be the natural way to cut that activation instead,
# but 4096 crashes the engine outright (see the MAX_BATCHED note) -- that lever
# is closed, so headroom has to come from here.
export VLLM_MOE_HOT_SLOTS="${VLLM_MOE_HOT_SLOTS:-54}"
# CUDA-graph mode. vLLM auto-enables "breakable" graphs for this hybrid model;
# under that mode every build on 2026-09-02 died of an Xid 31 MMU fault after
# 2-24 requests. With it OFF a 40-request varied soak (+ a 30k-token prompt)
# ran clean at the same speed (15.8 tok/s). Set to 1 to opt back in.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# Grouped prefill staging: a prefill chunk touches ~all 288 experts, which
# exceeds the 64-row staging buffer, so the layer used to be read zero-copy over
# PCIe (~23 GB/s, re-read per tile): 614 tok/s prefill, ~7 min to first token at
# 262k. With groups the experts are DMA-staged 64 at a time and reduced once:
# outputs equivalent by fresh-prompt logprob comparison (prefill here is not
# run-to-run deterministic, so that is the valid test), 1762 tok/s prefill,
# 10k TTFT 16.8 s -> 5.8 s, 40-request soak + 35k prompt clean. Set 0 to opt out.
export VLLM_MOE_PREFILL_GROUPS="${VLLM_MOE_PREFILL_GROUPS:-1}"
export VLLM_MOE_HOT_PROFILE="${VLLM_MOE_HOT_PROFILE:-$ROOT/hot-profile-125.json}"

# The profile records the offload it was captured at. Mismatch is not fatal --
# the pinner warns and falls back per layer -- but it silently costs ~5 tok/s,
# so say so at launch where it is visible.
if [ -n "$VLLM_MOE_HOT_SLOTS" ] && [ "$VLLM_MOE_HOT_SLOTS" != "0" ]; then
    if [ ! -f "$VLLM_MOE_HOT_PROFILE" ]; then
        echo "WARNING: hot-expert profile '$VLLM_MOE_HOT_PROFILE' not found;" >&2
        echo "         pinning is OFF and throughput drops to ~11-14 tok/s." >&2
    else
        prof_off=$("$VENV/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1])).get('offload_gb',''))" "$VLLM_MOE_HOT_PROFILE" 2>/dev/null)
        if [ -n "$prof_off" ] && [ "$prof_off" != "${CPU_OFFLOAD_GB%.*}" ]; then
            echo "WARNING: profile was captured at CPU_OFFLOAD_GB=${prof_off} but this" >&2
            echo "         launch uses ${CPU_OFFLOAD_GB}. Layer indices will not line up;" >&2
            echo "         regenerate the profile or match the offload." >&2
        fi
    fi
fi

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
# Overhead allowance: measured steady state is offload + ~16 GiB (141 GiB used at
# offload 125 on 2026-09-02); safetensors reads go through reclaimable page cache
# and are not counted by "available". 30 keeps a 14 GiB cushion above that.
need_gib=$(( ${CPU_OFFLOAD_GB%.*} + 30 ))
if [ "$avail_gib" -lt "$need_gib" ]; then
    echo "FATAL: need ~${need_gib} GiB host RAM (${CPU_OFFLOAD_GB} offload + 30 overhead)," >&2
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
echo "serving: $SERVED_NAME on :$PORT, max_len ${MAX_LEN:-327680}"

# --limit-mm-per-prompt zeroes the vision tower: this is wired up as a coding
# backend, and skipping it saves both VRAM and a slice of startup.
#
# SPEC=1 enables the MTP draft head (checkpoint layer 45). It matters more here
# than usual: decode is PCIe-bound on expert fetches, and verifying several
# tokens per forward reuses one fetch, so it converts saved bandwidth directly
# into tokens/s. Off by default only because the first boot should have the
# smallest possible failure surface.
SPEC_ARGS=()
if [ "${SPEC:-1}" = "1" ]; then
    # 2, not 1 or 3: SPEC_TOKENS=1 returns correct text but dies with
    # "CUDA error: an illegal memory access" partway through a longer
    # generation; 2 measured 10.1-11.2 tok/s against an 8.7 baseline.
    # SPEC_METHOD: mtp (default) | ngram | ngram_gpu | suffix | eagle ...; SPEC_EXTRA
    # is raw extra JSON fields, e.g. SPEC_EXTRA=',"prompt_lookup_max":6' for ngram.
    SPEC_ARGS=(--speculative-config "{\"method\":\"${SPEC_METHOD:-mtp}\",\"num_speculative_tokens\":${SPEC_TOKENS:-2}${SPEC_EXTRA:-}}")
fi

# --- optimized-run knobs ------------------------------------------------------
# KV_BYTES: hard cap on KV cache. vLLM otherwise sizes KV from LEFTOVER budget,
#   not need -- the 2026-08-29 run provisioned 2.1x concurrency and spent
#   14.5 GiB at 512k for a single agent. Capping at 1x frees ~7 GiB for resident
#   experts, and every GiB kept on the GPU is a GiB that stops crossing PCIe on
#   EVERY decoded token. Note: when set, this overrides gpu_memory_utilization
#   for KV sizing. ~14 KB/token at long context -> 512k needs ~7.2 GiB.
# AUTOTUNE: 1 (default) leaves FlashInfer autotune at vLLM's -O2 default. The
#   original --no-enable-flashinfer-autotune was inherited boilerplate; measured
#   +3.4% forward-pass rate on the Qwen stack.
if [ "${AUTOTUNE:-1}" = "1" ]; then AUTOTUNE_FLAG=""; else AUTOTUNE_FLAG="--no-enable-flashinfer-autotune"; fi
# BLOCK_SIZE: on SM120 DeepGEMM paged-MQA needs block_kv == 64 for the fp8
#   indexer (only fp4 may use 32). page_size = block_size / index_kpool must
#   therefore be a multiple of 64 -> block_size must be a multiple of 256.
#   Left unset, vLLM picks block_size by LCM and spec-decode depth can shift
#   it off that multiple, which fails at the opaque C++ assert.
if [ -n "${BLOCK_SIZE:-}" ]; then BLOCK_SIZE_FLAG="--block-size ${BLOCK_SIZE}"; else BLOCK_SIZE_FLAG=""; fi
# ENFORCE_EAGER: DMA expert staging is data-dependent (which experts this
#   step picked) and needs a device->host sync to drive the copies, which
#   CUDA graph capture forbids -- it fails with "operation failed due to a
#   previous error during capture". At ~207 ms per forward pass, graphs save
#   single-digit ms while staging is worth ~5x on the transfer, so trading
#   graphs away is heavily net-positive HERE. Do not copy this to a model
#   that fits in VRAM, where the trade reverses.
if [ "${ENFORCE_EAGER:-0}" = "1" ]; then EAGER_FLAG="--enforce-eager"; else EAGER_FLAG=""; fi
# 4 GiB is part of the matched set above: at offload 125 the leftover budget
# would hand KV ~14 GiB, starving the hot-expert cache that pays for the offload.
# KV_BYTES derives from MAX_LEN so the two can never disagree. Measured on this
# checkpoint: ~15.0-15.7 KiB of KV per token (4 GiB gave 272,771 tokens; 5.25 GiB
# gave 375,543). 17200 B/token adds ~10% headroom on the worse figure, and at the
# default context reproduces the 5,637,144,576 that was validated to 304,800
# tokens. This matters because Coldstart passes MAX_LEN but NOT KV_BYTES: moving
# the context slider used to leave KV sized for the old context, and vLLM then
# refuses to start ("N GiB KV cache is needed, larger than the available").
KV_BYTES="${KV_BYTES:-$(( ${MAX_LEN:-327680} * 17200 ))}"
if [ -n "$KV_BYTES" ]; then KV_CACHE_FLAG="--kv-cache-memory-bytes ${KV_BYTES}"; else KV_CACHE_FLAG=""; fi
# -----------------------------------------------------------------------------

# EXTRA_ARGS: unquoted passthrough for one-off flags (word-split on
# purpose), e.g. EXTRA_ARGS='--profiler-config.profiler=torch
# --profiler-config.torch_profiler_dir=/tmp/p'. NOTE: never put a
# comment INSIDE the backslash-continued invocation below -- it ends
# the continuation and every later flag is silently dropped.
"$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${MAX_LEN:-327680}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
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
    ${EXTRA_ARGS:-} \
    --load-format "${LOAD_FORMAT:-auto}" \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --enable-prefix-caching \
    ${EAGER_FLAG} \
    ${KV_CACHE_FLAG} \
    ${BLOCK_SIZE_FLAG} \
    ${AUTOTUNE_FLAG} \
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 &

VLLM_PID=$!
wait "$VLLM_PID"
