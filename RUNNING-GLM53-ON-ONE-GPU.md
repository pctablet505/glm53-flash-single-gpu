# Running GLM-5.3-Flash (NVFP4) on a single RTX PRO 6000 / SM120

A complete reproduction guide for getting the model to run **correctly**.
GLM-5.3-Flash is 320B total / 18B active; the NVFP4 checkpoint is **181.3 GiB**
and the card has ~95 GiB. It runs, but eight separate bugs stand between a fresh
checkout and correct output — and **one of them produces no error at all**, so
you can spend a day chasing it.

Everything below was measured on:
Ubuntu 26.04 (glibc 2.43) · RTX PRO 6000 Blackwell Workstation (SM120, 96 GiB)
· AMD Ryzen 9 9950X3D · 182 GiB RAM (dual-channel) · driver 595.84 · vLLM
PR #53906 @ `878631b60` · FlashInfer 0.6.18

> **Revision note (2026-09-02).** This document was written on 2026-08-29 and
> described the end state as *"10–12 tok/s at 512k context"*. The correctness
> content is unchanged and still accurate — but the performance claims have been
> superseded. The shipped configuration decodes at **~16 tok/s** (16.50 peak,
> ~15 on the current long-context settings) and now serves **327,680 context**,
> with a 304,800-token prompt actually prefilled; see [README](README.md) for
> the result and [OPTIMIZING](OPTIMIZING-GLM53-ON-ONE-GPU.md) for how.
> Superseded passages below are marked; the launch command in §6 is kept because
> it is the *minimal correct* configuration, which is a different and
> still-useful thing from the *fast* one.

Companion files: `GLM53-SM120.patch` (all source changes), `FINDINGS.md`
(condensed), `serve.sh` (the original launcher). The optimized launcher is
`serve-opt.sh` in the vLLM tree.

---

## 0. Read this first

Two facts decide whether this is worth your time:

1. > **SUPERSEDED, in part.** The original text read: *"Decode is PCIe-bound and
   > cannot be tuned out. A decoded token touches 4.43 GiB of expert weights.
   > With ~65% of experts in host RAM, ~2.9 GB crosses PCIe per token. That is
   > the 10-12 tok/s ceiling. Kernel choice moves it 6%; MTP moves it 31%;
   > config tuning ~10%. **Nothing moves it 3x.**"*
   >
   > The mechanism is right and the conclusion was wrong. Decode *is*
   > PCIe-bound, and it *was* tuned — not by making the link faster but by
   > making fewer bytes cross it, and by fixing an access pattern that was
   > getting 27% of the link's rate. 10.77 → 16.5 tok/s, a 1.53x, with no
   > quality loss. Nothing moves it 3x remains true.

2. **The default MoE kernel silently returns garbage on SM120** (bug 7). The
   model loads, serves, reports healthy, and emits one token forever. If you
   take nothing else from this document, take
   `--kernel-config '{"moe_backend":"marlin"}'`.

If you have two GPUs, put the whole model in VRAM and skip all of this.

---

## 1. Build environment

CUDA is a **triple constraint** on this platform:

| | |
|---|---|
| driver 595.84 | caps at CUDA **13.2** |
| Ubuntu 26.04, glibc 2.43 | CUDA **13.0** fails to compile (`rsqrt` exception-spec conflict in `mathcalls.h`) |
| ⇒ | **13.2.86 is the only satisfying point** |

```bash
uv venv --python 3.13 .venv-glm53
uv pip install --python .venv-glm53/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
# nvcc, cudart and nvrtc are THREE separate pip packages and drift apart.
# torch pulls cudart/nvrtc at 13.0; nvcc must be 13.2.86. Pin all of them:
uv pip install --python .venv-glm53/bin/python --no-deps \
  nvidia-cuda-nvcc==13.2.86 nvidia-cuda-runtime==13.2.86 nvidia-cuda-nvrtc==13.2.86
uv pip install --python .venv-glm53/bin/python --no-deps \
  --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.18 flashinfer-cubin==0.6.18   # 0.6.17 lacks SM90 NoPE MLA
```

**Two traps that cost hours:**

* **`nvcc` vs headers must match.** If they differ, Torch's `cuda.cmake` aborts
  configure with *"FindCUDA says CUDA version is 13.2, but the CUDA headers say
  13.0"* — 30 s in, after a long wait. Guard it before building:
  ```bash
  nvcc_ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  hdr=$(sed -n 's/^#define CUDART_VERSION *\([0-9]*\).*/\1/p' "$CUDA_HOME/include/cuda_runtime_api.h")
  [ "$nvcc_ver" = "$((hdr/1000)).$(((hdr%1000)/10))" ] || echo "MISMATCH"
  ```
* **Anything that re-resolves dependencies will undo the pins.** A plain
  `uv pip install -r requirements/*.txt` silently replaced `torch 2.13.0+cu130`
  with `+cpu` mid-session. Always `--no-deps`, and re-check `torch.version.cuda`
  is not `None` before building.

Then build (~20 min, `MAX_JOBS=44 NVCC_THREADS=2`, single arch
`TORCH_CUDA_ARCH_LIST=12.0f`), plus two more traps:

* pip CUDA wheels ship `lib/`, but FlashInfer's JIT hardcodes `-L$cuda_home/lib64`.
  Without `ln -sfn lib "$CUDA_HOME/lib64"` the first forward pass fails to LINK
  (`ld: cannot find -lcudart`) and surfaces as the misleading `Ninja build failed`.
* Pre-clone flash-attn and set `VLLM_FLASH_ATTN_SRC_DIR`; CMake's FetchContent
  path drags in ROCm/composable_kernel, which a CUDA build never uses.

**Run from a directory with no `vllm/` in it.** If the repo root is itself a
vLLM checkout, `sys.path[0]` shadows the compiled editable install and you get
`ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions`
— which reads as a broken build when the build is fine. `serve-opt.sh` now
enforces this with a hard check on `vllm.__file__`.

---

## 2. Memory: how 181 GiB fits in 96 GiB

90% of the checkpoint is routed MoE experts:

| component | GiB | |
|---|---|---|
| routed experts | **163.27** | offload these |
| attention (MLA + KDA) | 11.52 | resident |
| embeddings / lm_head | 2.36 | resident |
| shared expert | 2.02 | resident |
| norms, gates, dense MLP | 1.07 | resident |
| vision tower | 1.05 | disabled via `--limit-mm-per-prompt` |

> A later, profiler-era accounting of the same non-expert weights splits them
> slightly differently — attention projections 11.74 / shared expert 2.02 /
> dense MLP 1.58 / embedding 1.18 / lm_head 1.18, **18.01 GiB total**. The
> difference is bucketing, not a disagreement about the total. The later figure
> is the one used in the performance analysis, because that 18 GiB of *BF16*
> weight turns out to be the second-largest term in a decode step
> ([OPTIMIZING §1.2](OPTIMIZING-GLM53-ON-ONE-GPU.md#12-the-measured-decode-budget-2026-09-02-post-optimization)).

Use **UVA** offload with segment matching, not the prefetch backend:

```
--offload-backend uva --cpu-offload-gb 105 --cpu-offload-params experts
```

Top-8-of-288 routing means a decode step reads only the experts it selected, and
UVA's zero-copy reads exactly those. The prefetch backend moves *whole layers*
(~3.9 GiB each) regardless of routing — catastrophic for a sparse MoE.

> **Superseded value.** `--cpu-offload-gb 105` is the right number for the
> configuration described in this document. The optimized configuration uses
> **125**, which is *faster* only because the VRAM it frees funds a hot-expert
> cache. See [OPTIMIZING §4.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#44-the-matched-set--offload-more-to-go-faster).
> Do not raise the offload without the cache: 125 with no pinning is the slowest
> configuration we measured.

### Bug 1 — offloaded GPU memory is never released

Moving a parameter to host drops its GPU storage, but the caching allocator
keeps the block and, under `expandable_segments`, **extends the segment for the
next layer instead of reusing it**. Measured: `gpu_reserved` grew **4.06
GiB/layer** while live tensors grew 0.30. Over 43 MoE layers that is 175 GiB
reserved on a 96 GiB card — **offloading buys nothing** and you OOM around
layer 22, no matter what `--cpu-offload-gb` you pass.

*Fix:* `torch.cuda.empty_cache()` after each offloaded module (`uva.py`).
Afterwards `gpu_reserved` tracks `gpu_alloc` (4.44 vs 4.40).

### Bug 2 — host RSS is far above the offloaded bytes

Two independent rounding pathologies, both of which had to be turned off:

* **PyTorch's caching *host* allocator rounds to power-of-two blocks.**
  Pre-pinning routes the tensor through it: a 2.42 GiB `w13` takes 4 GiB, a
  1.21 GiB `w2` takes 2 GiB. **19.0 GiB of weights occupied 33.8 GiB of RAM.**
  *Fix:* hand `get_cuda_view_from_cpu_tensor` an **unpinned** tensor. The C++ op
  then pins it itself with an exact-size allocation
  (`csrc/libtorch_stable/cuda_view.cu`). Set
  `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`.
* **`cudaHostAlloc` hands out shared 4 GiB granules**, so the same
  ~2.4 GiB and ~1.2 GiB expert tensors each round up to a full block.
  *Fix:* allocate pinned host memory by `malloc` + `cudaHostRegister` instead —
  `PYTORCH_ALLOC_CONF=...,pinned_use_cuda_host_register:True,pinned_num_register_threads:8`.

With both applied, 19.0 GiB of weights occupy ~21.5 GiB resident.

> **An inconsistency in our own record, left visible rather than tidied away.**
> The launcher's inline comments attribute the *same* 19.0 GiB → 33.8 GiB
> measurement to *both* mechanisms, and quote the ratio variously as 1.85x and
> 1.94x (33.8/19.0 is 1.78x). Both mechanisms are real and both settings are
> needed, but the two comments cannot both be describing the same experiment.
> Treat the raw pair (19.0 → 33.8 GiB, fixed to ~21.5 GiB) as the measured fact
> and the ratios as unreliable.

> Counter-intuitive, and the opposite of what the flag name suggests. Pinning
> *earlier* is the pessimisation.

**Also:** there is no swap on many workstations, including this one. Pinned
pages never swap, so adding swap adds no capacity — only survivability for
everything *else* on the machine. `serve-opt.sh` instead refuses to launch
unless `MemAvailable >= CPU_OFFLOAD_GB + 30 GiB`.

> That allowance was **45 until 2026-09-02**, taken from a run that offloaded
> 110 GiB and peaked at ~168 GiB. Re-measured on the shipped configuration,
> steady state is **offload + ~16 GiB** — 141 GiB used at offload 125 — because
> the safetensors reads that produced the original peak go through *reclaimable
> page cache*, which `MemAvailable` already discounts. 30 keeps a 14 GiB
> cushion. An over-tight guard is not free either: it refuses launches that
> would have been fine, while claiming to be calibrated.

---

## 3. The model is NoPE MLA — four DeepSeek assumptions break

GLM-5.3 sets **`qk_rope_head_dim = 0`** (verified in the upstream config *and*
as the default in vLLM's own `Glm5NextConfig`). Almost every sparse-MLA code
path assumes DeepSeek's 64-dim RoPE.

### Bugs 3-4 — the fp8_ds_mla cache kernel

`concat_and_cache_mla` asserts `pe_dim == 64` and a fixed 656-byte entry, and
launches `dim3 block(96)` — 64 NoPE threads **plus a 32-thread RoPE warp that
must not run** when there is no RoPE data (it would read past an empty `k_pe`
and write past the entry).

*Fix:* allow `pe_dim` 0 or 64; size the entry `512 + 16 + 2*pe_dim` (**528**
bytes for NoPE, 656 for DeepSeek); launch `64 + pe_dim/2` threads. Mirror the
size in `mla_attention.py`'s `state_content_bytes`, which hardcoded 656.
Requires a rebuild.

### Bugs 5-6 — the rope-free lane is unreachable on SM120

`platforms/cuda.py` offers capability 12 only
`[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`, and the SM120 sparse kernel
implements only DeepSeek's 576-wide shape. FlashInfer's `_resolve_model_type`
supports exactly two layouts and GLM-5.3 matches **neither**:

| | latent | rope | d_qk | bytes |
|---|---|---|---|---|
| DSv3.2 / GLM-NSA | 512 | 64 | 576 | 656 |
| DSv4 | 448 | 64 | 512 | 584 |
| **GLM-5.3** | **512** | **0** | **512** | **528** |

The rope-free lane is **`FLASHINFER_MLA_SPARSE_SM90`** — despite the name — but
it is gated to `capability.major == 9` and hardcodes Hopper-only `backend="fa3"`.

*Fix:* list the SM90 backend for `head_size == 512`; gate `(9, 12)`; select
`fa2` on SM12x. FA3 compiles on SM120 but emits no device code, so you get
`no kernel image is available for execution on the device`. Needs FlashInfer
≥ 0.6.18. Matches upstream [vllm#53963](https://github.com/vllm-project/vllm/issues/53963);
the same approach was independently used on SM121 (DGX Spark).

> ~~**Caveat added 2026-09-02.** This is the backend that shows up in *both*
> unexplained `illegal memory access` failures we still have — below
> `--max-num-batched-tokens 8192`, and occasionally above it.~~
>
> **Amended later the same day** — and then amended again. The *"occasionally
> above 8192"* failure was an **Xid 31 MMU fault**: the kernel log named it, and
> CUDA reports such faults asynchronously, so the sparse-MLA frame was merely
> where the error surfaced. It is fixed by `VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus
> bounds checks in the expert-staging path
> ([OPTIMIZING §11](OPTIMIZING-GLM53-ON-ONE-GPU.md#11-stability--the-xid-31-mmu-fault)).
> The *"below 8192"* failure turned out to be a **second, independent** problem
> that survives that fix — 4096 still kills the engine on the first request
> ([OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all)).
> So it was two failures after all; this backend is the cause of neither.
> The lane itself is fine — and it is also the *only* lane: forcing the native
> `FLASHINFER_MLA_SPARSE_SM120` fails at startup because it expects DeepSeek's
> `qk_rope_head_dim=64` and a 576-wide query, which is exactly the NoPE mismatch
> this section is about.

---

## 4. Bug 7 — the one that produces no error

**Symptom:** the model loads, serves, `/health` is green, and every prompt
returns the same token forever (`"lock"`, id 1023). No error, no warning, no
NaN. Prompt-independent.

Sweeping all five NVFP4 MoE backends on SM120:

| backend | result |
|---|---|
| `flashinfer_trtllm` | refuses — *"kernel does not support current device"* |
| `flashinfer_cutedsl` | refuses — same |
| `flashinfer_cutlass` ← **auto picks this** | **silently wrong** |
| `cutlass` (vLLM) | **silently wrong** |
| **`marlin`** | **correct** |

Both CUTLASS-family kernels accept the work and return garbage. Their siblings
refuse loudly on the *same* hardware — so the failure mode you get is decided by
which kernel auto-selection happens to pick.

```
--kernel-config '{"moe_backend":"marlin"}'
```

Marlin warns *"your GPU does not have native support for FP4"* and falls back to
weight-only decompression.

> **Corrected 2026-09-02.** The original text said this fallback *"costs only
> ~6% versus the (broken) CUTLASS path, because decode is bandwidth-bound"*. It
> costs **nothing at all** — Marlin is simply the fastest NVFP4 MoE kernel on
> this GPU. Benchmarked at GLM dimensions with weights resident in VRAM, median
> of 100 iterations:
>
> | M | marlin | flashinfer_cutlass | vllm cutlass | b12x |
> |---|---|---|---|---|
> | 1 | **77.5 µs** | 99.8 | 110.0 | 409.4 |
> | 64 | **318.0** | 326.3 | 341.3 | 994.9 |
>
> The *"no native support for FP4"* warning is unconditional in the Marlin path
> and proves nothing about performance. Do not port anything to chase native
> FP4; end-to-end, `moe_backend=b12x` served at **1.2–1.3 tok/s**.
>
> A caution on that benchmark: it used 32 synthetic experts so the weights fit in
> VRAM, and in that harness all four backends agree with Marlin to cosine
> ≈ 0.986. **That does not contradict this bug.** A speed benchmark's
> correctness columns are not a correctness result on the real checkpoint's 288
> experts and their real scale distribution.

**Debugging note.** What finally isolated this was a differential test: swap one
component, hold everything else fixed. Before reaching it we had eliminated,
each by direct measurement, the indexer's uninitialised top-k buffer, PDL on
unvalidated silicon, zero expert scales, and the TileLang hyper-connection
kernels. Use `--load-format dummy` for this: it exercises construction and the
offloader without reading a byte of checkpoint, turning a 6-minute 181 GiB
experiment into a 30-second one. Truncating the config to 8 layers helps too
(see [OPTIMIZING §8.1](OPTIMIZING-GLM53-ON-ONE-GPU.md#81-truncate-the-model-to-8-layers)
for the trap in doing that).

---

## 5. Bug 8 — MTP hits an SM100-only kernel

Speculative decoding (the checkpoint carries a real MTP head at layer 45) routes
through DeepGEMM's paged-MQA attention.

*Fix:* `VLLM_USE_DEEP_GEMM=0`. Only the spec-decode path needs it; plain decode
is fine with DeepGEMM on. Use **`num_speculative_tokens: 2`** — 1 returns
correct text then dies with `CUDA error: an illegal memory access` partway
through a longer generation, and 3+ lowers acceptance sharply (GLM re-runs the
*same* MTP layer per draft step; at k=4 acceptance falls 60.2% → 33.7% and
throughput drops 19%).

MTP is worth **+31%** (8.7 → 11.9 tok/s sustained) and was the only lever that
materially moved a bandwidth-bound decode *before* the staging work.

> **Corrected 2026-09-02.** The original text said DeepGEMM's paged-MQA
> attention "asserts `arch_major == 10`". It no longer does — SM120 is accepted;
> the real constraint is `block_kv`, which must be exactly 64 for the fp8 indexer
> this model uses, which in turn requires `block_size` to be a multiple of 256.
> Full analysis in
> [OPTIMIZING §7](OPTIMIZING-GLM53-ON-ONE-GPU.md#7-corrections-to-the-original-runbook).
> `VLLM_USE_DEEP_GEMM=0` remains the shipped default because it costs nothing
> measurable and removes the need to reason about page sizes.

---

## 6. Working command (minimal correct configuration)

> This is the **correct-but-not-fast** configuration this document describes. For
> the 16.5 tok/s configuration, use `serve-opt.sh` or the expanded command in
> [README → Reproduce it](README.md#reproduce-it). One flag below is known to be
> wrong for the optimized path: `--no-enable-flashinfer-autotune` disables
> something vLLM's `-O2` default otherwise gives you for free.
>
> `--max-num-batched-tokens 4096` crashes the engine on the staging + pinning
> stack. **8192 is a floor.** ~~Withdrawn on 2026-09-02~~ as an artifact of an
> Xid 31 MMU fault that also hit at 8192 — and then **re-confirmed the same day**
> on a build where that fault is fixed: 4096 still dies, now on the *first*
> request. Two independent failures, one fixed and one not. See
> [OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all).

```bash
VLLM_USE_DEEP_GEMM=0 \
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
vllm serve <checkpoint> \
  --served-model-name glm53-flash \
  --port 8002 \
  --max-model-len 524288 \
  --gpu-memory-utilization 0.96 \
  --offload-backend uva --cpu-offload-gb 112 --cpu-offload-params experts \
  --kernel-config '{"moe_backend":"marlin"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
```

Boot is ~6 min: 181 GiB load, then a **silent** multi-minute Marlin repack
(96% CPU, no log output — do not mistake it for a hang), then graph capture.

## 7. Measured

| | this configuration | shipped configuration |
|---|---|---|
| decode | 8.7 tok/s (no MTP) → **10–12 tok/s** (MTP-2) | **16.50 tok/s** |
| context | 512k | **327,680** — and, unlike every earlier setting here, *tested*: a **304,800-token prompt prefills** in 205 s (1,487 tok/s). The usable ceiling is set by free VRAM, not by this flag ([OPTIMIZING §12](OPTIMIZING-GLM53-ON-ONE-GPU.md#12-long-context--the-ceiling-is-free-vram-not---max-model-len)) |
| prefill | ~350–570 tok/s cold; ~1,270 tok/s prefix-cached | ~~not re-measured~~ → **1729–1771 tok/s** on fresh ~10.3k-token prompts (TTFT 5.82–5.96 s), 1576 tok/s at 35,571 tokens — logged in `artifacts/prefill-fresh-2026-09-02.txt`. It measured **614 tok/s** until grouped staging landed — the staging cap that makes decode fast made prefill fall back to zero-copy BAR reads ([OPTIMIZING §10](OPTIMIZING-GLM53-ON-ONE-GPU.md#10-prefill--the-term-the-decode-benchmarks-hid)) |
| KV cache | ~14 KB/token at long context | **derived** from the context as `MAX_LEN × 17,200 B` → ~5.25 GiB / **375,543 tokens** at 327,680 (1.15x concurrency). Measured cost 15.0–15.7 KiB/token; the earlier 4 GiB cap gave 272,771 tokens at 262k |
| VRAM at 512k | 71.5 weights+ctx · 5.1 activation · 14.5 KV · 0.3 graphs | 55.3 model · 5.25 KV · ~24 GiB hot-expert cache (**54** slots, down from 62–66 to leave ~6.2 GiB free for long-prompt activations) |
| context cost | 512k costs ~7% decode vs 256k; 1M ~6% more again | — |

> **Measure KV at *your* context.** At 16k it appears to be 40 KB/token because
> per-sequence KDA state dominates; that is the figure `FINDINGS.md` originally
> quoted, and it is not wrong, it is just measured somewhere else.

**Sizing for one agent:** vLLM sizes KV from *leftover* budget, not need — at
512k it provisioned 2.1x concurrency, and at offload 125 with no cap it
provisioned **27.17 GiB / 1,870,432 tokens = 7.1x**. Cap it with
`--kv-cache-memory-bytes` and give the difference to resident experts; each GiB
moved off the host is a GiB that does not cross PCIe every token. Note that
setting it makes vLLM **skip memory profiling entirely**, which has consequences
of its own ([OPTIMIZING §6.6](OPTIMIZING-GLM53-ON-ONE-GPU.md#66-reserving-staging-buffers-before-the-cap-check--reverted)).

## 8. Not fixed

* **Fused-`w13` scale collapse (all modelopt NVFP4 MoE checkpoints).**
  `gate_proj.weight_scale_2` and `up_proj.weight_scale_2` are separate tensors,
  quantized independently — 179 of 288 experts differ by ~1.4x. vLLM fuses them
  and keeps only the gate scale, mis-scaling the up-projection. Not caused by
  abliteration; hardware-independent. The official NVFP4 build will warn the
  same way.
* **The non-UVA offload fallback is broken.**
  `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` dies with an illegal memory access and
  leaves a **D-state thread holding all VRAM** — only a reboot clears it, and
  `nvidia-smi --gpu-reset` refuses if the card is primary.
* **`--max-num-batched-tokens` below 8192 crashes the optimized stack**, and
  this one went round twice. It was first reported as an illegal memory access
  on the second request; ~~withdrawn on 2026-09-02~~ when the same fault hit at
  8192 after 12 requests (kernel log: **Xid 31 MMU fault**, addressed by
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus staging bounds checks,
  [OPTIMIZING §11](OPTIMIZING-GLM53-ON-ONE-GPU.md#11-stability--the-xid-31-mmu-fault));
  then **re-confirmed** on the fixed build, where 4096 dies on the *first*
  request. Two independent failures: the Xid is fixed, the floor is not.
  Unexplained, and expensive — shrinking the prefill chunk was the cheap way to
  buy VRAM for long context
  ([OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all)).
  (The pre-optimization 2026-08-29 stack ran fine at 4096, so it arrived with
  the staging work.)
* **The hot-expert cache is allocated after vLLM's memory profiling**, so the
  engine sizes KV against VRAM that is not free and serving can die with `CUDA
  out of memory. Tried to allocate 124.00 MiB` on a 95 GiB card. Unfixed; the
  only lever is fewer hot slots or a higher offload.

## 9. Honest assessment

It works, and as far as the public record goes this was the first single-GPU
SM120 deployment of this model.

> **Revised 2026-09-02.** The original closed with: *"**10-12 tok/s is a hard
> ceiling set by PCIe**, and a model that fits entirely in VRAM will beat it by a
> wide margin."*
>
> Half of that survived. The ceiling was **not** 10–12; it was 16.5, once the
> gather stopped wasting three quarters of the link and the resident experts
> stopped being chosen at random. The other half is unchanged and still the
> honest advice: Qwen3.8-Flash-Next on this same card runs at **173 tok/s**
> because it fits. This is worth doing if you need *this* model's quality on
> *this* hardware and can tolerate batch-style latency; otherwise fit the model
> to the card.
