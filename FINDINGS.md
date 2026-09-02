# GLM-5.3-Flash on a single RTX PRO 6000 (SM120) — condensed

The eight bugs between a fresh checkout and correct output, on one page.
vLLM built from PR #53906 (`0.29.0.dev0+glm53`), FlashInfer 0.6.18.

For the **performance** story — how 10.77 tok/s became 16.5 — start at
[README.md](README.md). This page is correctness only.

> **Revision note (2026-09-02).** Written 2026-08-29. The bug content is
> unchanged. Two things in the original were wrong and are corrected inline: the
> throughput figures (now superseded) and the claim that MoE kernel choice moves
> throughput by only ~6% (Marlin is simply the fastest, by a lot). The original
> heading also said "seven bugs" over a list of eight.
>
> **Revised again later on 2026-09-02.** A third correction: the
> `--max-num-batched-tokens` 8192 "floor" added earlier that day was reported as
> an **artifact** of an intermittent Xid 31 MMU fault that also hit at 8192.
> And the run line now carries a prefill number, because prefill — not decode —
> turned out to be the day's largest quality-neutral win (614 → 1729–1771 tok/s,
> logged in `artifacts/prefill-fresh-2026-09-02.txt`).
>
> **Revised a fourth time, later the same day.** The withdrawal above was wrong.
> Re-tested on a build where the Xid 31 fault is fixed, `MAX_BATCHED=4096`
> **still kills the engine**, now on the first request. There were two
> independent failures; one is fixed and one is not, and **8192 is a floor**.
> Also new: the context is now **327,680 and validated** — a 304,800-token
> prompt prefills — with the KV cache derived from it, funded by cutting hot
> slots 62 → 54 at a cost inside the run-to-run spread
> ([OPTIMIZING §12](OPTIMIZING-GLM53-ON-ONE-GPU.md#12-long-context--the-ceiling-is-free-vram-not---max-model-len)).

## Run it

> **Do not use the original one-liner.** It read
> `SPEC=1 GPU_UTIL=0.96 MAX_LEN=262144 MAX_BATCHED=4096 ./serve.sh`. Three
> things are wrong with it now:
> - **`MAX_BATCHED=4096` crashes the engine. 8192 is a floor.** (This was
>   briefly ~~withdrawn~~ as an artifact of an Xid 31 MMU fault that also hit at
>   8192, then re-confirmed on the fixed build, where 4096 dies on the *first*
>   request. Two independent failures — see
>   [OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all).)
> - `GPU_UTIL` buys nothing when the KV cache is explicitly capped.
> - `MAX_LEN` is now **327,680**, and the KV cache derives from it
>   (`MAX_LEN × 17,200 B`) so the two cannot disagree.

```bash
# serve-opt.sh lives in the vLLM working tree (here, ~/Projects/vllm-glm53),
# not in this repository -- every default in it is tied to that tree's patches.
cd <your-vllm-glm53-checkout> && ./serve-opt.sh   # all tuned defaults
```

The two non-obvious mandatory settings are launcher defaults:
`MOE_BACKEND=marlin` (bug 7 — without it the model loads, serves, and emits one
token forever with no error) and `VLLM_USE_DEEP_GEMM=0` (bug 8).
`SPEC=1` enables MTP; `SPEC_TOKENS` defaults to 2 (1 crashes mid-generation, 4
collapses acceptance).

**Measured: ~~10.6-12.1 tok/s~~ → 16.50 tok/s decode**, sustained over 8
back-to-back 200-token generations at 262k configured context (14.99 tok/s at the
current 54 hot slots, which trade a little decode for long-prompt headroom), **plus ~~614~~ → 1729–1771 tok/s prefill** (a fresh ~10.3k-token
prompt reaches its first token in 5.82–5.96 s instead of 16.8; logged in
`artifacts/prefill-fresh-2026-09-02.txt`). The longest prompt actually prefilled
is **304,800 tokens** (205 s, 1,487 tok/s). A 40-request varied soak plus a
35,553-token prompt ran 0 failures / 0 Xid. See [README](README.md) for the full
conditions.

## Hardware reality

| | |
|---|---|
| checkpoint | 181.29 GiB NVFP4 (163.27 GiB of it routed experts) |
| VRAM | ~95 GiB — cannot hold the model |
| approach | keep attention/embeddings/shared-expert resident, stream routed experts from host RAM over PCIe via UVA zero-copy |
| measured | 4.43 GiB of expert weights touched per decoded token |

Decode is **bandwidth-bound, not compute-bound** — but the bytes-per-token *and*
the bytes-per-second both turned out to be tunable. See
[OPTIMIZING](OPTIMIZING-GLM53-ON-ONE-GPU.md).

## The eight bugs

### 1. Offloaded GPU memory was never released
`UVAOffloader` moves a parameter to host, but the caching allocator keeps the
freed block and — under `expandable_segments` — extends the segment for the next
layer rather than reusing it. Measured: `gpu_reserved` grew **4.06 GiB/layer**
while live tensors grew 0.30. Over 43 MoE layers that is 175 GiB reserved on a
96 GiB card, so **offloading bought nothing**.
*Fix:* `torch.cuda.empty_cache()` after each offloaded module (`uva.py`).

### 2. Host RSS was far above the offloaded bytes
Two independent rounding pathologies. PyTorch's caching *host* allocator rounds
to power-of-two blocks (2.42 GiB `w13` → 4 GiB, 1.21 GiB `w2` → 2 GiB), and
`cudaHostAlloc` hands out shared 4 GiB granules. **19.0 GiB of weights occupied
33.8 GiB of RAM.**
*Fix:* hand `get_cuda_view_from_cpu_tensor` an **unpinned** tensor so the C++ op
pins it exact-size (`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`), **and**
allocate pinned memory by `malloc` + `cudaHostRegister`
(`PYTORCH_ALLOC_CONF=...,pinned_use_cuda_host_register:True`). → ~21.5 GiB.

### 3-4. `fp8_ds_mla` cache kernel assumed DeepSeek's shape
GLM-5.3 is **NoPE** MLA (`qk_rope_head_dim=0`). `concat_and_cache_mla` asserted
`pe_dim == 64` and a fixed 656-byte entry; the launch hardcoded `dim3 block(96)`,
i.e. 64 NoPE threads + a 32-thread RoPE warp that must not run when pe_dim=0.
*Fix:* allow `pe_dim` 0 or 64, size the entry `512 + 16 + 2*pe_dim`, launch
`64 + pe_dim/2` threads (`csrc/libtorch_stable/cache_kernels.cu`).

### 5-6. The rope-free MLA lane was unreachable on SM120
`platforms/cuda.py` listed only `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` for
capability 12, and SM120's sparse MLA implements only DeepSeek's 576-wide shape.
The rope-free lane is `FLASHINFER_MLA_SPARSE_SM90`, gated to `major == 9` and
hardcoding Hopper-only `backend="fa3"`.
*Fix:* list the SM90 backend for `head_size == 512`; gate `(9, 12)`; use `fa2`
on SM12x (`fa3` compiles but emits no kernel image → "no kernel image is
available for execution on the device"). Needs FlashInfer >= 0.6.18.
Matches upstream vllm-project/vllm#53963.

### 7. NVFP4 MoE kernels silently corrupt on SM120  ← the expensive one
| backend | result |
|---|---|
| `flashinfer_trtllm` | refuses — "kernel does not support current device" |
| `flashinfer_cutedsl` | refuses — same |
| `flashinfer_cutlass` (**auto default**) | **silently wrong** |
| `cutlass` (vLLM) | **silently wrong** |
| `marlin` | **correct** |

Both CUTLASS-family kernels accept the work and return garbage; the model emits
one token forever (`"lock"`, id 1023) with no error or warning. TRTLLM and
CuteDSL refuse loudly on the same hardware — the silent one is what auto picks.

### 8. MTP hits a DeepGEMM kernel that (then) rejected SM120
The MTP draft head routes through DeepGEMM's paged-MQA attention.
*Fix:* `VLLM_USE_DEEP_GEMM=0` — only the spec-decode path needs it, plain decode
works with DeepGEMM enabled.
*Corrected 2026-09-02:* the assert no longer excludes SM120; the real constraint
is `block_kv == 64` for the fp8 indexer, i.e. `--block-size` must be a multiple
of 256. See [OPTIMIZING §7](OPTIMIZING-GLM53-ON-ONE-GPU.md#7-corrections-to-the-original-runbook).

## Also found, not fixed

- **Fused-w13 scale collapse (affects ALL modelopt NVFP4 MoE checkpoints, not
  just the abliterated one):** the checkpoint stores
  `experts.N.gate_proj.weight_scale_2` and `experts.N.up_proj.weight_scale_2` as
  separate tensors, quantized independently, so their global scales differ —
  measured 179 of 288 experts, ~1.4x apart. vLLM fuses them into `w13` and keeps
  only `[:, 0]` (the gate scale), mis-scaling the up-projection. Inherent to how
  modelopt quantizes, NOT caused by abliteration.
- **Non-UVA offload fallback is broken:**
  `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` dies with `CUDA error: an illegal
  memory access`, leaving a D-state thread holding all VRAM (only a reboot
  clears it).
- **`--max-num-batched-tokens` below 8192 crashes the optimized stack.** First
  reported as an illegal memory access on the second request; ~~withdrawn~~ as
  an Xid 31 MMU fault that also hit at 8192 (fixed by
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + staging bounds checks, mechanism unproven);
  then **re-confirmed** on the fixed build, where 4096 dies on the first
  request. Two independent failures — the Xid is fixed, the floor is not. This
  closes the cheap route to VRAM headroom for long context.
- **The hot-expert cache is invisible to vLLM's memory profiling** (it is
  allocated afterwards), so KV is sized against VRAM that is not free and
  serving can die with `CUDA out of memory. Tried to allocate 124.00 MiB` on a
  95 GiB card. The principled fix broke serving and was reverted; headroom comes
  only from fewer hot slots or a higher offload.
- **The native SM120 sparse-MLA attention backend refuses this model** (it
  expects DeepSeek's `qk_rope_head_dim=64` / 576-wide query; GLM-5.3 is NoPE).
  The SM90 kernel runs on Blackwell by fallback.
- **The usable context is set by free VRAM, not by `--max-model-len`.** Long
  prompt activations are allocated per request and scale with length, so a
  180k-token prompt died with 3.4 GiB free while 304,800 tokens prefilled with
  6.2 GiB free. Never advertise a context you have not prefilled at.

## Measured

- 8.7 tok/s (marlin, no MTP)
- 10.6–12.1 tok/s (marlin + MTP-2), 11.9 sustained over 600 tokens — +31%, not
  the ~2x a naive bytes-per-token model suggests. GLM's MTP head is one layer
  re-run per speculative token, so acceptance falls off and each rejected draft
  costs a full forward pass; vLLM warns about this explicitly.
- **16.50 tok/s** with gathered-DMA staging + frequency-pinned hot experts
  ([OPTIMIZING](OPTIMIZING-GLM53-ON-ONE-GPU.md))
- ~~9.2 tok/s (cutlass, wrong output) — kernel choice moves throughput only ~6%~~
  → **superseded.** Marlin is the *fastest* NVFP4 MoE kernel on this GPU at every
  batch size (77.5 µs at M=1 vs 99.8 flashinfer_cutlass / 110.0 cutlass / 409.4
  b12x), and `moe_backend=b12x` served end-to-end at **1.2–1.3 tok/s**. The
  original 6% figure compared two *end-to-end* runs where the transfer dominated
  everything, which is a different measurement from kernel speed.
- **Prefill 614 → 1729–1771 tok/s** by staging the offloaded experts in 64-wide
  groups instead of bailing out of staging and reading them zero-copy over the
  BAR. Two fresh ~10.3k prompts: TTFT 5.82 / 5.96 s (was 16.8); a fresh 35,571
  prompt: 22.57 s = 1576 tok/s. Gated to prompts of ≥ 48 tokens — grouping
  *costs* a 14-token prompt 2.05 s, where zero-copy takes 0.85 s. Decode at
  those contexts is 18.2–18.4 tok/s. Logged:
  `artifacts/prefill-fresh-2026-09-02.txt`.
  ([OPTIMIZING §10](OPTIMIZING-GLM53-ON-ONE-GPU.md#10-prefill--the-term-the-decode-benchmarks-hid))
- MTP-2 on a coding prompt at T=0: 17.10 tok/s, 64% of draft tokens accepted =
  1.28 per step (position 0: 75%, position 1: 53%); a different generation gave
  1.81. Deeper drafts are blocked — depth 5 exceeded the 4 GiB KV cap in force
  at the time (the cap is 5.25 GiB now; not retested), depth 4 trips a DeepGEMM
  `block_kv` assert.
- KV cache: ~40 KB/token at 16k (per-sequence KDA state dominates); 15.0–15.7
  KiB/token at long context (4 GiB → 272,771 tokens at 262k; 5.25 GiB →
  375,543). The launcher now sizes it as `MAX_LEN × 17,200 B`.
  **Measure it at your own context.**
- Hot-expert slots at the top of the ladder are a **weak** lever: 48 slots →
  14.92 tok/s, 54 → 14.99, inside the ±1 tok/s spread. Slots bought long-context
  headroom almost for free. (The full ladder is not weak — 0 → 66 slots is the
  whole 13.53 → 16.50 win.)
- `reasoning_effort` accepts only `low` / `high` / `max`; **anything else
  silently becomes `max`**. On one hard problem: 183 / 484 / 1,812 reasoning
  tokens, with `max` taking ~2x the wall time of `high` for a shorter answer.
  Use `high` in agent loops
  ([OPTIMIZING §13.2](OPTIMIZING-GLM53-ON-ONE-GPU.md#132-reasoning_effort-measured)).
- Model natively supports 1,048,576 context; 1M costs ~23% decode vs 256k
