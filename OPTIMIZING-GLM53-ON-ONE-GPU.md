# Optimizing GLM-5.3-Flash (NVFP4) on a single RTX PRO 6000 / SM120

Companion to [`RUNNING-GLM53-ON-ONE-GPU.md`](RUNNING-GLM53-ON-ONE-GPU.md), which
covers getting the model to **run** — the CUDA triple-constraint, the build
traps, and eight correctness bugs (one of which returns garbage with no error).
This one covers getting it to run **fast**: the measured cost model, the two
changes that produced the result, and a long list of things that did not work.

Measured on: Ubuntu 26.04 (glibc 2.43) · RTX PRO 6000 Blackwell Workstation
(SM120, 96 GiB) · AMD Ryzen 9 9950X3D · **182 GiB DDR5, dual-channel** ·
PCIe 5.0 x16 · driver 595.84 · vLLM PR #53906 + local patches · FlashInfer
0.6.18.

> **Revision note (2026-09-02).** This document was written on 2026-09-01, when
> the best measured figure was 10.8 tok/s and the gathered-DMA work was an
> unfinished projection. It has been reconciled with the finished
> implementation, the torch-profiler trace, and the hot-expert pinning result.
> Superseded claims are marked **~~struck through~~** or in a `SUPERSEDED`
> call-out rather than deleted — several of them were reasonable at the time and
> the reason they were wrong is the useful part.
>
> **Two corrections to this document's own header**, both material:
> - It described the host memory as *"DDR5-5600 **quad-channel**"*. It is a
>   dual-channel desktop platform. Measured CPU STREAM read is **64.6 GB/s**,
>   not the ~179 GB/s that assumption implies. Several estimates elsewhere were
>   built on the wrong figure.
> - It quoted VRAM bandwidth as 1792 GB/s while
>   [`FURTHER-OPTIMIZATION-RESEARCH.md`](FURTHER-OPTIMIZATION-RESEARCH.md)
>   quoted ~1670 GB/s. Neither was measured here; both are vendor-derived.
>   Treat it as **~1.8 TB/s** and do not build a tight argument on it.

> **Second revision note (2026-09-02, later the same day).** Everything above
> §10 is about **decode**, which turned out to be the wrong place to spend the
> last of the day. Two sections have been appended and one has been withdrawn:
>
> - **§10 Prefill** — the term no decode benchmark in this document ever
>   measured. 614 → **1729–1771 tok/s**, a 10k-token prompt's TTFT 16.8 s →
>   ~5.9 s. Larger than any single decode win here, and quality-neutral.
>   Logged: `artifacts/prefill-fresh-2026-09-02.txt`.
> - **§11 Stability** — the recurring `illegal memory access` was an **Xid 31
>   MMU fault**, not a batch-size threshold.
> - **§6.7 is withdrawn.** The `--max-num-batched-tokens` 8192 "floor" was an
>   artifact of that fault, which also hit at 8192.

> **Third revision note (2026-09-02, later still).** Four more things landed, and
> one of them reverses a reversal:
>
> - **§6.7 is withdrawn no longer.** Re-tested with the stability fix shipped,
>   `--max-num-batched-tokens 4096` *still* kills the engine on the first
>   request. The floor is real and is a **separate** failure from the Xid 31 MMU
>   fault. See §6.7 and §11.4 for the full sequence — it is a worked example of
>   how a correct withdrawal can still reach a wrong conclusion.
> - **§12 Long context** — the advertised `--max-model-len` was never the usable
>   ceiling; **free VRAM** is. Context is now 327,680 and a **304,800-token
>   prompt has actually been prefilled** (205 s, 1,487 tok/s).
> - **§4.5 carries a correction against this document's own claim:** the
>   throughput cost of cutting hot slots was **over-claimed** here. 48 slots
>   measure 14.92 tok/s and 54 measure 14.99 — inside the run-to-run spread.
> - **§13 Serving it to agents** — the two things that mattered more than a
>   tok/s once this became a coding backend: an "amnesia" symptom that was the
>   engine crashing, and `reasoning_effort` measured rather than assumed.

---

## Bottom line

> **SUPERSEDED.** The original bottom line read: *"GLM-5.3-Flash tops out around
> 10.8 tok/s on this hardware and that is not interactive. The ceiling is PCIe,
> not compute, not kernels, and not configuration."*
>
> The first sentence is wrong — it is **16.5 tok/s** — and the second is right in
> a way the first sentence obscured. The ceiling *is* PCIe; what was wrong was
> the assumption that nothing could change how many bytes cross it.

**GLM-5.3-Flash runs at 16.5 tok/s of decode and ~1750 tok/s of prefill on this
hardware with no loss of output quality.** Getting there took exactly three
changes: a gather kernel that reads host memory the way the DMA engine wants
(§2, §3), a frequency-ranked VRAM cache of hot experts (§4), and — for prefill,
which the first two do not help at all — walking the experts in groups instead
of bailing out of staging (§10). Everything else in this document is a negative
result.

On the *same card* Qwen3.8-Flash-Next (NVFP4, fits entirely in VRAM at 78.5 GiB)
measures **173 tok/s** with nothing left to tune. Do this only if you need *this
specific model's* quality and can tolerate batch-style latency. If any model
that fits in 96 GiB will do, fit the model to the card and stop reading.

That said, §2 and §4 apply to **any** MoE offload stack, not just this model.

---

## Contents

1. [The performance model — where the time actually goes](#1-the-performance-model)
2. [The transfer-mode discovery](#2-the-transfer-mode-discovery)
3. [The gather kernel: device-side indices and CUDA graphs](#3-the-gather-kernel)
4. [Hot-expert pinning: frequency beats module order](#4-hot-expert-pinning)
5. [A false start, documented in full](#5-a-false-start-documented-in-full)
6. [Configuration tuning: eight negative results](#6-configuration-tuning-eight-negative-results)
7. [Corrections to the original runbook](#7-corrections-to-the-original-runbook)
8. [Reproducibility tooling — iterate in 90 s, not 6 min](#8-reproducibility-tooling)
9. [Where it stands and what is left](#9-where-it-stands-and-what-is-left)
10. [Prefill — the term the decode benchmarks hid](#10-prefill--the-term-the-decode-benchmarks-hid)
11. [Stability — the Xid 31 MMU fault](#11-stability--the-xid-31-mmu-fault)
12. [Long context — the ceiling is free VRAM, not `--max-model-len`](#12-long-context--the-ceiling-is-free-vram-not---max-model-len)
13. [Serving it to agents](#13-serving-it-to-agents)

---

## 1. The performance model

### 1.1 The arithmetic (2026-09-01, pre-optimization)

GLM-5.3-Flash is 320B total / 18B active, 45 layers, 42 MoE layers, 288 routed
experts, top-8. Per decoded token the model touches **4.43 GiB** of expert
weights (42 × 8 × ~13.5 MiB).

The checkpoint is 181.3 GiB against ~95 GiB of usable VRAM, so most experts live
in host RAM behind a UVA (zero-copy) mapping. Measured on the live server at 512k
context, MTP-2, `--cpu-offload-gb 105` (**~65% of expert bytes host-resident**):

| quantity | measured |
|---|---|
| decode | 9.7 – 10.8 tok/s |
| forward passes/s | 4.84 |
| **time per forward pass** | **207 ms** |
| tokens per pass (MTP-2) | 2.00 – 2.20 |
| bytes crossing PCIe per pass | 3.08 GB |
| **implied effective bandwidth** | **15.1 GB/s** |

```
host experts over PCIe   3.08 GB / 15.1 GB/s   =  204 ms   (98.6%)
GPU-resident experts     1.67 GB / ~1800 GB/s  =    0.9 ms
everything else                                    ~2 ms
```

204 of 207 ms was one transfer, at 27% of the link's rate. That framing was
correct and it pointed at the right thing.

### 1.2 The measured decode budget (2026-09-02, post-optimization)

Arithmetic is not a profile. A torch-profiler trace of the **optimized**
configuration (offload 125, 66 hot slots, CUDA graphs on) gives, per decode step,
median of 12:

```
staging gather (PCIe)  34.06 ms   <- still dominant
dense bf16 GEMM        18.98 ms   <- second, and never suspected
MoE experts (marlin)    3.58 ms
attention               0.47 ms   <- 0.1% of GPU time
norm/activation/other   1.38 ms
```

GPU busy exceeds the wall span by ~13.7 ms, so some overlap already happens.

Two things follow that no amount of arithmetic would have produced:

- **Attention is not the bottleneck and never was.** On a 262k-context model.
  0.47 ms.
- **The second-largest term is 18.01 GiB of BF16 weights that the NVFP4
  quantizer never touched** — attention projections 11.74 GiB, shared expert
  2.02, dense MLP 1.58, embedding 1.18, lm_head 1.18. All read every token. At
  ~1.8 TB/s the floor is 10.74 ms, so that phase runs at ~57% of peak bandwidth.

> **Note on a discrepancy.** RUNNING §2 lists the non-expert weights as
> attention 11.52 / embeddings+lm_head 2.36 / shared expert 2.02 / norms, gates,
> dense MLP 1.07 — a slightly different split summing to ~17 GiB. The
> profiler-era breakdown above (17.7 GiB itemised, 18.01 GiB total) is the later
> measurement and is the one used in this document. The difference is in how
> "dense MLP" and "norms/gates" were bucketed, not in the total.

**How to capture this trace.** This fork has **no `VLLM_TORCH_PROFILER_DIR`** —
the environment variable does nothing. Use the CLI flags
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=DIR`,
then `POST /start_profile` and `POST /stop_profile`. `nsys` and `ncu` are not
installed here.

**How not to read it.** The captured window contained 2 prefill steps at ~811 ms
of CUDA time each alongside the 12 decode steps, so the profiler's aggregate
operator table is dominated by *prefill*: the Marlin MoE kernel appears as the
single largest line item at 37% of CUDA time, the exact opposite of its 3.58 ms
share of a decode step. Decompose decode steps only.

> **And then read the prefill steps anyway.** Those two 811 ms steps were the
> only visible sign of §10's finding, and this document filed them under
> "artifact to exclude" for a day. Excluding a term from a decode budget is
> correct; concluding it does not matter is not. Under grouped staging (§10)
> those same chunks run ~2.9x faster.

### 1.3 Why residency looked like a dead end, and why that was half right

> **SUPERSEDED.** The original §1.2 concluded: *"residency is a dead end on one
> card. The transfer rate is not."* Both halves need qualification.

The original argument: capping the KV cache freed VRAM and moved residency
33.4% → 35.2%, cutting PCIe bytes by 2.7% — inside run-to-run noise, and that is
exactly what was measured (§6.1). To halve the transfer you would need ~67%
residency, and the VRAM budget caps it near 40%.

That argument is arithmetically fine and **it is still true for residency chosen
by module order**. What it missed is that a resident expert is only worth
anything if it is *used*, and vLLM's offloader picks residents arbitrarily. Once
residency is chosen by **frequency**, 23% of capacity buys 61% of the fetches
(§4), and the same VRAM does 2.7x the work. Residency is not a dead end; *blind*
residency is.

### 1.4 The offload-sensitivity experiment, and the ceiling it predicts

The single most informative measurement of 2026-09-01 was varying how much
crosses PCIe and fitting a line (staging on, no pinning yet):

```
offload 102 GiB (63% of experts remote)  -> 78.3 ms/token
offload 125 GiB (77% remote)             -> 87.6 ms/token

linear fit:  34.8 ms fixed  +  68.6 ms x traffic_fraction
```

So **transfer is only ~55% of the time**, not 98.6% — the fixed 34.8 ms is real
and does not go away. Consequences:

- With **infinite VRAM and zero PCIe traffic**, this stack caps near **29 tok/s**.
- At the achievable residency, it predicted **~16 tok/s**. Hot-expert pinning
  delivered 16.5. The model was right.
- **No overlap is available architecturally.** The MoE router depends on the same
  layer's attention output, so an expert fetch cannot be hidden behind that
  layer's compute. During decode the GPU shows SM util 96–100% but only 210 W of
  600 W and 14.6% memory-controller utilisation — SMs stalled on PCIe, not
  computing.

---

## 2. The transfer-mode discovery

The most portable finding here, and it generalises to any MoE-offload stack.

### 2.1 The measurement

Fetching 8 experts × 13.5 MiB (108 MiB, one MoE layer's working set for one
token) from host memory to VRAM, five ways:

| method | time | bandwidth |
|---|---|---|
| `index_select` from the UVA view | 10.50 ms | **10.78 GB/s** |
| per-expert `copy_` from the UVA view | 2.52 ms | **44.89 GB/s** |
| raw `cudaMemcpyAsync` from the UVA view | 2.53 ms | 44.75 GB/s |
| per-expert `copy_` from a pinned host tensor | 2.54 ms | 44.60 GB/s |
| single contiguous DMA (practical ceiling) | 1.96 ms | 57.86 GB/s |

### 2.2 The lesson

**UVA memory is not inherently slow. The gather *kernel* is.**

`index_select` on a host-mapped tensor launches a gather kernel: the SMs issue
64–128 B loads across the PCIe BAR from inside the kernel, with limited
outstanding requests and no DMA-engine involvement. That is the 10.78 GB/s path,
and it is what a naive "just read the offloaded weights" implementation gets.

A per-row `copy_` on the *same view* lowers to `cudaMemcpyAsync`, which runs on
the copy engine at full link speed — **4.2x faster from identical memory**.

Two corollaries that saved a large amount of work:

- **You do not need a pinned host-side handle.** Copying from the device-side UVA
  view is exactly as fast as copying from the original pinned host tensor
  (44.89 vs 44.60 GB/s). Hours went into machinery to obtain a host handle before
  this was measured. See §5.
- **Per-expert blocks are contiguous.** The scatter is in *which* experts get
  picked, never within one, so 8 separate 13.5 MiB copies reach ~99% of the
  single-contiguous-DMA ceiling. Batching them buys nothing.

### 2.3 Where to intervene in vLLM

`MarlinExperts.apply()` in
`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` receives both the
full expert weights (`w1`, `w2`) and `topk_ids`, which is everything needed:

1. take the experts this step selected,
2. gather those rows into a reused VRAM buffer (never `index_select`),
3. hand the kernel an `expert_map` mapping global id → staged row, −1 elsewhere.

Step 3 is the neat part: `expert_map` is vLLM's **existing** expert-parallelism
translation table. The Marlin kernels already honour it, so **no kernel change is
required** — staged weights of shape `[n_selected, ...]` plus a local map is
exactly the shape EP already produces.

Two guards are essential, because getting them wrong produces plausible garbage
rather than an error (the same failure class as bug 7 in the runbook):

- **Only gather tensors whose dim 0 really is the expert axis.** A per-tensor
  scalar must pass through untouched; indexing it by expert id silently reads
  adjacent memory.
- **Bail out entirely if any *other* per-expert tensor would remain globally
  indexed** while the staged ones are locally indexed — biases, input scales,
  zero-points, `g_idx`. Mixed indexing reads the wrong rows with no error.

Detect offloaded tensors with `cudaPointerGetAttributes`; a UVA view reports
`cudaMemoryTypeHost` (1). Cache the answer per `data_ptr` — weights are allocated
once, so it costs one driver call per parameter for the process lifetime.

> Practical note: resolve the symbol from the **already-loaded** runtime
> (`ctypes.CDLL(None)`), not `ctypes.CDLL("libcudart")`, which would map a second
> copy of the CUDA runtime into the process. And oversize the attribute buffer —
> `cudaPointerAttributes` has grown across CUDA releases and declaring it too
> small corrupts the heap.

---

## 3. The gather kernel

> This section did not exist on 2026-09-01. It was §8's "what we would do next",
> item 1, carrying a **projected** ~30 tok/s. The projection was wrong (the
> fixed 34.8 ms of §1.4 is why); the change itself was worth **10.77 → 13.53
> tok/s**.

The per-row `copy_` of §2 is fast but needs the expert ids **on the host** to
drive the loop, and `experts.tolist()` is a device-to-host sync, which CUDA graph
capture forbids. That forced `--enforce-eager` on the whole engine.

At 207 ms per forward pass that trade is obviously fine. It stops being fine as
the pass gets shorter: **11.77 tok/s eager vs 13.53 tok/s with graphs**, with
identical staging. Eager was costing 13% — most of what the staging won.

So the gather is a Triton kernel taking indices as a **device tensor**. Measured
(8 rows of **4 MiB**; note the different row size from §2.1, which is why the
absolute figures differ — the ranking does not):

```
torch.index_select                  1.45 ms    23.1 GB/s
per-row copy_ (cudaMemcpyAsync)     0.81 ms    41.4 GB/s
this kernel                         0.64 ms    52.1 GB/s   <- 92% of 56.4 GB/s H2D
```

`BLOCK = 8192` measured fastest; 4096 ties; ≥16384 falls to ~49 GB/s.

### 3.1 Deduplication without a data-dependent shape

`topk_ids` repeats experts — the same expert is selected by several of the tokens
verified in one MTP step. The obvious `torch.unique` is unusable: its output
shape is data-dependent and graph capture rejects it. Instead the grid stays
statically shaped and each program skips its row if that expert already appeared
at a lower row:

```python
r = tl.program_id(0)
e = tl.load(idx + r).to(tl.int64)
js = tl.arange(0, MAXN)
prev = tl.load(idx + js, mask=js < r, other=-1).to(tl.int64)
if tl.sum((prev == e).to(tl.int32)) > 0:
    return          # already staged at a lower row
```

Only unique bytes cross PCIe; the grid never changes shape.

### 3.2 The expert map must resolve to the *first* occurrence

Because duplicates are skipped, only the **first** row of a repeated expert is
filled. A plain scatter (`m[experts] = arange(n)`) will not do: PyTorch leaves
the winner **undefined** when scatter indices repeat, and empirically returns
last-ish occurrences — precisely the rows that were never written.

```python
order = torch.arange(n, dtype=torch.int32, device=experts.device)
m = torch.full((global_num_experts,), n, dtype=torch.int32, device=experts.device)
m.scatter_reduce_(0, experts.to(torch.int64), order, reduce="amin")
return torch.where(m == n, torch.full_like(m, -1), m)
```

`amin` is deterministic. The whole map is built on device, so it stays
capture-safe.

### 3.3 One stager for the whole model, sized for the worst case

MoE layers run sequentially and each is done with its staged experts before the
next begins, so **one shared instance** serves all of them. Per-layer stagers
multiply the footprint by the layer count (45x here) and fail as *"No available
memory for the cache blocks"*.

The staging buffer is allocated once at `MAX_STAGED_EXPERTS` rows, not at the
current step's count: vLLM hands every remaining byte to the KV cache after
profiling, so a buffer that grows later has nowhere to go and dies mid-request.

---

## 4. Hot-expert pinning

> Also new since 2026-09-01. It was §8 item 3 — *"hotness-aware caching"* — with
> an LRU design that measurement had already half-killed. What shipped is not an
> LRU. **13.53 → 16.5 tok/s.**

### 4.1 The mechanism vLLM gives you, and why it wastes VRAM

With `cpu_offload_gb` set, vLLM makes residency a **whole-layer** decision: some
layers are entirely in VRAM, the rest entirely on the host. Consequences:

- the resident share of expert *fetches* equals the resident share of *layers*;
- **which** experts are resident is decided by module iteration order — i.e.
  arbitrarily. 36% of experts resident serves 36% of fetches, and no more.

### 4.2 The skew is per-layer, and the aggregate hides it

> **SUPERSEDED, and this is the most important correction in this repository.**
> An earlier note recorded *"GLM-5.3's expert routing is near-uniform with no
> locality; caching levers are capped at ~1.15x."* That was an artifact of
> summing per-expert counts **across layers**. Different layers are hot on
> different experts, so aggregating that way washes the skew out.

Recomputed from the shipped profile (33 offloaded layers × 288 experts, 12,000
staged steps), the same data two ways:

| capacity | **aggregate** (summed across layers) | **per layer** |
|---|---|---|
| top 10% | 13.8% of activations | **38.9%** |
| top 25% | 32.5% | **64.1%** |
| top 35% | 43.2% | **74.3%** |
| top 50% | 58.8% | **85.7%** |

The live aggregate dump at a different offload agreed with the left column
(*"top-25% of experts capture 34.1% of activations, uniform would be 25.0%"*) —
which is exactly why it was believed.

Measured per layer, **the hottest ~36% of a layer's experts serve 75% of that
layer's fetches.** Hit rate against a uniform slot budget per layer:

```
 33 slots/layer -> 42.65%      132 -> 82.99%
 66 slots/layer -> 61.46%      165 -> 89.62%
 99 slots/layer -> 73.98%
```

### 4.3 What ships

A per-layer VRAM cache of the frequency-ranked hottest experts, plus a
**device-resident slot table** the gather kernel reads to choose its source per
row:

```python
sl = tl.load(slot + e).to(tl.int64)
if sl >= 0:
    v = tl.load(cache + sl * row_bytes + offs, mask=m, other=0)   # VRAM, ~1.8 TB/s
else:
    v = tl.load(src   + e  * row_bytes + offs, mask=m, other=0)   # PCIe, ~52 GB/s
```

The table lives on the device for the same reason the indices do: a host round
trip would break graph capture, and eager mode costs 13% — which would eat most
of what the cache wins. A hit is effectively free; only misses cost link time.

### 4.4 The matched set — offload *more* to go faster

| configuration | tok/s |
|---|---|
| offload 105 GiB, no pinning *(the old default)* | 13.73 |
| offload 125 GiB, no pinning | **11.34** ← worst |
| offload 125 GiB + 68 hot slots/layer | **16.76** |
| offload 125 GiB + 66 hot slots/layer *(shipped when this was measured)* | 16.50 |
| offload 125 GiB + 62 hot slots/layer | 15.8 prose / 17.1 coding — **not a controlled comparison**: those runs also carry breakable graphs off and grouped prefill on, and the 66 → 62 step was never A/B'd on its own |
| offload 125 GiB + **54** hot slots/layer *(shipped now — see §4.5 and §12)* | **14.99**, against 14.92 at 48 slots on the same benchmark. The slots were spent on long-context headroom (§12) |

Offloading more is faster **only** because the VRAM it frees funds the cache.
Take the cache away and 125 is the worst configuration on the list. The launcher
header that said *"above ~100 just costs speed"* was correct before pinning
existed and wrong afterwards.

### 4.5 The slot ladder, and the VRAM wall

| slots/layer | outcome |
|---|---|
| 96 | `CUDA out of memory. Tried to allocate 384.00 MiB` **at startup** |
| 76 | `... 608.00 MiB` at startup |
| 68 | boots, then OOMs **mid-serving** |
| **66** | stable |

66 slots × 33 offloaded layers × ~13.5 MiB ≈ **29 GiB** of VRAM. Dropping from
68 to 66 gives up the two **coldest** experts of the pinned set — ~0.9 GiB for a
fraction of a point of hit rate, with prefill speed untouched. The symmetric
floor is `--cpu-offload-gb 96`, which dies at init with `Tried to allocate
6.75 GiB` during the Marlin repack.

> **Update, later on 2026-09-02: the shipped value is now 54.** The wall is not
> a fixed line, because the hot cache is allocated *after* vLLM's memory
> profiling (§6.6) — so "stable" means "stable for the workload you happened to
> run", and long-prompt traffic moves it. Slots are also the *only* place VRAM
> headroom comes from here (`gpu-memory-utilization` buys nothing, §6.5;
> shrinking the prefill chunk crashes the engine, §6.7), so the shipped count
> has walked **68 → 66 → 62 → 54** as headroom was needed elsewhere. Each step
> down gives up the *coldest* experts of the pinned set. Treat the ladder as a
> method, not as a table of safe values.

> **Correction to this document's own claim about what that costs.** The
> paragraph above used to say only that the cost was "small and hard to see".
> It was then **measured**, and it is smaller than that phrasing implies:
>
> | slots | decode | free VRAM |
> |---|---|---|
> | 48 | 14.92 tok/s | 8,817 MiB |
> | **54** *(shipped)* | **14.99 tok/s** | 6,177 MiB |
>
> 0.07 tok/s apart, well inside the ±1 tok/s run-to-run spread, because 48 → 54
> is only about **3 points of cache hit rate** on the ladder in §4.2. In this
> range the slot count is a **weak lever and the headroom is nearly free** —
> earlier revisions of this file implied the trade was more expensive than it
> is. That is not a statement about the whole ladder: 0 → 66 slots is the entire
> 13.53 → 16.50 tok/s win. It is a statement about its flat top.


### 4.6 Two silent failures in the profile plumbing

The profile is recorded from real traffic (`VLLM_MOE_SKEW_PERLAYER=<path>`,
rewritten every 6000 staged steps) and is **tied to the `cpu_offload_gb` it was
captured at**: layer *N* means "the *N*-th offloaded layer", so a different
offload gives a different set of host-side layers.

- Profile captured at offload 125, used at 127: **15.09 vs 16.76 tok/s**. No
  error. The launcher now compares the `offload_gb` stored inside the profile
  against the environment and warns at launch; the pinner warns too.
- The writer originally keyed layers by `sorted()` order of the `w1` data
  pointer while the pinner read them back by **discovery** order. Both parsed.
  Every layer received another layer's hot set: measured hit rate **45% against a
  predicted 73%**. The writer now keys by discovery order, and the pinner
  refuses any layer whose entry is missing or the wrong length — with a warning,
  falling back to the plain gather for that layer.

### 4.7 Two neighbouring ideas that do **not** pay

- **Temporal locality / LRU.** A 16-slot-per-layer LRU over 288 experts measured
  a **9.0% hit rate** (10188 / 113208 fetches) against 5.6% at random. GLM-5.3
  activates 8 of 288 (2.8%) — far sparser than the Mixtral-class models where
  caching papers show wins. Real, and useless.
- **Water-filling per-layer slot budgets.** Redistributing the same total slots
  toward the more skewed layers gains **+0.07 pp** of hit rate. The layers' hot
  *sets* differ but their distribution *shapes* are alike, so uniform allocation
  is already optimal. A clean negative that saved writing a solver.

---

## 5. A false start, documented in full

The negative results are the point of this document, and this was the most
expensive one.

### 5.1 The wrong assumption

Reasoning that a DMA copy needs a *host* tensor as its source, and that vLLM's
C++ `get_cuda_view_from_cpu_tensor` keeps no Python-visible host handle, the
offloader was patched to (1) allocate an exact-size CPU tensor, (2) pin it in
place with `cudaHostRegister` (avoiding torch's `pin_memory()`, which rounds to
power-of-two — measured 1.48x RSS), and (3) keep the handle on the parameter.

In isolation every piece verified: exact-size pinning (1.00x RSS vs 1.48x),
`is_pinned()` → `True`, the UVA view still worked and aliased our pages, and DMA
from it hit 45.2 GB/s.

### 5.2 What it actually did

A/B on an 8-layer truncated model, 18.56 GiB offloaded:

| | peak RSS | anon | shmem | ratio | engine |
|---|---|---|---|---|---|
| holding the handle | 39.61 GiB | 23.07 | **18.58** | **2.13x** | failed |
| not holding it | 20.04 GiB | 19.50 | 0.02 | **1.08x** | started |

The extra reference kept a **superseded copy of every expert alive**. The Marlin
scale repack runs *after* offloading and rebinds the parameter data; with a
reference pinning the pre-repack copy, both survived. At full scale:

```
Killed process (VLLM::Worker)  anon-rss:108.2 GiB  shmem-rss:45.0 GiB  = 153 GiB
```

Host RAM is 182 GiB, with no swap. The load climbed past 160 GiB and froze the
machine.

### 5.3 The resolution, and the lesson

The `cudaMemcpyAsync`-from-UVA benchmark (§2.1, row 3) showed the host handle
**was never needed at all**. The entire patch — and the OOM it caused — was
reverted.

**Benchmark the primitive before building machinery on top of it.** Five minutes
of `cudaMemcpyAsync` timing would have skipped the whole detour. The seductive
part was that every *component* verified correctly in isolation; the flawed step
was the premise that made them necessary.

A second, subtler lesson: **the timeline is what identified the duplication.**
Correlating a memory sampler against log timestamps showed `shmem` sitting at 0.0
while offload ran and then climbing to exactly the offloaded size *during the
repack window*. Point-in-time snapshots would not have shown it.

---

## 6. Configuration tuning: eight negative results

Baseline for 6.1–6.4: **10.77 tok/s**, MTP-2, 512k context,
`--cpu-offload-gb 105`, `moe_backend=marlin`. 6.5–6.8 are on the optimized
stack at 262k.

### 6.1 KV cache cap — worked mechanically, no speedup *(but ship it anyway)*

vLLM sizes the KV cache from **leftover** budget, not from need. The original run
provisioned **2.1x concurrency** and spent **14.5 GiB at 512k for a single
agent**. Capping it worked exactly as intended — `kv_cache_size_tokens=606452`,
1.16x concurrency, full 512k retained, ~7 GiB freed — and produced **no
measurable speedup**, because it moved residency 33.4% → 35.2%, i.e. 2.7% of the
dominant term.

> **Amended.** That conclusion holds for *blind* residency. On the shipped
> configuration the KV cap is load-bearing: at offload 125 the leftover budget
> would hand KV ~14 GiB (measured uncapped: **27.17 GiB / 1,870,432 tokens =
> 7.1x concurrency**), starving the hot-expert cache that pays for the offload.
> `KV_BYTES=4 GiB` gives **272,771 tokens = 1.04x concurrency at 262k**, which
> is exactly one full-context sequence.
>
> **Amended again (2026-09-02, later): the cap is now derived, not fixed.**
> `KV_BYTES = MAX_LEN × 17,200 B`, i.e. 5,636,096,000 B (~5.25 GiB) at the
> shipped 327,680 context, giving **375,543 tokens = 1.15x concurrency**. The
> constant is the measured 15.0–15.7 KiB/token plus ~10% headroom on the worse
> figure (4 GiB → 272,771 tokens; 5.25 GiB → 375,543). Deriving it removes a
> failure mode rather than buying speed: anything that sets the context without
> also setting the cap — a supervisor script, a GUI with a context slider — used
> to leave KV sized for the *old* context, after which vLLM refuses to start
> with *"N GiB KV cache is needed, larger than the available"*. See §12.
>
> Note the semantics change too: when `--kv-cache-memory-bytes` is set, vLLM
> **skips memory profiling entirely** —
> *"Initial free memory 94.36 GiB, reserved 4.0 GiB memory for KV Cache as
> specified by kv_cache_memory_bytes config and skipped memory profiling."*
> That 94.36 GiB is measured before the weights land (the model then reports
> *"Model loading took 55.28 GiB"*), which is a large part of why the mid-serving
> OOM in §6.6 exists.

### 6.2 MTP depth 4 — actively worse

| | k=2 | k=4 |
|---|---|---|
| decode | **10.77 tok/s** | 8.72 tok/s |
| MTP acceptance | 60.2% | **33.7%** |
| tokens/pass | 2.20 | 2.35 |
| passes/s | 4.90 | 3.71 |

**Acceptance nearly halved.** GLM re-runs the *same* MTP layer for each draft
step, so draft quality degrades sharply with depth — tokens/pass barely moved
while pass rate fell 24%.

The instructive contrast is Qwen3.8-Flash-Next on the same card:

| | forward pass | cost of one extra draft step | k=4 result |
|---|---|---|---|
| Qwen3.8-Flash-Next | 17 ms | 8.5% | wash (173.7 → 173.8) |
| GLM-5.3-Flash | 207 ms | ~2% | **worse** (10.77 → 8.72) |

Two completely different reasons for the same conclusion. **Do not reason about
MTP depth from pass duration alone — measure acceptance per position.**

### 6.3 FlashInfer autotune — nothing on GLM, +3.4% on Qwen

`--no-enable-flashinfer-autotune` appears in several reference configs with no
stated rationale, and it *overrides* vLLM's own `-O2` default, which enables
autotune. Removing the flag:

- **GLM-5.3-Flash: no measurable change** (decode is PCIe-bound; kernel choice is
  irrelevant when the SMs are stalled).
- **Qwen3.8-Flash-Next: +3.4% forward-pass rate**, consistent across code, prose
  and repetitive workloads (57.9 → 59.9 passes/s).

The contrast is itself the lesson: kernel-level tuning pays only when you are not
bandwidth-starved.

### 6.4 `--cpu-offload-gb 99` — OOM during the Marlin repack

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.12 GiB.
  ... in marlin_utils_fp4.permute_scales -> _nvfp4_compute_scale_factor
```

The Marlin scale permutation needs a **transient 1.12 GiB** the steady-state
budget does not account for. On the optimized stack the same wall appears at
`--cpu-offload-gb 96` with a 6.75 GiB allocation. **~105 GiB is the floor** for
the un-pinned configuration; the shipped one runs at 125 for unrelated reasons
(§4.4).

### 6.5 `--gpu-memory-utilization` — buys nothing

0.95 and 0.96 produce byte-identical results — the same
`GPU KV cache size: 272,771 tokens` (measured at the 4 GiB cap and 262k context
of the time) — because `KV_BYTES` is explicit and
weights + hot cache are fixed. There is no slack for the utilisation fraction to
redistribute. Headroom comes from hot slots or offload, nowhere else.

### 6.6 Reserving staging buffers before the cap check — reverted

`stage()` returns early when `n > MAX_STAGED_EXPERTS` (64), and vLLM's memory
profiling runs a full max-size prefill where `n` is far above the cap — so it
returns before the ~29 GiB hot cache is ever allocated. The cache lands later,
during CUDA-graph capture, leaving ~2 GiB of real margin, and serving eventually
dies with `CUDA out of memory. Tried to allocate 124.00 MiB` on a 95 GiB card.

Moving the cap check after the allocations makes vLLM see the truth and fail
**loudly at startup** — correct behaviour — but serving then hits `CUDA error:
an illegal memory access`; the profiling run's allocator state does not survive.
**Reverted.** What was kept is the smaller half: buffers are reserved before the
`if not any(stageable)` early exit.

### 6.7 `--max-num-batched-tokens` below 8192 — illegal memory access

> **WITHDRAWN (2026-09-02, later).** The crashes below were real and reproducible
> and the conclusion drawn from them was still wrong. The same fault hit a run at
> **8192** after 12 requests; smaller values only changed the allocation layout so
> the corruption surfaced on request 2 instead of request 12. The cause is the
> Xid 31 MMU fault in **§11**. 8192 remains the *tested* default; 2048 and 4096
> are **untested since the fix**, not known-bad. Do not cite this section as a
> floor.

> **REINSTATED (2026-09-02, ~07:5x) — the withdrawal above was itself wrong.**
> With `VLLM_USE_BREAKABLE_CUDAGRAPH=0` shipped (so the Xid 31 fault is gone)
> and the staging bounds checks in place, **`MAX_BATCHED=4096` still dies**:
> a 500 on the very first short decode request, engine dead behind it.
>
> So there are **two** failures here, not one. The Xid 31 MMU fault was real and
> is fixed; the sub-8192 floor is real, independent of it, and **not** fixed.
> **Treat 8192 as a hard floor.**
>
> What the withdrawal got right: the *evidence* (8192 did fault, so batch size
> could not be the whole story). What it got wrong: concluding from that single
> counter-example that the batch-size effect was therefore an illusion. One
> confound does not dissolve a reproducible result — it means you have two
> results. Re-testing after removing the confound is the step that was missing,
> and it took ten minutes. See §11.4.

Shrinking the prefill workspace is the obvious way to buy the VRAM headroom §6.6
needs. That route is closed:

| value | result |
|---|---|
| 2048 | `CUDA error: an illegal memory access` on the **second** request |
| 4096 | same |
| 8192 | 8 sustained 200-token generations at **16.50 tok/s** |

~~Stack is in `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`. It is
*not* a collision with `index_topk=2048` — 4096 fails too. Not root-caused;~~
**treat 8192 as a floor** — that last clause is back, on fresh evidence.
(The pre-optimization 2026-08-29 stack ran at 4096 without this failure, which
still says the failure arrived with the staging work rather than being inherent
to a batch size. It just is not the *same* failure as the Xid.)

**The consequence is the expensive part.** Long-prompt activation scales with
chunk × length, so shrinking the prefill chunk is the obvious, cheap way to buy
VRAM headroom for long context — cheaper than giving up hot-expert slots, which
is what pays for decode throughput. That lever is closed. §12 is what "closed"
cost.

### 6.8 Summary

| change | effect |
|---|---|
| KV cap 14.5 → 8 GiB at 512k, un-pinned | works, +1.8 pp residency, **no speedup** |
| KV cap → 4 GiB at 262k, pinned | **required** — funds the hot cache |
| KV cap derived as `MAX_LEN × 17,200 B` (5.25 GiB at 327,680) | **required** — stops context and KV size from disagreeing (§12) |
| offload 112 → 105 GiB | fine; **99 OOMs** in the repack, 96 OOMs harder |
| autotune enabled | **no change on GLM** (+3.4% on Qwen) |
| MTP depth 4 | **worse** (−19%) |
| `--block-size 256` | required only for depth ≥ 4; irrelevant at k=2 |
| `gpu-memory-utilization` 0.95 → 0.96 | **nothing**, byte-identical |
| `max-num-batched-tokens` < 8192 | **crashes the engine** — withdrawn as an artifact, then **re-confirmed on a clean build**; 4096 dies on the first request (§6.7, §11.4) |
| hot slots 54 instead of 62–66 | **−0 to −1 tok/s (inside the spread)**, and it is what makes a 304,800-token prompt possible (§12) |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | **required for stability**, costs nothing (15.77 tok/s in a 15.7–16.5 band) — §11 |
| `VLLM_MOE_PREFILL_GROUPS=1` | **2.9x prefill**, decode unchanged — §10 |
| hot-expert profile re-captured on mixed T=0/0.7/1.0 traffic at offload 130 | **worse on everything** (13.02 prose, 16.10 coding, 1698 prefill) — §10.4 |

---

## 7. Corrections to the original runbook

### 7.1 DeepGEMM now supports SM120

The runbook states that MTP routes through DeepGEMM's paged-MQA attention which
"asserts `arch_major == 10`", fixed with `VLLM_USE_DEEP_GEMM=0`. **That is no
longer accurate.** The current assert (`csrc/apis/attention.hpp:320`) is:

```cpp
DG_HOST_ASSERT(
    (arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or
    (arch_major ==  9 and (block_kv == 32 or block_kv == 64)) or
    (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64)) or
                           (not is_fp4 and block_kv == 64))));
```

SM120 (`arch_major == 12`) **is** accepted. The real constraint is `block_kv`.
`VLLM_USE_DEEP_GEMM=0` is still the shipped default because it costs nothing
measurable on the plain path and keeps `SPEC=1` usable without reasoning about
page sizes.

### 7.2 The `block_kv` constraint, and why MTP depth 4 tripped it

On SM120 the branch depends on the data type: `is_fp4` allows `block_kv` 32 or
64; **not `is_fp4` requires exactly 64**. The GLM-5.3 indexer cache is **fp8**,
so only 64 is legal.

But vLLM's page-size selection (`vllm/models/glm5next/nvidia/attention.py`) picks
from `PAGED_MQA_PAGE_SIZES = (32, 64)` without knowing that:

```python
storage_block_size = spec.block_size // index_kpool     # index_kpool = 4
max_page_size, min_page_size = 64, 32
if   storage_block_size <= max_page_size:      page_size = storage_block_size
elif storage_block_size % max_page_size == 0:  page_size = max_page_size
else:                                          page_size = min_page_size   # 32 -> illegal
```

Since `page_size = block_size / index_kpool` and `index_kpool = 4`:

> **`block_size` must be a multiple of 256** for `page_size` to be 64.

Left unset, vLLM derives `block_size` from `math.lcm` of the cache-group specs,
and **speculative-decode depth shifts that LCM**. At `k=2` it lands on a multiple
of 256; at `k=4` it does not, `page_size` falls through to 32, and you get the
opaque C++ assert with no hint about MTP being the cause. The knob is
`--block-size 256`.

This is a latent bug worth fixing upstream: the fallback to `min_page_size`
should be device- and dtype-aware, or the assert should name `block_size`.

---

## 8. Reproducibility tooling

A full load is 181 GiB and ~5–6 minutes, which makes config sweeps impractical.

### 8.1 Truncate the model to 8 layers

Combined with `--load-format dummy` (which never reads a byte of the checkpoint),
an 8-layer config exercises the entire construction, offload, repack and
graph-capture path at ~1/6 scale in ~90 seconds.

**The trap:** truncating `num_hidden_layers` alone fails with

```
ValueError: `num_hidden_layers` (8) must be equal to the number of `mlp_layer_types` (45)
```

Every per-layer list must be truncated too. This model has three
(`layer_types`, `mlp_layer_types`, `indexer_types`), and a future revision may
add more, so truncate generically by matching list length against the layer
count:

```python
import json

N = 8
d = json.load(open("config.json"))

def truncate(o, path=""):
    if not isinstance(o, dict):
        return
    n_layers = o.get("num_hidden_layers")
    for k, v in list(o.items()):
        if isinstance(v, dict):
            truncate(v, path + k + ".")
        elif isinstance(v, list) and n_layers and len(v) == n_layers:
            o[k] = v[:N]
    if n_layers and n_layers > N:
        o["num_hidden_layers"] = N

truncate(d)
json.dump(d, open("config.json", "w"), indent=2)
```

Copy only `config.json` plus the tokenizer files into the truncated directory.

### 8.2 A/B harness

Wrap "tear down → start memory sampler → launch → wait → report" in a **script
file**, not an inline command (see the `pgrep -f` trap in
[LAB-NOTEBOOK](LAB-NOTEBOOK.md#operational-traps)). Report peak host RSS split
into anon and shmem, normalised by the offloaded bytes:

```
dmaON    staging=1  offloaded=18.56GiB  peakRSS=39.61 (anon 23.07 + shmem 18.58)  ratio=2.13x  api=DOWN
dmaOFF   staging=0  offloaded=18.56GiB  peakRSS=20.04 (anon 19.50 + shmem 0.02)   ratio=1.08x  api=UP
```

Sampling `RssAnon` and `RssShmem` separately from `/proc/<pid>/status` — rather
than just `VmRSS` — is what made §5.2 diagnosable.

### 8.3 Measure forward-passes/s, not tok/s

**MTP acceptance varies ±5 pp run to run even at `temperature=0`**, because the
MoE kernels are not bit-deterministic (atomics in reductions), so generations
diverge and draft acceptance with them.

```
throughput (tok/s) = forward passes/s x tokens accepted per pass
```

Derive passes/s from the `/metrics` delta across the benchmark window, not
lifetime totals:

```
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total

tokens_per_pass = 1 + (accepted_delta / drafts_delta)
passes_per_sec  = decode_tok_s / tokens_per_pass
```

Also: **isolate prefill from decode with streaming.** Timing a non-streamed
request conflates them; at 156k context prefill is 14 s against a 2 s decode, and
we briefly "measured" decode falling from 161 to 10.5 tok/s with context length
before realising it was entirely a prefill artifact. Streamed, decode is flat
across context.

Two further traps: **streaming chunks are not tokens** (with MTP a chunk carries
~2–4 — count `usage.completion_tokens`), and **choose a realistic workload**
(*"count from 1 to 150"* gives ~100% acceptance and flatters throughput by ~50%;
prose ~55%, code ~66%).

### 8.4 Instrumentation built into the stack

| env var | what it does |
|---|---|
| `VLLM_MOE_DMA_STAGING=0` | fall back to plain zero-copy reads (the pre-optimization path) |
| `VLLM_MOE_TIME_GATHER=1` | CUDA-event timing of the gather itself, reported every 2000 calls as in-situ GB/s. Events record on the stream and are read back in batches, so this does not serialize the pipeline the way `torch.cuda.synchronize()` would |
| `VLLM_MOE_SKEW_DUMP=<path>` | aggregate per-expert histogram + concentration curve (**note §4.2 — this is the view that misleads**) |
| `VLLM_MOE_SKEW_PERLAYER=<path>` | per-layer counts; this is the profile the pinner consumes |
| `VLLM_MOE_HOT_PROFILE`, `VLLM_MOE_HOT_SLOTS` | enable pinning |
| `VLLM_MOE_ROUTER_STATS=<path>` | router weight by rank (needs eager — under graphs the Python does not run per step) |
| `VLLM_MOE_DECODE_TOPK=<k>` | opt-in router pruning. **Degrades quality; not shipped on.** |
| `VLLM_MOE_PREFILL_GROUPS=1` | grouped DMA staging during prefill (§10). **Shipped on**; set 0 for the zero-copy path |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | vLLM's own knob, not ours. **Shipped off**; leaving it on reproduces the Xid 31 in §11 |
| `SPEC_METHOD` / `SPEC_EXTRA` (launcher) | swap the draft method (`mtp` \| `ngram` \| `ngram_gpu` \| `suffix` \| …) and append raw JSON to the speculative config. Both blocked past depth 2 — §11.3 |
| `EXTRA_ARGS` (launcher) | unquoted flag passthrough, e.g. the profiler flags above. Word-split on purpose |

**Caveat on any profile captured with `VLLM_MOE_PREFILL_GROUPS=1`:** a group walk
touches every expert whether routing wanted it or not, so the histogram carries a
**uniform +1 per expert per chunk**. Rankings survive (the constant is uniform),
but raw top-k shares read about **17% low**. Subtract the per-layer minimum
before estimating a hit rate from such a profile.

---

## 9. Where it stands and what is left

**Done and measured**, against the 10.77 tok/s baseline:

| change | result |
|---|---|
| Triton dedup gather, device-side indices, CUDA graphs restored | **13.53** |
| frequency-ranked hot-expert cache, offload 125 + 66 slots | **16.50** |
| grouped prefill staging, gated at ≥ 48 prompt tokens (§10) | prefill **614 → 1729–1771 tok/s**; decode unchanged |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + staging bounds checks (§11) | 0 failures / 0 Xid in two 40-request soaks, at unchanged speed |
| context 262,144 → 327,680 with `KV_BYTES` derived from it, funded by 62 → 54 hot slots (§12) | a **304,800-token prompt prefills** (205 s, 1,487 tok/s); decode cost is inside the run-to-run spread |

**Closed by measurement** — see
[FURTHER-OPTIMIZATION-RESEARCH.md](FURTHER-OPTIMIZATION-RESEARCH.md) for the
full accounting: native-FP4 kernels (Marlin is fastest at every batch size),
CPU co-execution (0.895x here), GPUDirect Storage (converges on the same x16
ingress), LRU caching (9% hit), per-layer water-filling (+0.07 pp), MTP depth,
autotune, `gpu-memory-utilization`.

**Left, in order of expected value:**

1. **FP8 for the 18.01 GiB of BF16 non-expert weights** — the profiler's second
   term, ~5–6 ms/token. Costs ~31 dB SNR (measured on our own tensors: they are
   genuine BF16 masters, not upcast FP8), and **no pre-quantized checkpoint
   fits** — a full-FP8 GLM-5.3-Flash is ~331 GB against ~278 GB of total memory
   here. This is a quantize-it-yourself job with a real quality cost.
2. ~~**Root-cause the sparse-MLA illegal memory access** that kills the engine
   below `max-num-batched-tokens` 8192 and, occasionally, above it.~~
   **Half settled (§11):** the *"occasionally, above it"* half was an Xid 31 MMU
   fault, and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus bounds checks in the staging
   path stop it in two 40-request soaks (mechanism still unproven). The *"below
   8192"* half came back: on the fixed build, 4096 still kills the engine on the
   first request (§6.7, §11.4). Root-causing **that** is now the higher-value
   job of the two, because it is the cheap source of the VRAM headroom that long
   context needs (§12).
3. **Re-examine the masked-group prefill variant.** It stages only the experts a
   chunk actually routes to, instead of all 288, and it measured **0.36 s TTFT
   on the 14-token control** against **0.85 s** for the zero-copy path that
   ships (and 2.05 s for ungated grouping). It was withdrawn for "run-to-run
   nondeterminism" that §10.6 later showed the *baseline* has as well — a
   misattribution, so the variant was never proven either way. Cheap to redo,
   and if it holds up under the fresh-prompt logprob test it retires the
   ≥ 48-token gate. **Untested, not shipped.**
4. **Overlap transfer with compute.** Architecturally blocked *within* a layer
   (the router depends on that layer's attention output), but double-buffering
   across the two halves of a layer's expert set is untried.
5. **Fix the fused-`w13` scale collapse** (runbook §8). A *correctness* issue,
   not performance — there is little point optimising throughput on mis-scaled
   experts.

**And the honest framing:** the quality-neutral *decode* ceiling on this box is
~16–17 tok/s, the absolute ceiling with zero PCIe traffic is ~29, and a model
that simply fits runs at 173. Every technique here manages the ratio between
181 GiB of weights and 96 GiB of VRAM; none of them removes it.

**And a framing correction.** For the workload this deployment actually serves —
long agentic coding turns that resend their whole context — decode tok/s was
never the binding number. A 10k-token turn used to spend 16.8 s before its first
token and ~11 s generating 200 after it; §10 took that turn from ~27 s to ~17 s
without moving decode at all. If you are tuning one of these, measure the thing
your user waits for before deciding what to optimise.

---

## 10. Prefill — the term the decode benchmarks hid

Sections 1–9 measure decode, and every number in them is still true. They are
also the wrong half of the problem for the workload this stack serves. Streamed,
on the 16.5 tok/s configuration:

```
10,296-token prompt      TTFT 16.8 s   =   614 tok/s prefill
decode after that                          18.8 tok/s
262k context, extrapolated                 ~7 minutes before the first token
```

Decode at long context is fine. **Prefill is 34x slower than decode per token of
work**, and a client that resends its conversation every turn pays it every
turn.

### 10.1 Why prefill was slow: the optimization's own guard rail

`stage()` returns `None` when `n > MAX_STAGED_EXPERTS` (64). That cap is what
keeps a pathological step from claiming the card, and for decode it never binds:
a decode step routes to 8 experts per layer (top-8 of 288), so it always stages
and always gets the 52 GB/s DMA gather of §3.

A prefill chunk routes to **essentially all 288 experts of every layer**. So
`n = 288`, `stage()` declines, and Marlin falls back to reading the offloaded
experts zero-copy through the UVA mapping: BAR reads at ~23 GB/s, re-read per
Marlin tile, roughly **98 GiB per chunk** after hot-cache hits.

This is §2's transfer-mode finding reappearing in the one code path that was
never measured against it. The property that makes decode cheap — sparsity — is
also what made the fast path skip prefill entirely.

### 10.2 The fix: walk the experts in groups

`VLLM_MOE_PREFILL_GROUPS=1` turns the bail-out into a loop. Per layer:

1. take the experts in 64-wide groups (`MAX_STAGED_EXPERTS` at a time);
2. `stage()` the group — hot-cache hits are served from VRAM as usual;
3. `moe_align_block_size` with an `expert_map` that hides every expert outside
   the group (`ignore_invalid_experts=True` drops the `-1` ids);
4. run the inner `_fused_marlin_moe` for the group into the **same** per-pair
   workspace;
5. after the last group, one fp32 `moe_sum` over the `[M, topk, K]` view.

The reason this is exact rather than approximately-exact is worth stating
precisely, because the obvious implementation is not:

> Every `(token, expert)` pair belongs to **exactly one** group, and the second
> Marlin GEMM writes only the sorted **valid** rows of the per-pair buffer. So
> filling that buffer one group at a time and reducing once at the end touches
> the very same rows, in the same order, that a single pass would.

**What not to do.** The first draft ran each group to completion and summed five
**bf16** partial outputs. That flipped tokens at T=0 — five bf16 additions of
partial sums is a different computation from one fp32 reduction of disjoint
rows, and on a knife-edge argmax the difference shows. (`moe_sum` also requires
the 3-D view; the flat one fails in its C++ op with `aoti_torch_get_stride`.)

### 10.3 Measured, and the gate that shipped with it

Re-measured on the shipped server (pid 211628, bare `./serve-opt.sh` defaults),
streaming, TTFT = time to the first content chunk, on salts **never sent to that
server before** so no prefix-cache reuse is possible. Logged verbatim in
`artifacts/prefill-fresh-2026-09-02.txt`:

| prompt | prompt tokens | TTFT | prefill | decode after |
|---|---|---|---|---|
| 10k fresh #1 | 10,313 | **5.82 s** | **1771 tok/s** | 18.2 tok/s |
| 10k fresh #2 | 10,314 | **5.96 s** | **1729 tok/s** | 18.2 tok/s |
| 30k fresh | 35,571 | **22.57 s** | **1576 tok/s** | 18.4 tok/s |
| 14-token control *(below the gate — zero-copy path)* | 14 | **0.85 s** | — | — |

against **614 tok/s / 16.8 s** on the same class of prompt before grouping.

**~2.9x, quality-neutral, decode untouched.**

*(An earlier unlogged pass reported 1762 tok/s / 5.8 s, and that figure is still
quoted in `serve-opt.sh` and in a few places in these documents. The logged
range supersedes it and agrees with it to within two runs' spread. Prefer the
range: a single number from a single unlogged run is how "1762" became a fact
nobody could check.)*

The gate — `topk_ids.shape[0] >= 48` — exists because grouping is a loss on
short prompts, and finding that out cost an hour of misattribution:

- A 14-token prompt routes to only ~100 experts. Staging all 288 in groups made
  its TTFT **2.05 s**, where the zero-copy path serves the same prompt in
  **0.85 s** (the control row above).
- The 200-token decode benchmark then read **12.5 tok/s instead of 15.8** — and
  was read as a decode regression. It was not. **The decode rate was unchanged;
  the entire delta was TTFT.** One streamed request with TTFT separated from the
  inter-token rate settles this in seconds.
- A 40-request soak taken in that window shows both signs of the same effect at
  once, and the arithmetic checks out: a 150-token math generation at 16.67
  tok/s takes 9.0 s, so adding 2.05 s of TTFT predicts **13.6 tok/s** end-to-end
  against **13.91 measured**, while the 9,107-token class went **7.83 → 11.34**.

`shape[0]` is static per captured CUDA graph, so the gate is capture-safe. The
crossover is around 40 tokens; 48 is the shipped value.

### 10.4 Two things this changes about measurement

**Prefix caching partly works, and is invisible.** The earlier finding — an
identical prompt re-sent took 16.9 s against 16.6 — was too strong. An exact
repeat now prefills in ~3.2 s, and on the pre-grouping build the same test read
9.6 s against 16.8. But `vllm:prefix_cache_hits` and
`vllm:prefix_cache_queries` were **0 for every one of these requests**, so
nothing but wall-clock timing distinguishes a reused prefix from a cold one
here. Measure it on your own traffic; do not trust the counters.

**A hot-expert profile captured with grouping on is biased.** A group walk
touches every expert regardless of routing, so the per-layer histogram gains a
**uniform +1 per expert per chunk**. Ranking is preserved — that is what the
pinner consumes — but raw top-k *shares* read about **17% low**. Subtract the
per-layer minimum before estimating hit rates from such a profile.

### 10.5 A negative result on the same axis: re-profiling on mixed traffic

The shipped hot-expert profile was captured on **one prose prompt at T=0** while
the real traffic is coding, which looked like free throughput. It is not.

Offline, a profile captured over the mixed soak workload (prose/code/chat/long/
math at T=0/0.7/1.0, a coding prompt and a 30k prompt) at offload 130 shares only
**44% of its per-layer top-64 set** with the shipped profile, and its decode-only
hit estimate at 66 slots is ~50% against 61.5%.

Measured, offload 130 + `hot-profile-130.json` + 64 slots (VRAM 89.2 GiB):

| | mixed profile @ 130 | shipped @ 125 |
|---|---|---|
| prose decode | 13.02 tok/s | **15.77** |
| coding prompt | 16.10 tok/s | **17.10** |
| prefill | 1698 tok/s | **1762** (the figure current at the time) |
| VRAM margin | +4 GiB | — |

Worse on everything but the margin. The most likely reason is in the capture,
not the idea: two thirds of that workload runs at T=0.7 or T=1.0, and sampling
randomizes tokens, which **flattens routing** — so the profile blurs exactly the
skew it exists to record. Five more GiB of host-side experts accounts for the
rest. Reverted. **If retried: capture at T=0, on the real workload only.**

### 10.6 Correctness: the claim that had to be withdrawn

The launcher comment and the source comment for this path both used to say the
grouped result is *"bit-identical to the single pass (T=0 outputs identical on
3k and 10k prompts)"*. **That evidence is worthless and the claim is withdrawn**
— for a reason that generalises well past this stack. (Both comments were
corrected in the engine tree on 2026-09-02: they now state the fresh-prompt
logprob equivalence and the reason greedy-text comparison cannot establish it.)

**Baseline prefill here is non-deterministic run to run.** The unmodified build,
same 3k and 9k prompts, T=0, `max_tokens=40`, hashed **A, B, A** across three
runs. The grouped path produced three distinct hashes as well, one equal to a
baseline hash. The pattern matches prefix-cache state reuse — mamba `align` mode
caches state at scheduler-step boundaries, a continuation from cached state
rounds differently in bf16, a knife-edge argmax flips — not a race.

So a greedy-text comparison on this stack cannot distinguish "identical" from
"lucky". It produced a false positive (two lucky draws, believed) and then a
false negative (a masked-group variant removed for "nondeterminism" the baseline
has too — see §9, item 3: that variant is a live candidate, not a closed
branch).

**The valid test** compares per-token `prompt_logprobs` on **fresh** prompts —
new salt, nothing cached — and scores the candidate against the baseline's own
fresh-vs-cached drift as the control:

| comparison | tokens | p50 | p90 | p99 | max | \|Δ\| > 1 nat |
|---|---|---|---|---|---|---|
| baseline vs itself (fresh vs cached), 3k | 3,219 | 0.000006 | 0.00054 | 0.057 | 3.75 | 6 (0.19%) |
| baseline vs itself (fresh vs cached), 9k | 10,300 | 0.000011 | 0.00013 | 0.0065 | 4.41 | 5 (0.05%) |
| **grouped vs baseline**, 3k | 3,219 | 0.000008 | 0.00069 | 0.075 | 4.26 | 4 (0.12%) |
| **grouped vs baseline**, 9k | 10,300 | 0.000012 | 0.00022 | 0.014 | 5.01 | 7 (0.07%) |

Every statistic of the candidate rows, tail included, sits inside the control's.
The defensible statement is therefore: **grouped prefill is numerically
indistinguishable from the baseline's own run-to-run variability.** Not
bit-identical — and no measurement taken here could have established that.

---

## 11. Stability — the Xid 31 MMU fault

### 11.1 What the failure actually was

The recurring mid-serving death looked like this on the Python side:

```
CUDA error: an illegal memory access was encountered
  ... v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
```

and like this in the kernel log:

```
Xid 31 ... MMU Fault
```

An MMU fault is a dereference of an unmapped address. Three claims this document
made about it were wrong:

1. **That `--max-num-batched-tokens` below 8192 caused it** (§6.7). The same
   fault hit a run at 8192 after 12 requests. Smaller batches only changed the
   allocation layout, so corruption surfaced on request 2 rather than 12.
   *(This correction was itself half wrong — a **separate** sub-8192 failure
   does exist. See §11.4.)*
2. **That the sparse-MLA backend was implicated by the stack.** CUDA's own error
   text says these are *reported asynchronously at some later call*. The stack
   names the kernel that noticed the fault, not the one that caused it.
3. **That it was inherent to the model or the card.** The previous day ran 22
   launches and ~80 requests with zero such faults. Whatever introduced it
   arrived with that day's work.

### 11.2 What fixed it

**Bounds checks on every id-indexed access in the staging path** — both Triton
gather kernels, `build_local_expert_map`'s `scatter_reduce_`, and the Marlin
call site:

```python
e = tl.load(idx + r).to(tl.int64)
if (e < 0) | (e >= n_experts):
    return          # never fetched; the expert map never points at this row
```

This is the fix that matches the symptom mechanically. A padded or sentinel id
(`-1`, or `n_experts`) either reads past the host tensor or — worse — reads
garbage from past the slot table and then *dereferences that garbage* into the
cache: an unmapped address, surfacing in whatever kernel happens to run next.

**`VLLM_USE_BREAKABLE_CUDAGRAPH=0`.** vLLM auto-enables "breakable" CUDA graphs
for this hybrid model. Every build that ran under them on 2026-09-02 faulted
between request 2 and request 24. With them off:

| soak | requests | result |
|---|---|---|
| bounds checks + breakable=0, no grouping | 40 varied (T=0/0.7/1.0, four 9k prefills, one 30k prompt) | **0 failures, 0 Xid**, 15.77 tok/s (band 15.7–16.5) |
| shipped build (+ grouped prefill) | 40 varied + a 35,553-token prompt | **0 failures**, 592 s |

**The mechanism is not proven.** The supporting observation is that the profiler
showed kernels overlapping across streams while the staging buffers are
*shared*, which is consistent with a race that breakable mode makes possible.
Two clean soaks against same-day faults on every breakable build is enough to
ship on; it is not a root cause, and this section should be read as "measured
fix, unknown mechanism".

`--max-num-batched-tokens 8192` stays the default — and, after the re-test in
§11.4, it is a **hard floor** again rather than merely the tested value.

### 11.3 Two adjacent branches that closed

**The native SM120 attention backend cannot be used.** vLLM lists both
`FLASHINFER_MLA_SPARSE_SM90` and `FLASHINFER_MLA_SPARSE_SM120` as candidates for
this model on this card and picks the SM90 one — i.e. the Hopper kernel running
on Blackwell via the fallback from the runbook's bugs 5–6. Forcing the native
backend was the *other* candidate fix for the Xid, and it fails at startup:

```
SM120 sparse MLA v32/GLM expects kv_lora_rank=512, qk_rope_head_dim=64,
and query head dim 576
```

GLM-5.3 is NoPE MLA (`qk_rope_head_dim=0`, latent 512, query head dim 512+0), so
the native kernel would need the same shape-assumption surgery the SM90 path
already received. Closed until someone ports it.

**Speculation is finished at MTP-2.** Measured on the shipped config, coding
prompt, T=0: **17.10 tok/s**, cumulative acceptance **64% of draft tokens = 1.28
accepted per step** (position 0: 75%, position 1: 53%), so 2.28 tokens per
forward pass. (A second coding run at offload 130 measured 1.81 accepted per
step on its own generation — acceptance is generation-specific; always quote the
prompt.)

`ngram`, `ngram_gpu` and `suffix` exist in this fork, and the launcher exposes
`SPEC_METHOD` / `SPEC_EXTRA`. Both routes to a deeper draft are blocked:

- `num_speculative_tokens=5` needs **4.27 GiB of KV per request against the
  4 GiB `KV_BYTES` cap** in force at the time — startup `ValueError`.
  *(Open item: the cap is now derived from the context and is 5.25 GiB at
  327,680 (§12.2), which is larger than 4.27. Whether depth 5 now starts has
  **not** been retested, and the acceptance argument below still says it would
  not be worth much.)*
- `num_speculative_tokens=4` (5 query tokens per step) dies in the
  sparse-attention KPOOL indexer: DeepGEMM `csrc/apis/attention.hpp:320`,
  `DG_HOST_ASSERT` on `block_kv` for arch 12, via `sparse_attn_indexer_kpool →
  fp8_fp4_paged_mqa_logits`. MTP-2's 3 query tokens pass. This is the same
  assert §7.2 dissects, where `--block-size 256` is the knob; **whether that
  clears it at this depth on the current stack was not retested.**

And the ceiling is arithmetic regardless: at depth 2 a free drafter can only save
the single MTP layer, ~2–3% of a step, and prompt-lookup acceptance on generic
code sits below a trained MTP head's. **MTP-2 stays.**

### 11.4 The floor that was not an artifact after all

§11.1 lists three claims this document made about the crash and got wrong. One of
those three was itself wrong, and the correction is more useful than the original
mistake.

The claim was *"`--max-num-batched-tokens` below 8192 causes an illegal memory
access"*. It was withdrawn when the same fault hit a run at 8192 — a genuine
counter-example, honestly applied. But the conclusion drawn from it, *"therefore
the batch size never mattered"*, does not follow from one counter-example. It
was never re-tested on a build where the confound was gone.

Re-tested at ~07:5x with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` shipped and the bounds
checks in:

```
MAX_BATCHED=4096   ->  500 on the FIRST short decode request, engine dead
MAX_BATCHED=8192   ->  the shipped configuration; two clean 40-request soaks
```

There were **two independent failures**. The Xid 31 MMU fault is fixed. The
sub-8192 floor is real, unfixed, and unexplained, and it now fails *immediately*
rather than on request 2 — which is itself informative, because the noise that
was masking it is gone.

Three things to take from the sequence:

1. **A confound invalidates an attribution, not a phenomenon.** "The crash also
   happens at 8192" refutes "8192 is safe". It says nothing about whether 4096
   has a problem of its own.
2. **Re-run the original experiment after you fix the confound.** Ten minutes.
   The withdrawal stood for six hours and was quoted in five documents.
3. **Mark reversals, do not delete them.** This section exists because the
   struck-through text in §6.7 was still there to re-test against.

---

## 12. Long context — the ceiling is free VRAM, not `--max-model-len`

The server advertised `--max-model-len 262144` for most of this work. No prompt
anywhere near that length had ever been sent. When one finally was, it failed —
and raising the advertised number did not fix it, because the advertised number
was never the constraint.

### 12.1 What actually limits it

**Long-prompt activations are allocated per request and scale with sequence
length.** The attention workspaces and the sparse indexer are not reserved up
front the way weights and the KV cache are, so what decides whether a 200k-token
prompt survives is **how much VRAM happens to be free when it arrives**.

Measured, varying only `VLLM_MOE_HOT_SLOTS` — which on this stack is the only
thing that frees VRAM at all (§6.5 rules out `gpu-memory-utilization`, §6.7 rules
out shrinking the prefill chunk):

| hot slots | free VRAM | longest prompt prefilled | TTFT |
|---|---|---|---|
| 60 | 3,397 MiB | **180k died** — `CUDA out of memory. Tried to allocate 116.00 MiB` | — |
| **54** *(shipped)* | 6,177 MiB | **304,800 tokens** = 93% of the advertised 327,680 | 205 s (1,487 tok/s) |
| 48 | 8,817 MiB | 233,670 tokens *(largest tried at this setting, not a limit)* | 158 s |

Note the shape of the failure: **116 MiB**. It is not that a 180k prompt needs
gigabytes more than a 100k one; it is that the margin had been squeezed to
almost nothing by everything else, and long prompts are what finally notices.

**The rule: never advertise a context you have not prefilled at.** Both earlier
settings — 262,144, and the first attempt at 327,680 — were advertised and never
tested near their limit. An untested `--max-model-len` is a promise the server
makes on your behalf and breaks in front of a user.

### 12.2 Why the KV cache is now derived from the context

```bash
KV_BYTES="${KV_BYTES:-$(( ${MAX_LEN:-327680} * 17200 ))}"
```

The two used to be independent knobs. 17,200 B/token is the measured KV cost
(15.0–15.7 KiB/token: 4 GiB gave 272,771 tokens, 5.25 GiB gave 375,543) plus
about 10% headroom on the worse figure.

The reason to derive rather than to document is that **anything driving the
launcher will set one and not the other**. A supervisor with a context slider
passes `MAX_LEN`; it has no reason to know about `KV_BYTES`. Left independent,
moving the slider up leaves KV sized for the old context and vLLM refuses to
start with *"N GiB KV cache is needed, larger than the available"* — a startup
failure whose message points at the KV cache, when the thing that changed was the
context. Deriving one from the other makes that state unrepresentable.

### 12.3 What it cost

62 → 54 hot slots, and **that cost is inside the measurement noise**: 14.99
tok/s at 54 against 14.92 at 48, on a ±1 tok/s spread, because the top of the
hit-rate ladder is flat (§4.5). The cheap lever that *should* have paid for this
— a smaller prefill chunk, since activation scales with chunk × length — is
closed by §6.7.

---

## 13. Serving it to agents

Once this server became a coding backend rather than a benchmark, two things
mattered more than any tok/s in this document.

### 13.1 "The model forgets its context" was the engine dying

Symptom: agent sessions loop, re-running their own first step — *"Let me start by
exploring the codebase structure"* — as if the conversation had been erased. The
natural suspects are the client truncating history or the chat template dropping
turns.

Neither. **Every `chat/completions` 500 in the logs coincides exactly with an
engine death:**

```
01:01:47   CUDA out of memory
01:39:16   CUDA out of memory
02:32:55   Xid 31 illegal memory access
```

83 of 86 agent requests returned 200; the three failures were those three deaths.
A dead turn leaves a failed or empty assistant message in the client's history,
and GLM's template renders that as a **fresh thread** — so the model really does
re-plan from zero. The amnesia is a faithful rendering of a crash. Fixed by the
same two changes as §11.

**The diagnostic order is the transferable part: verify the server first.** One
direct request to the endpoint, outside the agent, settles four questions at
once — is `finish_reason` correct (`tool_calls`), is `content` empty because
reasoning leaked into it (no: reasoning came back in its own field), are the
`tool_calls` well-formed (yes), does the reasoning parser separate cleanly (yes).
Server exonerated in a single request; every minute after that went to the logs
rather than to the client's configuration. The client is the *easier* thing to
suspect and the harder thing to disprove.

Two genuine client-side misconfigurations did turn up alongside it, and both had
been invisible because the crash was louder:

- A model config advertising **`maxInputTokens: 500000`** against a server
  serving 262,144. Set it to context − max output − margin (227,328 at the time:
  262,144 − 32,768 − 2,048), and **recompute it when the context changes** — it
  is 327,680 now (§12).
- A local proxy that bounds reasoning effort **streamed its retry**. When a turn
  came back answerless it retried — but streamed the retry as well, so an
  answerless retry arrived at the client as a *second* blank turn, doubling the
  symptom it was meant to fix. It now retries **non-streaming**, reuses the same
  answerless guard, and emits an explicit bracketed notice if the retry is still
  answerless. Silence is the one response an agent loop cannot recover from.

### 13.2 `reasoning_effort`, measured

GLM-5.3's chat template accepts exactly three values — **`low`, `high`, `max`** —
and it is a **soft system line, not a token budget**. The failure mode is the
default: anything else — unset, a typo, the entirely plausible `medium` —
**silently becomes `max`**, which is the most expensive setting. Nothing warns.

Measured on one hard problem:

| effort | reasoning tokens | notes |
|---|---|---|
| `low` | 183 | |
| `high` | 484 | |
| `max` | 1,812 | ~2x the wall time of `high`, and a **shorter** final answer |

Two conclusions:

- **The thinking budget is not what makes turns blank.** Even `max` spends 1,812
  reasoning tokens — 7.2% of a 32,768-token output allowance. Blank turns come
  from a small *per-request* `max_tokens`: at 512, `max` returns an empty turn
  (the budget goes entirely to thinking) while `high` still answers. If your
  agent gets blanks, look at the per-request cap before the server config.
- **Use `high` for agent and tool loops.** Many cheap turns beat a few expensive
  ones when each turn is one tool call. Reserve `max` for one-shot hard
  questions *with a large output budget*, and expect it to cost about twice the
  wall time with no guaranteed gain — on this measurement it produced less
  answer, not more.
