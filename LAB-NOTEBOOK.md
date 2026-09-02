# Lab notebook: 10.77 → 16.5 tok/s decode, 614 → ~1750 tok/s prefill

The chronological account of how GLM-5.3-Flash got from "runs, slowly" to
16.5 tok/s of decode and ~1750 tok/s of prefill on one RTX PRO 6000, including
everything that did not work.

The other documents in this repository are organised by *topic*, which makes
them useful as reference and useless as an account of how anyone would actually
arrive at the answer. This one is organised by *time*, because the shape of the
work — four of the five biggest wins came from discarding a conclusion we had
already written down, and two of those conclusions are still in this file with
the strike-through on them — is the transferable part.

**How to read it.** Every number here carries the configuration it was measured
under. Where a claim was later shown to be wrong, the wrong version is kept and
marked, because "we believed X, measured Y" is the useful artifact.

---

## Contents

- [Day 0 — does it run at all](#day-0--does-it-run-at-all-2026-08-2829)
- [Day 1 — the performance model, and a false start that OOM-killed the box](#day-1--the-performance-model-and-a-false-start-that-oom-killed-the-box)
- [Day 1 — the transfer-mode discovery](#day-1--the-transfer-mode-discovery)
- [Day 1 — the most expensive analytical mistake: "routing is near-uniform"](#day-1--the-most-expensive-analytical-mistake-routing-is-near-uniform)
- [Day 1 — the gather kernel, and why it had to be graph-capturable](#day-1--the-gather-kernel-and-why-it-had-to-be-graph-capturable)
- [Day 1 — hot-expert pinning: the matched set](#day-1--hot-expert-pinning-the-matched-set)
- [Day 2 — the kernel microbenchmark that saved an afternoon](#day-2--the-kernel-microbenchmark-that-saved-an-afternoon)
- [Day 2 — the profiler, and what was actually in the budget](#day-2--the-profiler-and-what-was-actually-in-the-budget)
- [Day 2 — the OOM that keeps coming back, and the patch that backfired](#day-2--the-oom-that-keeps-coming-back-and-the-patch-that-backfired)
- [Day 2 — the launcher bug I introduced](#day-2--the-launcher-bug-i-introduced)
- [Day 2 — quality against speed, and where it stopped](#day-2--quality-against-speed-and-where-it-stopped)
- [Day 2, later — the crash was not what we had written down](#day-2-later--the-crash-was-not-what-we-had-written-down)
- [Day 2, later — the biggest win of the day was prefill](#day-2-later--the-biggest-win-of-the-day-was-prefill)
- [Day 2, later — the correctness test that proved nothing](#day-2-later--the-correctness-test-that-proved-nothing)
- [Day 2, later — closing speculation with numbers](#day-2-later--closing-speculation-with-numbers)
- [Day 2, later — the re-profile that lost](#day-2-later--the-re-profile-that-lost)
- [Day 2, later — the floor that came back](#day-2-later--the-floor-that-came-back)
- [Day 2, later — the context we had been advertising and never testing](#day-2-later--the-context-we-had-been-advertising-and-never-testing)
- [Day 2, evening — the agent that kept forgetting, and the effort knob](#day-2-evening--the-agent-that-kept-forgetting-and-the-effort-knob)
- [Operational traps](#operational-traps)
- [Open failures](#open-failures)
- [Lessons, ranked by what they cost](#lessons-ranked-by-what-they-cost)

---

## Day 0 — does it run at all (2026-08-28/29)

Covered in full in [RUNNING-GLM53-ON-ONE-GPU.md](RUNNING-GLM53-ON-ONE-GPU.md);
summarised here only because the shape of it matters for what follows.

Eight bugs stood between a fresh checkout and correct output. Seven of them
announce themselves. **Bug 7 does not**: on SM120 the auto-selected NVFP4 MoE
backend (`flashinfer_cutlass`) accepts the work and returns garbage — the model
loads, `/health` is green, and every prompt emits the same token forever
(`"lock"`, id 1023). vLLM's own `cutlass` does the same. Their siblings
(`flashinfer_trtllm`, `flashinfer_cutedsl`) *refuse loudly* on the same
hardware. Only `marlin` is both accepted and correct.

That is the defining hazard of this whole project and it recurs below in three
different disguises: **on this stack, the plausible-garbage failure mode is
common and the loud failure mode is the lucky one.** Every optimization that
touches expert indexing was built with that in mind.

End of day 0: **8.7 tok/s** without MTP, **10.6–12.1 tok/s** with MTP-2.

---

## Day 1 — the performance model, and a false start that OOM-killed the box

The arithmetic said the whole problem was one transfer. Per decoded token the
model touches **4.43 GiB** of expert weights (42 MoE layers × top-8 of 288 ×
~13.5 MiB). At `--cpu-offload-gb 105`, ~65% of that is host-resident and crosses
PCIe every token. Measured on the live server at 512k / MTP-2:

```
time per forward pass                 207 ms
bytes crossing PCIe per pass         3.08 GB
implied effective bandwidth          15.1 GB/s     <- of a 56.4 GB/s link
```

Decomposed, 204 of 207 ms was that transfer. Compute was ~1.5% of the budget.
So: make the transfer faster.

**The false start.** The reasoning was that a DMA copy needs a *host* tensor as
its source, and vLLM's C++ `get_cuda_view_from_cpu_tensor` keeps no
Python-visible host handle. So the offloader was patched to allocate an
exact-size CPU tensor, pin it in place with `cudaHostRegister` (avoiding torch's
`pin_memory()`, which rounds to power-of-two), and keep the handle on the
parameter as `_vllm_cpu_source`.

Every component verified in isolation. Exact-size pinning: 1.00x RSS vs 1.48x.
`is_pinned()` → `True`. The UVA view still worked and aliased our pages. DMA
from it hit 45.2 GB/s.

Then the A/B on an 8-layer truncated model, 18.56 GiB offloaded:

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

Host RAM is 182 GiB and there is no swap. The load climbed past 160 GiB and
froze the machine.

What identified the duplication was not a snapshot but a *timeline*: a memory
sampler recording `RssAnon` and `RssShmem` separately from `/proc/<pid>/status`,
correlated against engine log timestamps, showed `shmem` at 0.0 during offload
and climbing to exactly the offloaded size **during the repack window**. `VmRSS`
alone would have shown a big number and nothing else.

The whole patch was reverted. It was never needed — see the next section.

---

## Day 1 — the transfer-mode discovery

This is the finding that generalises beyond this model. Fetching 8 experts ×
13.5 MiB (108 MiB, one MoE layer's working set for one token) from host memory
to VRAM, five ways:

| method | time | bandwidth |
|---|---|---|
| `index_select` from the UVA view | 10.50 ms | **10.78 GB/s** |
| per-expert `copy_` from the UVA view | 2.52 ms | **44.89 GB/s** |
| raw `cudaMemcpyAsync` from the UVA view | 2.53 ms | 44.75 GB/s |
| per-expert `copy_` from a **pinned host tensor** | 2.54 ms | 44.60 GB/s |
| single contiguous DMA (practical ceiling) | 1.96 ms | 57.86 GB/s |

**UVA memory is not slow. The gather *kernel* is.** `index_select` on a
host-mapped tensor launches a gather kernel: the SMs issue 64–128 B loads across
the PCIe BAR from inside the kernel, with limited outstanding requests and no DMA
engine involvement. A per-row `copy_` on the *same view* lowers to
`cudaMemcpyAsync` and runs on the copy engine at full link speed — **4.2x faster
from identical memory**.

Row 3 of that table is what made the previous section's work unnecessary:
copying from the device-side UVA view is exactly as fast as copying from the
original pinned host tensor. **Five minutes of `cudaMemcpyAsync` timing would
have skipped the entire detour.** The seductive part was that every *component*
of the patch verified correctly; the flawed step was the premise that made them
necessary.

Row 5 also settles a question nobody needed to ask again: per-expert blocks are
contiguous — the scatter is in *which* experts get picked, never within one — so
8 separate 13.5 MiB copies reach ~99% of the single-contiguous-DMA ceiling.
Batching them buys nothing.

---

## Day 1 — the most expensive analytical mistake: "routing is near-uniform"

At 04:06 the router histogram was dumped and it said this:

```
MoE skew @4000 steps: top-25% of experts capture 34.1% of activations
                      (uniform would be 25.0%); top-35% -> 44.6%
MoE skew @8000 steps: top-25% -> 33.2%;  top-35% -> 43.9%
```

That is weak skew. Combined with an LRU cache measuring a **9.0% hit rate**
(10188/113208 fetches, 16 slots of 288 = 5.6% capacity — barely above random),
the conclusion written down was:

> GLM-5.3's expert routing is near-uniform with no locality. Caching levers are
> capped at ~1.15x.

**That conclusion was wrong, and it would have closed the branch that eventually
produced the entire win.** It survived about eighteen hours.

The error was in the *aggregation*, not the measurement. The histogram summed
per-expert counts **across layers**. Different layers are hot on different
experts, so summing washes the skew out. Recomputed from the shipped profile
(33 offloaded layers × 288 experts, 12,000 staged steps) the same data says:

| capacity | **aggregate** (summed across layers) | **per layer** |
|---|---|---|
| top 10% | 13.8% of activations | **38.9%** |
| top 25% | 32.5% | **64.1%** |
| top 35% | 43.2% | **74.3%** |
| top 50% | 58.8% | **85.7%** |

Measured per layer, **the hottest ~36% of a layer's experts serve 75% of that
layer's fetches**. The hit-rate curve at a uniform slot budget per layer:

```
 33 slots/layer -> 42.65%      132 -> 82.99%
 66 slots/layer -> 61.46%      165 -> 89.62%
 99 slots/layer -> 73.98%
```

Two related properties genuinely do *not* pay, and it is worth being precise
about which is which, because they get conflated constantly:

- **Temporal locality** (does the next token reuse the last token's experts?) —
  measured, 9% hit. Real, and useless.
- **Static per-layer frequency skew** (are some experts simply hotter, over a
  long window?) — measured, 61.5% hit at 23% capacity. This is the one that pays.
- **Variable per-layer slot allocation** ("water-filling" the same total slots
  toward the more skewed layers) — measured, **+0.07 pp**. The layers'
  distribution *shapes* are alike even though their hot *sets* differ, so uniform
  allocation is already optimal. A clean negative that saved building a solver.

---

## Day 1 — the gather kernel, and why it had to be graph-capturable

The per-row `copy_` version worked and was fast, but it had to drive the loop
from the host — `experts.tolist()` — and a device-to-host sync is illegal inside
CUDA graph capture. So the whole engine ran under `--enforce-eager`.

That trade was defensible at 207 ms per forward pass, where graph launch
overhead is noise. It stopped being defensible as the pass got shorter:
**11.77 tok/s eager vs 13.53 tok/s with graphs**, identical staging. Eager was
costing 13%.

So the gather was rewritten as a Triton kernel that takes its indices as a
**device tensor**. No sync, so it captures cleanly. Measured (8 rows of 4 MiB
from a host-mapped view — note this is a different row size from the table two
sections up, so the absolute numbers differ; the ranking does not):

```
torch.index_select                  1.45 ms    23.1 GB/s
per-row copy_ (cudaMemcpyAsync)     0.81 ms    41.4 GB/s
this kernel                         0.64 ms    52.1 GB/s   <- 92% of 56.4 GB/s H2D
```

`BLOCK = 8192` measured fastest; 4096 ties; 16384 and above fall to ~49 GB/s.

Three things in that kernel are not obvious, and each of them is a bug that was
found the hard way:

**1. It cannot deduplicate with `torch.unique`.** `topk_ids` repeats experts —
the same expert is picked by several of the tokens verified in one MTP step —
but `torch.unique` has a **data-dependent output shape**, which graph capture
rejects. Instead the grid stays statically shaped and each program checks whether
its expert already appeared at a lower row, returning without copying if so.
Only unique bytes cross PCIe; the grid never changes shape.

```python
r = tl.program_id(0)
e = tl.load(idx + r).to(tl.int64)
js = tl.arange(0, MAXN)
prev = tl.load(idx + js, mask=js < r, other=-1).to(tl.int64)
if tl.sum((prev == e).to(tl.int32)) > 0:
    return          # this expert was already staged at a lower row
```

**2. The expert map must point at the *first* occurrence.** Because duplicates
are skipped, only the first row of a repeated expert is actually filled. The
obvious `m[experts] = arange(n)` will not do: PyTorch leaves the winner
**undefined** when scatter indices repeat, and empirically returns last-ish
occurrences — which are exactly the rows that were never written.
`scatter_reduce_(..., reduce="amin")` is deterministic and correct.

This is the plausible-garbage failure mode again: a wrong `expert_map` reads the
wrong weights, produces finite numbers, and raises nothing.

**3. Only tensors genuinely indexed by expert on dim 0 may be gathered**, and if
*any other* per-expert tensor would remain globally indexed while the staged
ones are locally indexed, staging must bail out entirely. Mixed indexing reads
the wrong rows with no error. The check is explicit in `MarlinExperts.apply()`:

```python
others = (self.w1_bias, self.w2_bias, self.a1_gscale, self.a2_gscale,
          self.w1_zp, self.w2_zp, self.w13_g_idx, self.w2_g_idx,
          self.w13_g_idx_sort_indices, self.w2_g_idx_sort_indices)
unstaged_per_expert = any(
    t is not None and t.ndim >= 1 and t.shape[0] == global_num_experts
    for t in others
)
```

The neat part of the whole integration is that no kernel change was needed.
Staged weights of shape `[n_selected, ...]` plus a local `expert_map` is exactly
the shape vLLM's expert-parallelism path already produces, and the Marlin
kernels already honour that translation table.

**13.53 tok/s**, from 10.77.

---

## Day 1 — hot-expert pinning: the matched set

vLLM's `cpu_offload_gb` makes residency a **whole-layer** decision: some layers
are entirely in VRAM, the rest entirely on the host. So the resident share of
expert *fetches* equals the resident share of *layers*, and **which** experts
those are is decided by module iteration order — i.e. arbitrarily.

Given the per-layer skew above, that is a large amount of VRAM spent badly. The
fix is a per-layer VRAM cache of the frequency-ranked hottest experts, plus a
device-resident slot table the gather kernel reads to decide, per row, whether to
read VRAM at ~1.8 TB/s or PCIe at ~52 GB/s:

```python
sl = tl.load(slot + e).to(tl.int64)
if sl >= 0:
    v = tl.load(cache + sl * row_bytes + offs, mask=m, other=0)   # VRAM hit
else:
    v = tl.load(src   + e  * row_bytes + offs, mask=m, other=0)   # PCIe miss
```

The slot table is on the device precisely so the branch costs no host sync and
graphs stay captured — the point from the previous section, reused.

**And then the counter-intuitive part.** Because the cache needs VRAM, you buy it
by offloading *more*:

| configuration | tok/s |
|---|---|
| offload 105 GiB, no pinning *(the old default)* | 13.73 |
| offload 125 GiB, no pinning | **11.34** ← worst of the three |
| offload 125 GiB + 68 hot slots/layer | **16.76** |
| offload 125 GiB + 66 hot slots/layer *(shipped when measured)* | 16.50 |

These knobs are a **matched set**. Offloading more is faster only because the
VRAM it frees funds the cache; take the cache away and 125 is the worst
configuration on the list. The old launcher header — *"above ~100 just costs
speed"* — was correct when written and became wrong the moment pinning existed.

### The slot ladder, and where it stops

More slots is monotonically better for hit rate and abruptly fatal for VRAM:

| slots/layer | outcome |
|---|---|
| 96 | `CUDA out of memory. Tried to allocate 384.00 MiB` **at startup** |
| 76 | `... 608.00 MiB` at startup |
| 68 | boots, then OOMs **mid-serving** |
| **66** | stable |

66 costs ~29 GiB of VRAM (66 slots × 33 offloaded layers × ~13.5 MiB). The two
slots given up between 68 and 66 are the **coldest** of the pinned set: ~0.9 GiB
freed for a fraction of a point of hit rate, and prefill speed is untouched.

Lowering the offload has a symmetric floor: `--cpu-offload-gb 96` dies at init
with `Tried to allocate 6.75 GiB`.

*(Later on 2026-09-02 the shipped count went to **62**. The ladder is a method,
not a table of safe values: "stable" means stable for whatever traffic was run,
and the cache is invisible to vLLM's memory profiling, so long-prompt traffic
moves the wall. Each step down sheds the coldest experts of the pinned set —
66 → 62 is ~1.8 GiB for a fraction of a point of hit rate.)*

### Two silent failures in the profile plumbing

The profile is per-layer frequency counts, recorded from real traffic, and it is
**tied to the offload it was captured at** — layer *N* in the profile means "the
*N*-th offloaded layer", so a different `cpu_offload_gb` gives a different set of
host-side layers and the indices stop lining up.

- A profile captured at offload 125 and used at 127 measured **15.09 vs 16.76
  tok/s**. No error, no warning. Roughly 10% gone. The launcher now compares the
  `offload_gb` recorded inside the profile against the environment and says so at
  launch; the pinner warns too.
- The profile writer originally keyed layers by `sorted()` order of the `w1`
  data pointer, while the pinner read them back by *discovery* order. Both
  parsed. Every layer got **another layer's hot set**: measured hit rate 45%
  against a predicted 73%. Nineteen points of hit rate, silently. The writer now
  keys by discovery order, and the pinner refuses (with a warning) any layer
  whose profile entry is missing or the wrong length.

Both of these are the same failure class as bug 7: a wrong answer that parses.

---

## Day 2 — the kernel microbenchmark that saved an afternoon

Marlin logs *"Your GPU does not have native support for FP4"* on every startup
and falls back to weight-only decompression. The obvious hypothesis was that the
dequant emulation was costing real time and that porting the staging path to a
kernel with native FP4 support (`b12x`, FlashInfer, CUTLASS) would win.

Instead of porting, we benchmarked the kernels in isolation first — GLM-5.3
dimensions (hidden 4096, moe_inter 2048, top-8), 32 experts so the weights fit in
VRAM, 8 routings, 100 iterations after 25 warmup, median of the expert-GEMM
region:

| M | marlin | flashinfer_cutlass | vllm cutlass | b12x |
|---|---|---|---|---|
| 1 | **77.5 µs** | 99.8 (0.78x) | 110.0 (0.70x) | 409.4 (**0.19x**) |
| 2 | **133.9** | 159.3 | 176.5 | 398.4 |
| 4 | **217.0** | 234.7 | 244.9 | 406.5 |
| 64 | **318.0** | 326.3 | 341.3 | 994.9 |

**Marlin wins at every batch size.** The "native FP4 is being emulated" warning
is unconditional in the Marlin path and proves nothing about performance. The
hypothesis died in the time it took to run a benchmark instead of a port.

We ran the end-to-end version anyway, because the microbenchmark's scale is not
the real model's: `moe_backend=b12x` on the live server produced a steady
**1.2–1.3 tok/s** before the engine fell over. An order of magnitude worse than
the configuration we already had.

Two things about that benchmark are worth passing on:

- **It measured speed, not correctness.** In the synthetic harness all four
  backends agree with Marlin to cosine ≈ 0.986. That does **not** contradict
  bug 7 — 32 synthetic experts with synthetic scales is not the real
  checkpoint's 288 experts with their real scale distribution. Do not read the
  correctness columns of a speed benchmark as a correctness result.
- **The FlashInfer rows failed the first time** with `RuntimeError: Could not
  find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`. FlashInfer's
  JIT needs `CUDA_HOME` pointed at the venv's `nvidia/cu13` and needs `ninja`
  installed. The first `bench_results.json` has four `error` entries where the
  FlashInfer numbers should be; `bench_fi.json` is the rerun.

Also closed on the same grounds, earlier: `b12x` calls `del w1, w2` and
hard-raises on `expert_map`; FlashInfer *accepts* an `expert_map` and then never
reads it — silently wrong. Neither can host this staging design.

---

## Day 2 — the profiler, and what was actually in the budget

Up to this point the cost model was arithmetic (bytes ÷ bandwidth) validated by
end-to-end throughput. It was time to look.

**Getting a trace at all.** This fork has **no `VLLM_TORCH_PROFILER_DIR`** — the
environment variable everyone reaches for does nothing. It uses CLI flags:

```
--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=/some/dir
```

then `POST /start_profile` and `POST /stop_profile`. `nsys` and `ncu` are not
installed on this box.

Per decode step (median of 12 steps from the trace):

```
staging gather (PCIe)  34.06 ms   <- dominant, as expected
dense bf16 GEMM        18.98 ms   <- second, and never suspected
MoE experts (marlin)    3.58 ms
attention               0.47 ms   <- 0.1% of GPU time
norm/activation/other   1.38 ms
```

GPU busy exceeds the wall span by ~13.7 ms, so some overlap already happens.

**Attention was never the bottleneck.** On a 262k-context model, that is not the
intuition anyone brings to the problem, and it is worth stating loudly: 0.47 ms.
Every hour that could have gone into attention backends would have been wasted.

The surprise is the **18.98 ms of dense BF16 GEMM**. That is the 18.01 GiB of
**non-expert** weights, which the NVFP4 quantizer left entirely in BF16:

```
attention projections   11.74 GiB
shared expert            2.02
dense MLP                1.58
embedding                1.18
lm_head                  1.18
```

Every one of those bytes is read for every token. At ~1.8 TB/s the floor is
10.74 ms, so we are running that phase at ~57% of peak bandwidth. Compressing
them to FP8 is the largest untried lever (~5–6 ms/token) and it is not free —
see below.

**A trap in reading the trace.** The captured window contains 2 prefill steps at
~811 ms of CUDA time each alongside the 12 decode steps. The profiler's
aggregate operator table is therefore dominated by *prefill* — the Marlin MoE
kernel shows up as the single largest line item at 37% of CUDA time, which is
the exact opposite of its 3.58 ms share of a decode step. The decode
decomposition above is decode steps only. Do not read a mixed trace's summary
table as a decode budget.

---

## Day 2 — the OOM that keeps coming back, and the patch that backfired

Three identical crashes across the day:

```
CUDA out of memory. Tried to allocate 124.00 MiB.
GPU 0 has a total capacity of 94.97 GiB of which 81.69 MiB is free.
```

Always *after* load succeeded, always mid-serving, on a card with 95 GiB.

**Root cause.** `stage()` returns early when `n > MAX_STAGED_EXPERTS` (64).
vLLM's memory profiling runs a full max-size prefill where `n` is far above that
cap, so it returns before `_maybe_pin` ever allocates the ~29 GiB hot-expert
cache. vLLM then sizes KV and workspaces believing that VRAM is free. The cache
is allocated later, **during CUDA-graph capture**, leaving ~2 GiB of real margin.

The shipped configuration makes this worse in an interesting way. With
`--kv-cache-memory-bytes` set, vLLM does not profile at all:

```
Initial free memory 94.36 GiB, reserved 4.0 GiB memory for KV Cache as specified
by kv_cache_memory_bytes config and skipped memory profiling.
```

`94.36 GiB` is measured before the weights land (the model then reports *"Model
loading took 55.28 GiB"*). So nothing in the engine's accounting has ever seen
either the resident weights or the hot cache at the moment the KV decision is
made. The pinning log line — `Hot-expert pinning ACTIVE: 66/288 ...` — appears
*during* `Capturing CUDA graphs`, which is exactly the point being made.

**The principled fix, and why it is not shipped.** Moving the cap check after the
allocations makes the reservation happen during profiling, so vLLM sees the
truth. Tried it. vLLM then fails **loudly at startup** (`Tried to allocate
512.00 MiB`), which is correct behaviour — but serving afterwards hits `CUDA
error: an illegal memory access`. The profiling run's allocator state does not
survive. Reverted.

What *was* kept is the smaller half of that change: the staging buffers are
reserved before the `if not any(stageable): return None` early exit, so at least
they are never sized late.

```python
# Reserve unconditionally, before any early return: profiling runs a full
# prefill batch where n exceeds the cap, so an allocation placed after the
# cap check never happens during profiling and the KV cache eats the VRAM.
for key, (t, do) in enumerate(zip(tensors, stageable)):
    if do:
        self._buffer(key, t)
```

**Then the obvious way to buy headroom turned out to be closed too.** Shrinking
the prefill workspace should free VRAM. Instead:

| `--max-num-batched-tokens` | result |
|---|---|
| 2048 | `CUDA error: an illegal memory access` on the **second** request |
| 4096 | same |
| 8192 | 8 sustained 200-token generations, **16.50 tok/s** |

The stack is in `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`. It is
*not* a collision with `index_topk=2048`, because 4096 fails too. Not
root-caused; ~~**treat 8192 as a floor.**~~ (The pre-optimization 2026-08-29
stack ran at 4096 without this failure, so the floor belongs to the current
staging + pinning path, not to the model.)

> **This conclusion is wrong and is corrected later the same day**
> ([Day 2, later](#day-2-later--the-crash-was-not-what-we-had-written-down)).
> The last sentence is the only part that survives, and it was the clue: the
> failure arrived with that day's staging work. The batch size was never the
> variable — the same fault hit at 8192 after 12 requests, and it is an Xid 31
> MMU fault. Left in place because the reasoning below it (*"shrinking the
> workspace is the obvious way to buy headroom"*) is still how anyone would get
> here.

**And `gpu-memory-utilization` cannot help either.** 0.95 and 0.96 produce
byte-identical results — the same `GPU KV cache size: 272,771 tokens` — because
`KV_BYTES` is explicit and weights + cache are fixed. There is no slack for the
utilisation fraction to redistribute.

So real headroom comes from exactly two places: fewer hot slots, or a higher
offload. That is why the shipped default was 66 and not 68 — and, by the end of
the day, 62.

---

## Day 2 — the launcher bug I introduced

I added an `EXTRA_ARGS` passthrough to `serve-opt.sh` and wrote a comment
explaining it — *inside* the backslash-continued `vllm serve` invocation. A
comment ends the continuation. Bash then reads the rest of the flag list as new
commands and hits an unterminated quote:

```
./serve-opt.sh: line 277: unexpected EOF while looking for matching `"'
```

But the server had **already started**, with everything after the comment
silently dropped. Compare the engine's own `non-default args` from that run
against the next one:

```
BROKEN  {..., 'cpu_offload_gb': 125.0, 'max_num_batched_tokens': 8192, 'max_num_seqs': 1}
        moe_backend='auto'

FIXED   {..., 'enable_auto_tool_choice': True, 'tool_call_parser': 'glm47',
         'reasoning_parser': 'glm47', 'kv_cache_memory_bytes': 4294967296,
         'enable_prefix_caching': True,
         'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2},
         ... moe_backend='marlin'}
```

Gone: the MoE backend (falling back to `auto`, i.e. **the silently-wrong CUTLASS
kernel**), MTP, prefix caching, the KV cap, and both parsers. The run looked
like a run. This bug class had bitten before, in the same file.

The structural fix is not a comment-placement rule, it is **never putting
commentary inside a continued command** — the note now sits above the
invocation, where a stray `#` cannot truncate anything:

```bash
# EXTRA_ARGS: unquoted passthrough for one-off flags (word-split on
# purpose) ... NOTE: never put a comment INSIDE the backslash-continued
# invocation below -- it ends the continuation and every later flag is
# silently dropped.
"$VENV/bin/vllm" serve "$MODEL" \
    ...
```

The general lesson is the cheap one: **when a launcher can lose flags silently,
verify the engine's own echo of its configuration, not the script you wrote.**
vLLM prints `non-default args` at startup; grep it.

---

## Day 2 — quality against speed, and where it stopped

Two levers remained. Both change model outputs, and the standing constraint on
this deployment is that it is used for critical work and must not hallucinate
more in exchange for speed. Both were measured anyway, because a documented
"no" is worth more than an assumption.

### Router pruning: +24%, and visibly worse

Decode fetches every one of the top-8 routed experts, and PCIe traffic scales
with how many distinct experts a step touches. The router's weight by rank,
averaged over 3000 steps of real traffic:

```
[0.269 0.186 0.137 0.109 0.091 0.078 0.069 0.061]
cumulative: [0.269 0.455 0.592 0.701 0.792 0.870 0.939 1.000]
```

The last two ranks carry 13% of the routed weight between them. Dropping them
measured **20.48 tok/s vs 16.50 — +24%**.

And the output degrades visibly. Repetition loops (*"Jupiter (planet) (planet)
(planet)"*), explanations drifting into word salad. Simple facts survived —
Paris, 17×23=391 — which is precisely why a spot check would have passed it.

Shipped **off** (`VLLM_MOE_DECODE_TOPK`, default 0). The implementation selects
by weight with `torch.topk` and renormalizes (GLM sets `norm_topk_prob=True`)
rather than slicing `[:, :k]`, because the router's output order is not
guaranteed weight-sorted and slicing the wrong experts would degrade quality
*without* the compensating repetition tell. Static output shape, so graphs still
capture.

Reducing top-k further does not rescue it either: from the weight-by-rank table,
top-4 discards 30% of the routed weight and still projects only ~17.7 tok/s.

### FP8 for the 18 GiB of dense weights: not free, and no checkpoint fits

The profiler said 18.98 ms/step goes into BF16 dense GEMM. FP8 would save
roughly 10 ms/token. Three separate findings closed it:

1. **"FP8 is free because FP8 is the native checkpoint format" is refuted by
   direct measurement.** The BF16 dense tensors in this checkpoint are *not*
   FP8-representable — round-tripping them through e4m3 has non-zero error, so
   they are genuine BF16 masters, not upcast FP8. Properly scaled FP8 (per-tensor
   *and* per-channel) gives **~31 dB SNR**: the expected figure for e4m3's three
   mantissa bits, and a real if mild loss.
2. **No suitable checkpoint exists.** A full-FP8 GLM-5.3-Flash is ~331 GB. This
   box has 182 GiB host + 96 GiB VRAM ≈ 278 GB. It cannot fit *at all* — that
   rules out `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` and
   `orcarouter/...-Uncensored-FP8` on capacity alone.
3. `LibertAIDAI/GLM-5.3-Flash-NVFP4` is the **same layout as ours** (NVFP4
   experts, everything else BF16, ~181 GiB). There is no FP4-experts +
   FP8-dense variant to download.

Quantizing the dense weights ourselves remains the largest untried lever. It
would cost ~31 dB of signal, and it has not been done.

**The quality-neutral ceiling is ~16–17 tok/s.** 25 tok/s requires stacking both
of the above.


---

## Day 2, later — the crash was not what we had written down

Two sections above, this notebook records `--max-num-batched-tokens` below 8192
as a landmine: 2048 and 4096 both die with an illegal memory access on the
*second* request, 8192 is a floor, not root-caused. Two of those three claims
were wrong — or so it looked for the next six hours. (Spoiler, and the reason
this section is worth reading twice: the floor came *back*. See
[the floor that came back](#day-2-later--the-floor-that-came-back).)

At 02:32 the same fault hit a run at **8192**, after 12 requests. That kills the
floor — or rather, it kills *"8192 is safe"*, which is not the same claim, and
conflating the two is the mistake made here. The reading at the time was: the
batch size was never the variable, and what smaller values did was change the
allocation layout so the corruption surfaced on request 2 instead of request 12
— a classic memory-corruption tell, read as a threshold. Half of that is right.
The half that is wrong is assuming one counter-example dissolves a reproducible
result rather than revealing a second one.

The kernel log had the answer the whole time:

```
Xid 31 ... MMU Fault
```

An MMU fault is an access to an unmapped address. The Python traceback pointed
at `flashinfer_mla_sparse_sm90.py`, which is why the attention backend spent a
day as the prime suspect — but CUDA's own message says errors of this class are
*reported asynchronously at some later call*. **A stack is where the error was
noticed, not where it was caused**, and we had been reading it as though it
were. For scale, the previous day had 22 runs and ~80 requests with zero such
faults, so whatever changed came from that day's work.

Two changes went in:

1. **Bounds-check every id-indexed access in the staging path** — both Triton
   gather kernels, `build_local_expert_map`'s `scatter_reduce_`, and the Marlin
   call site. This is the fix that matches the symptom exactly: a padded or
   sentinel expert id (`-1`, or `n_experts`) indexes past the host tensor, or —
   worse — reads garbage from past the slot table and then *dereferences that
   garbage into the cache*, which is precisely an unmapped address.

   ```python
   if (e < 0) | (e >= n_experts):
       return          # such a row is never fetched; the map never points at it
   ```

2. **`VLLM_USE_BREAKABLE_CUDAGRAPH=0`.** vLLM auto-enables "breakable" CUDA
   graphs for this hybrid model. Every build that ran under them on 2026-09-02
   faulted between request 2 and request 24. With them off, a 40-request varied
   soak at T=0/0.7/1.0 — four 9k-token prefills and a 30k-token prompt included
   — ran **0 failures, 0 Xid, at unchanged speed: 15.77 tok/s** against a
   15.7–16.5 band. It is now the launcher default.

**The mechanism is still not proven** and this notebook is not going to pretend
otherwise. The supporting observation is that the profiler had shown kernels
overlapping across streams while the staging buffers are *shared* — consistent
with a race that breakable mode makes possible. Two clean soaks against
same-day faults on every breakable build is enough to ship on. It is not a root
cause.

The third candidate fix died on its own: vLLM lists both
`FLASHINFER_MLA_SPARSE_SM90` and `FLASHINFER_MLA_SPARSE_SM120` for this model and
picks the SM90 one — the Hopper kernel running on Blackwell by the fallback from
day 0's bugs 5–6. Forcing the native SM120 backend fails at startup with *"SM120
sparse MLA v32/GLM expects kv_lora_rank=512, qk_rope_head_dim=64, and query head
dim 576"*. GLM-5.3 is NoPE MLA (`qk_rope_head_dim=0`, latent 512) — the same four
broken DeepSeek assumptions from day 0, in a kernel nobody has ported yet.

`--max-num-batched-tokens 8192` remains the *tested* default. Smaller values
were, at this point, believed "not known to be unsafe on their own" — they
simply had not been retested since the fix. **They were retested a few hours
later and they are unsafe**; see
[the floor that came back](#day-2-later--the-floor-that-came-back).

---

## Day 2, later — the biggest win of the day was prefill

Everything in this notebook up to here is a decode number. That is a choice of
benchmark, and it hid the term that dominates a real coding turn.

Streamed, on the 16.5 tok/s configuration:

```
10,296-token prompt   TTFT 16.8 s   =  614 tok/s prefill
                                       262k context => ~7 minutes before token 1
decode after that                      18.8 tok/s   <- fine
```

Decode at long context was never the problem. **Time to first token was**, and
no decode benchmark on this page would ever have shown it.

**The cause is the optimization's own guard rail.** `stage()` returns `None`
when `n > MAX_STAGED_EXPERTS` (64). A decode step touches 8 experts per layer,
so it stages. A prefill chunk routes to essentially *all* 288, so it does not —
and Marlin falls back to reading every offloaded expert zero-copy over the BAR
at ~23 GB/s with tile re-reads, roughly 98 GiB per chunk after cache hits,
instead of the 52 GB/s DMA gather. The property that makes decode fast (top-8 of
288) is exactly what stops applying during prefill.

**The fix is to stop treating "too many experts" as a bail-out and start
treating it as a loop.** `VLLM_MOE_PREFILL_GROUPS=1` walks the 288 experts in
64-wide groups per layer: stage the group (hot-cache hits still count),
`moe_align_block_size` with an `expert_map` that hides every other expert, run
the inner `_fused_marlin_moe` into the **same** per-pair workspace, and finish
with one fp32 `moe_sum` over the `[M, topk, K]` view.

Re-measured afterwards on the shipped server (pid 211628, bare `./serve-opt.sh`
defaults), streaming, TTFT = time to the first content chunk, on salts **never
sent to that server before** so no prefix-cache reuse is possible — logged in
`artifacts/prefill-fresh-2026-09-02.txt`:

| prompt | prompt tokens | TTFT | prefill | decode after |
|---|---|---|---|---|
| 10k fresh #1 | 10,313 | **5.82 s** | **1771 tok/s** | 18.2 tok/s |
| 10k fresh #2 | 10,314 | **5.96 s** | **1729 tok/s** | 18.2 tok/s |
| 30k fresh | 35,571 | **22.57 s** | **1576 tok/s** | 18.4 tok/s |
| 14-token control *(zero-copy path)* | 14 | **0.85 s** | — | — |

against **614 tok/s / 16.8 s** before grouping. Decode is untouched.

The logging is itself part of the lesson. The figure this notebook carried for
most of the day — "1762 tok/s, 5.8 s" — came from a single pass whose output was
never written down, and it took a deliberate re-run with fresh salts to turn it
into something anyone could check. It survived, to within two runs' spread. It
might not have.

### The two mistakes inside that

**Summing partial outputs is not the same as filling one buffer.** The first
draft ran each group to completion and added five bf16 partial results. It
flipped tokens at T=0. The shipped version works because every `(token, expert)`
pair belongs to exactly one group and the second Marlin GEMM writes only the
sorted *valid* rows of the per-pair buffer — so the groups fill **disjoint rows
of one workspace** and there is exactly one reduction, in fp32, at the end. Five
bf16 additions of nearly-orthogonal partial sums is a different computation from
one fp32 reduction, and on a knife-edge argmax that difference is visible.
(`moe_sum` also needs the 3-D view, or its C++ op fails with
`aoti_torch_get_stride`.)

**The short-prompt trap, and the measurement error it caused.** The first
version grouped *every* prefill. A 14-token prompt routes to only ~100 experts,
so staging all 288 made it **slower**: 2.05 s to the first token, against 0.85 s
for the zero-copy path on that same prompt. The 200-token
benchmark then read **12.5 tok/s instead of 15.8**, and that was called a decode
regression.

It was not. The decode rate was unchanged; the whole difference was TTFT, and a
single streamed run separating the two showed it immediately. The same
arithmetic explains a 40-request soak taken in that window:

| class | before grouping | during (ungated) | why |
|---|---|---|---|
| math (15-token prompt, 150 gen) | 16.67 | 13.91 | 150 tokens at 16.67 = 9.0 s; **+2.05 s of TTFT predicts 13.6** |
| prose (19-token prompt, 220 gen) | 13.73 | 12.83 | same mechanism, longer generation dilutes it |
| **long (9,107-token prompt, 120 gen)** | **7.83** | **11.34** | the case grouping is for |

(Those are whole-request rates — completion tokens over wall time — so they
include TTFT by construction. That is what makes them the right measurement here
and the wrong one for a decode claim. The long class re-sends the *same* 9k
prompt eight times, so it exercises prefix reuse as much as cold prefill.)

The gate is `topk_ids.shape[0] >= 48` — static per captured graph, so it is
CUDA-graph-safe — and the crossover is around 40 tokens.

**Always separate TTFT from decode before calling something a decode
regression.** It cost an hour and a wrongly-reverted patch (see the next
section) to learn a thing that one streamed request would have shown.

### Prefix caching, partially and unobservably

The earlier finding was blunt: an identical prompt re-sent took 16.9 s against
the original 16.6 s, i.e. `--enable-prefix-caching` bought nothing. That is now
too strong. On the shipped build an exact repeat prefills in **~3.2 s**, and on
the pre-grouping build the same test measured 9.6 s against 16.8. So reuse
*does* happen for identical prefixes — inconsistently, and it is invisible:
`vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` both read **0** for
every one of these requests. Timing is the only evidence available.

---

## Day 2, later — the correctness test that proved nothing

The grouped prefill path shipped at 04:05 with this claim, which sat in the
launcher and in the source comment until both were corrected later that day:

> *bit-identical to the single pass (T=0 outputs identical on 3k and 10k
> prompts)*

**Withdrawn.** Not because the path is wrong — it is fine, see below — but
because the test cannot distinguish "correct" from "lucky".

The baseline prefill on this stack, with **no code change at all**, produces
different output run to run. The same 3k and 9k prompts at T=0, `max_tokens=40`,
hashed **A, B, A** across three runs of the unmodified build. The grouped path
produced three distinct hashes too, one of which equalled a baseline hash. The
pattern matches prefix-cache state reuse — mamba `align` mode caches state at
scheduler-step boundaries, a continuation from cached state rounds differently
in bf16, and a knife-edge argmax flips — rather than a race.

Two lucky draws at 04:05 produced "identical" and were believed. Then the same
invalid test was used in the other direction: a masked-group variant (which
skipped unneeded experts instead of staging all 288) was **removed for
"nondeterminism" that the baseline has as well**. A test that cannot fail
correctly also cannot pass correctly.

That second casualty was not free. On the 14-token control the masked variant
measured **0.36 s** to the first token, against **0.85 s** for the zero-copy
path that now ships and 2.05 s for ungated grouping — so the code that got
deleted was faster than *either* shipped alternative, on exactly the case the
≥ 48-token gate exists to work around. It is an open item now, not a closed
branch.

**The valid test** measures the candidate against the baseline's *own* drift.
Capture per-token `prompt_logprobs` on **fresh** prompts (new salt, so nothing is
cached), thousands of values, and compare the build-to-build difference against
the baseline's fresh-vs-cached difference on the same prompts:

| comparison | tokens | p50 | p90 | p99 | max | \|Δ\| > 1 nat |
|---|---|---|---|---|---|---|
| baseline vs itself (fresh vs cached), 3k | 3,219 | 0.000006 | 0.00054 | 0.057 | 3.75 | 6 (0.19%) |
| baseline vs itself (fresh vs cached), 9k | 10,300 | 0.000011 | 0.00013 | 0.0065 | 4.41 | 5 (0.05%) |
| **grouped vs baseline**, 3k | 3,219 | 0.000008 | 0.00069 | 0.075 | 4.26 | 4 (0.12%) |
| **grouped vs baseline**, 9k | 10,300 | 0.000012 | 0.00022 | 0.014 | 5.01 | 7 (0.07%) |

The two rows that matter sit *inside* the control's spread on every statistic,
including the tail. So the defensible sentence is: **grouped prefill is
numerically indistinguishable from the baseline's own run-to-run variability.**
It is not bit-identical, and nothing measured here could have shown that it was.

Generalised: **an equivalence claim needs a control that measures the noise you
are claiming to be inside of.** Without one, "identical" is a coin flip you
reported as a result.

---

## Day 2, later — closing speculation with numbers

`FURTHER §6` had left n-gram drafting stacked on MTP as the cheap untried idea —
"low effort (a config flag), so worth trying". It was tried.

MTP-2 on the shipped config, coding prompt, T=0:

```
17.10 tok/s
cumulative acceptance   64% of draft tokens  =  1.28 accepted per step
  position 0   75%
  position 1   53%
                        =>  2.28 tokens per forward pass
```

(A second coding run, at offload 130, measured 1.81 accepted per step on its own
generation. Acceptance is workload- and generation-specific; quote it with the
prompt.)

`ngram`, `ngram_gpu` and `suffix` all exist in this fork and the launcher takes
`SPEC_METHOD` / `SPEC_EXTRA`. Both routes to a *deeper* draft are blocked:

- `num_speculative_tokens=5` raises the per-request KV requirement to **4.27 GiB
  against the 4 GiB `KV_BYTES` cap** and vLLM refuses at startup. Fundable from
  hot slots if it were ever worth it.
- `num_speculative_tokens=4` (5 query tokens per step) dies in the
  sparse-attention KPOOL indexer — DeepGEMM `csrc/apis/attention.hpp:320`,
  `DG_HOST_ASSERT` on `block_kv` for arch 12, reached through
  `sparse_attn_indexer_kpool → fp8_fp4_paged_mqa_logits`. MTP-2's 3 query tokens
  pass. This is the same assert §7.2 of the optimization doc analyses, where the
  fix is `--block-size 256`; whether that clears it at this depth on the current
  stack was **not** retested.

And at depth 2 the ceiling is arithmetic anyway: a free drafter can only save
the one MTP layer, ~2–3% of a step, and prompt-lookup acceptance on generic code
is below a trained MTP head's. **MTP-2 stays.** Do not spend launches on n-gram
unless the indexer constraint lifts.

---

## Day 2, later — the re-profile that lost

The hot-expert profile that produces the entire pinning win was captured on
**one prose prompt at T=0**. The traffic this server actually serves is coding.
That is an obvious
mismatch and it looked like free throughput.

Offline the two profiles disagreed enough to be interesting: a profile captured
over the mixed soak workload (prose/code/chat/long/math at T=0/0.7/1.0, plus a
coding prompt and a 30k prompt) at offload 130 shares only **44% of its top-64
set per layer** with the shipped one, and its decode-only hit estimate at 66
slots was ~50% against the shipped profile's 61.5%. That should have been the
warning.

Measured, at offload 130 with the new profile and 64 slots (VRAM 89.2 GiB):

| | mixed profile @ 130 | shipped @ 125 |
|---|---|---|
| prose decode | 13.02 tok/s | **15.77** |
| coding prompt | 16.10 | **17.10** |
| prefill | 1698 tok/s | **1762** (the figure current at the time) |
| VRAM margin | +4 GiB | — |

**Worse on everything except the margin.** The likely cause is in the capture,
not the idea: two thirds of the mixed workload runs at T=0.7 or T=1.0, and
sampling randomizes tokens, which *flattens* routing — so the profile records a
blurrier version of the very skew it exists to capture. Five more GiB of experts
sitting host-side does the rest. Reverted to offload 125 and the prose profile.
If this is ever retried: **capture at T=0, on the real workload only.**

One methodological trap fell out of it. A histogram captured with grouped
prefill switched on carries a **uniform +1 per expert per chunk**, because a
group walk touches every expert whether routing wanted it or not. Rankings
survive — the constant is uniform — but raw top-k shares read about **17% low**,
so subtract the per-layer minimum before estimating a hit rate from such a
profile. Cost of the whole branch: two 6-minute launches and 15 minutes of
driving, for a clean negative.

---

## Day 2, later — the floor that came back

Earlier that day this notebook recorded a satisfying correction: the
`--max-num-batched-tokens 8192` "floor" was an artifact, because the same
`illegal memory access` had hit a run **at** 8192 after 12 requests. One
counter-example, cleanly applied, and a claim retired in five documents.

Then the Xid 31 fault was actually fixed (`VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus
staging bounds checks), which meant the original experiment could finally be run
without its confound. It took ten minutes:

```
MAX_BATCHED=4096   ->  500 on the FIRST short decode request; engine dead
MAX_BATCHED=8192   ->  the shipped configuration, two clean 40-request soaks
```

**The floor is real, and it is a second failure, independent of the Xid.** The
withdrawal was right about the evidence — 8192 *did* fault, so batch size could
not be the whole story — and wrong about the conclusion it drew from that
evidence, which was that the batch-size effect had therefore never existed. A
confound invalidates an attribution, not a phenomenon. When you find one, the
next step is to re-run the original experiment without it, not to delete the
result.

It also failed *faster* on the clean build (request 1 rather than request 2),
which is what removing noise from an experiment looks like.

The cost of this is not bookkeeping. Long-prompt activation scales with
chunk × length, so a smaller prefill chunk was the cheap way to buy VRAM
headroom for long context — cheaper than giving up hot-expert slots, which is
what pays for decode throughput. That lever is closed, and the next section is
what closing it cost.

---

## Day 2, later — the context we had been advertising and never testing

The server had said `--max-model-len 262144` for days. Nobody had ever sent it a
prompt anywhere near that. When the first genuinely long one arrived, it died:

```
CUDA out of memory. Tried to allocate 116.00 MiB
```

on a 95 GiB card, at 180k tokens. And raising `--max-model-len` to 327,680 did
not help, because the advertised number was never the constraint.

**What is:** long-prompt activations — attention workspaces and the sparse
indexer — are allocated **per request** and scale with sequence length. They are
not reserved up front the way weights and KV are. So the real ceiling is *how
much VRAM is free when the prompt arrives*, and on this stack the only thing
that frees VRAM is hot-expert slots (`gpu-memory-utilization` buys nothing; the
prefill chunk cannot shrink, see above):

| hot slots | free VRAM | longest prompt prefilled | TTFT |
|---|---|---|---|
| 60 | 3,397 MiB | **180k died** (116 MiB short) | — |
| **54** *(shipped)* | 6,177 MiB | **304,800 tokens** — 93% of the advertised ceiling | 205 s (1,487 tok/s) |
| 48 | 8,817 MiB | 233,670 tokens *(largest tried, not a limit)* | 158 s |

Look at the failure size: **116 MiB**. Nothing about a long prompt needs
gigabytes more than a medium one — the margin had simply been squeezed to
nothing by everything else, and long prompts are what finally notices. Every
optimization in this notebook spent VRAM; this is where the bill arrived.

**And the thing I expected to hurt did not.** Cutting slots is cutting the cache
that produced the entire decode win, so 62 → 54 should have cost throughput.
Measured: **48 slots 14.92 tok/s, 54 slots 14.99** — 0.07 apart on a ±1 tok/s
spread, because 48 → 54 is only about three points of cache hit rate. Earlier
entries in this notebook implied the slot trade was more expensive than that;
**that was an over-claim**, and it is worth naming, because it had been quietly
deterring exactly the change that made long context work. The ladder is steep at
the bottom (0 → 66 slots *is* the 13.53 → 16.50 win) and flat at the top, and we
had been reasoning about the top with the bottom's intuition.

Two things shipped out of this:

1. **Never advertise a context you have not prefilled at.** Both earlier
   settings were advertised and untested. An untested `--max-model-len` is a
   promise the server makes on your behalf and breaks in front of a user.
2. **The KV cap now derives from the context** —
   `KV_BYTES = MAX_LEN × 17,200 B` (the measured 15.0–15.7 KiB/token plus ~10%).
   They used to be independent knobs, and the supervisor GUI in front of the
   launcher passes `MAX_LEN` and not `KV_BYTES`, so moving its context slider
   left KV sized for the old context and vLLM refused to start with *"N GiB KV
   cache is needed, larger than the available"* — a message that points at the
   KV cache when the thing that changed was the context. Deriving one from the
   other makes that state unrepresentable, which is a better fix than
   documenting it.

---

## Day 2, evening — the agent that kept forgetting, and the effort knob

Two problems that had nothing to do with throughput and mattered more than any
of it, now that this server is a coding backend rather than a benchmark.

### "It forgets everything after a few tool calls"

Agent sessions looped: re-running their own first step — *"Let me start by
exploring the codebase structure"* — as if the conversation had been erased.
Every instinct points at the client: history truncation, a context window
mismatch, a template dropping turns.

The log said otherwise. **All three `chat/completions` 500s coincide exactly
with an engine death:**

```
01:01:47   CUDA out of memory
01:39:16   CUDA out of memory
02:32:55   Xid 31 illegal memory access
```

83 of 86 agent requests returned 200; the three failures were those three
deaths. A dead turn leaves a failed or empty assistant message in the client's
history, and GLM's chat template renders that as a **fresh thread** — so the
model genuinely re-plans from zero. The "amnesia" is a faithful rendering of a
crash. The fix was the stability work, not anything in the client.

**The order that worked: verify the server first.** One direct request to the
endpoint, outside the agent, answered four questions at once — `finish_reason`
correct (`tool_calls`), `content` empty for the right reason (reasoning came
back in its own field, not leaked into content), `tool_calls` well-formed,
reasoning parser separating cleanly. Server exonerated in a single request, and
everything after that went to the logs instead of to client configuration. The
client is the easier thing to suspect and the much harder thing to disprove;
spend the one request that eliminates the other half of the system.

Two genuine client-side misconfigurations turned up alongside it — both had been
invisible because the crash was louder:

- The editor's model config advertised **`maxInputTokens: 500000`** against a
  server serving 262,144. Corrected to context − max output − margin (227,328 =
  262,144 − 32,768 − 2,048), and it needs recomputing now the context is
  327,680.
- A local proxy that bounds reasoning effort **streamed its retry**. When a turn
  came back answerless it retried — but streamed the retry too, so an answerless
  retry arrived at the client as a *second* blank turn. It was doubling the
  symptom it existed to fix. It now retries **non-streaming**, reuses the same
  answerless guard, and emits an explicit bracketed notice if the retry is still
  empty. Silence is the one response an agent loop cannot recover from.

### `reasoning_effort` is three values, and the default is the expensive one

GLM-5.3's chat template accepts **`low`, `high`, `max`** and nothing else, and it
is a *soft system line*, not a token budget. Anything else — unset, a typo, the
entirely plausible `medium` — **silently becomes `max`**. Nothing warns.

Measured on one hard problem:

```
low    183 reasoning tokens
high   484
max  1,812     ~2x the wall time of high, and a SHORTER final answer
```

- **The thinking budget is not what makes turns blank.** Even `max` spends 1,812
  tokens — 7.2% of a 32,768-token output allowance. Blank turns come from a
  small *per-request* `max_tokens`: at 512, `max` returns an empty turn (the
  budget goes entirely to thinking) while `high` still answers. When an agent
  gets blanks, check the per-request cap before the server config.
- **`high` for agent and tool loops**, where many cheap turns beat a few
  expensive ones. `max` only for one-shot hard questions with a large output
  budget — and expect to pay double the wall time for, on this measurement, less
  answer rather than more.

---

## Operational traps

These cost more wall-clock time than any of the engineering.

### `pgrep -f` / `pkill -f` match your own shell

Hit **three times in one session**. The Bash tool runs `bash -c '<the whole
command>'`, so the command's own text *is* a live command line and `-f` matches
it. The shell kills itself, the call dies with **exit 144**, and the rest of a
launch-and-verify sequence never runs — leaving orphaned servers holding 90 GiB
of VRAM.

The `[p]attern` bracket trick is **not sufficient**: it stops the pattern
matching itself, but the plain string almost always appears elsewhere in the same
command (in the launch line you are about to run, or inside a heredoc you are
writing).

Write the PID to a pidfile at launch and `kill $(cat file)`; or capture `$!`
right after `setsid nohup ... &`; and keep kill and launch in **separate**
invocations.

### Rewriting a shell script while it is running

Bash reads a script lazily **by byte offset**. Truncating and rewriting a file in
place while an instance is executing shifts every offset after the edit point, so
the running shell can resume mid-token and execute garbage — or silently skip the
rest. Adding 40 lines near the top of a 224-line launcher is enough to do it.

This happened to `serve-opt.sh` while a supervisor was relaunching it in a loop;
a vLLM started and died in that window and the edit could not be ruled out as the
cause. Unfalsifiable, self-inflicted noise.

Replace by **atomic rename**: write `path.tmp`, copy the mode, `os.rename()`. The
running bash keeps its open inode and finishes the old version untouched.

### A second CUDA process next to a 125 GiB pinned server

Host RAM went over and the box OOM-killed, once, because a benchmark process was
started alongside the running server. Pinned pages cannot be reclaimed and there
is no swap, so this is a hard kill, not a slowdown. The launcher's host-RAM guard
protects the *launch*; nothing protects you from starting a second CUDA process
afterwards. Don't.

*Correction 2026-09-02:* that guard asked for `offload + 45 GiB`, a figure taken
from a 110 GiB offload that peaked at ~168 GiB of host use. Re-measured on the
shipped configuration, steady state is **offload + ~16 GiB** — 141 GiB used at
offload 125 — because the safetensors reads that inflated the original peak go
through *reclaimable page cache*, which `MemAvailable` already discounts. The
allowance is now **30**, keeping a 14 GiB cushion over the measurement. An
over-conservative guard is not free: it refuses launches that would have been
fine, and the calibration it claims to have is fiction.

### A launcher replaced by `mv` can lose its execute bit

The rule above about replacing a running script by atomic rename has a second
half nobody mentions. Writing the replacement with Python's `open(path, "w")`
creates it **0644**. After the rename the launcher is no longer executable, so
`./serve-opt.sh` silently does nothing: no process, no error worth noticing, an
empty log that looks exactly like a slow start. `chmod +x` the temp file — or
`shutil.copymode(old, new)` — *before* the rename, every time.

### Page-cache eviction of host-resident weights

Reading a 118 GB archive while a model was serving pushed ~50 GB of its host
expert table out; every subsequent lookup faulted from disk and the desktop
froze. Pinned pages cannot swap — but *verify* that they are genuinely pinned
rather than assuming; `cudaHostAlloc`, `cudaHostRegister` and torch
`pin_memory()` differ, and not every model component gets pinned. Avoid bulk file
I/O while serving.

### The Marlin repack looks like a hang

After loading, Marlin repacks the NVFP4 weights with **no log output** for
several minutes at ~96% CPU. It also needs transient VRAM headroom (this is what
makes `--cpu-offload-gb 96` fail with a 6.75 GiB allocation) and transient host
memory (this is the window in which the day-1 duplication appeared).

### Iterate on an 8-layer model, not a 181 GiB one

`--load-format dummy` plus a config truncated to 8 layers exercises the entire
construction / offload / repack / graph-capture path in ~90 seconds instead of
~6 minutes, reading zero bytes of checkpoint. The trap is that truncating
`num_hidden_layers` alone fails —

```
ValueError: `num_hidden_layers` (8) must be equal to the number of `mlp_layer_types` (45)
```

— because every per-layer list must be truncated too. This model has three
(`layer_types`, `mlp_layer_types`, `indexer_types`) and a future revision may add
more, so truncate generically by matching list length against the layer count.
See [OPTIMIZING §8.1](OPTIMIZING-GLM53-ON-ONE-GPU.md#81-truncate-the-model-to-8-layers).

### Measure forward-passes/s, not tok/s

MTP acceptance varies ±5 pp run to run **even at temperature 0**, because the MoE
kernels are not bit-deterministic (atomics in reductions), so generations diverge
and acceptance with them. Token rate inherits that noise;
`throughput = passes/s × tokens accepted per pass`, and passes/s is the stable
half. Derive it from `/metrics` deltas across the benchmark window, never
lifetime totals.

Related: streaming **chunks are not tokens** (with MTP a chunk carries ~2–4 —
count `usage.completion_tokens`), and *"count from 1 to 150"* gives ~100%
acceptance and flatters throughput by ~50%. Prose is ~55%, code ~66%.

---

## Open failures

Honesty about what is still broken:

- ~~**The engine can die on a later request.** A run in the shipped
  configuration served eight 200-token generations cleanly and then died on the
  ninth with `CUDA error: an illegal memory access` inside the MTP draft path
  (`speculator.propose` → `_build_draft_attn_metadata` →
  `flashinfer_mla_sparse_sm90.py:400 build` → `_kv_lens_host`). Same file as the
  `MAX_BATCHED < 8192` failures, so the two are plausibly one bug; not
  root-caused.~~
  **Resolved well enough to ship, 2026-09-02.** The two *were* one bug, and it
  was neither the batch size nor that file: an Xid 31 MMU fault, addressed by
  staging bounds checks plus `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
  ([above](#day-2-later--the-crash-was-not-what-we-had-written-down)). Evidence
  is two clean 40-request soaks against faults on every breakable-graph build
  the same day. **The mechanism is unproven** and the fix has not been exercised
  over days of continuous serving, so this moves from "open failure" to
  "watched".
- **Fused-`w13` scale collapse is unfixed** and affects every modelopt NVFP4 MoE
  checkpoint: `gate_proj.weight_scale_2` and `up_proj.weight_scale_2` are
  quantized independently — 179 of 288 experts differ by ~1.4x — and vLLM fuses
  them keeping only the gate scale, mis-scaling the up-projection.
- **The non-UVA offload fallback is broken.**
  `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` dies with an illegal memory access and
  leaves a **D-state thread holding all VRAM** — only a reboot clears it.
- **`--max-num-batched-tokens < 8192` is a landmine**, and still not
  understood. ~~Withdrawn~~ as an artifact of the Xid above (the same fault hit
  at 8192) and then **re-confirmed** on the fixed build, where 4096 dies on the
  *first* request. Two independent failures; this one is unfixed, and it is the
  reason long-context headroom has to come out of hot-expert slots.
- **The hot-expert cache is still invisible to vLLM's memory accounting.** It is
  allocated after profiling, so KV is sized against VRAM that is not free, and
  serving can still die with `CUDA out of memory. Tried to allocate 124.00 MiB`.
  Unfixed — the principled fix broke serving and was reverted — and the only
  lever is fewer hot slots, which is why the shipped count has walked
  68 → 66 → 62 → 54.
- **A withdrawn optimization is still owed a verdict.** The masked-group prefill
  variant (stage only the experts a chunk routes to) measured **0.36 s TTFT on
  the 14-token control**, against 0.85 s for the shipped zero-copy path, and was
  removed on a misattributed nondeterminism claim. Re-test it with the
  fresh-prompt logprob method; if it holds, the ≥ 48-token gate goes away.
  **Untested, not shipped.**
- **The native SM120 sparse-MLA backend cannot be used at all** (NoPE shapes);
  we run the SM90 kernel on Blackwell by fallback.
- **The long-context ceiling is a measured point, not a guarantee.** 304,800
  tokens prefilled at 54 hot slots with 6.2 GiB free. Anything that consumes
  VRAM — a second CUDA process, more slots, concurrency above
  `--max-num-seqs 1` — moves that line, and it moves *silently* until a prompt
  dies.

---

## Lessons, ranked by what they cost

1. **Benchmark the primitive before building machinery on it.** A five-minute
   `cudaMemcpyAsync` timing would have skipped an entire offloader patch, a 2.13x
   host-RSS regression, and an OOM that froze the machine. Every *component* of
   that patch verified; the premise did not.
2. **Check what your aggregation is hiding.** Summing per-expert counts across
   layers said "routing is near-uniform, caching is capped at ~1.15x". The same
   raw data, grouped per layer, is the entire 13.5 → 16.5 tok/s win. One
   `groupby` was the difference between a closed branch and the result.
3. **On this stack, wrong answers parse.** A mis-keyed profile cost 19 points of
   hit rate; an offload-mismatched profile cost 10% of throughput; a naive
   `expert_map` scatter would read the wrong weights; the default MoE kernel
   returns garbage. None of them raise. Build the guard *and* the warning at the
   same time as the feature.
4. **Profile before you theorise about the second-largest term.** The arithmetic
   correctly identified PCIe as the bottleneck and then invited a whole day of
   attention-backend speculation. Attention is 0.47 ms — 0.1%. The real number
   two was 18 GiB of BF16 weights nobody had thought about.
5. **Benchmark the thing the user waits for.** Two days of decode benchmarks
   never showed that a 10k-token prompt cost 16.8 seconds before its first
   token — the term that dominates every real coding turn. One streamed request
   with TTFT separated from the inter-token rate exposed it, and fixing it was
   the largest quality-neutral win of the project (2.9x). The corollary cost an
   hour in the other direction: a TTFT regression on short prompts was read as a
   decode regression because the benchmark reported one number.
6. **An equivalence claim needs a control for the noise.** "Outputs identical at
   T=0" was reported as bit-identity, twice, on a stack whose *baseline* gives
   different output run to run. The same invalid test then got a good patch
   reverted. Measure the candidate against the baseline's own drift, or do not
   claim equivalence.
7. **A stack trace is where an error was noticed.** CUDA says so in the message
   itself, and we still spent a day suspecting the attention backend that
   happened to be running when an asynchronous MMU fault surfaced. The kernel
   log had `Xid 31` the whole time.
8. **A confound invalidates an attribution, not a phenomenon.** Finding that the
   crash also happened at 8192 correctly killed *"8192 is safe"*. It was then
   over-read as *"batch size never mattered"*, and the original experiment was
   never re-run once the confound was fixed. It took ten minutes when someone
   finally did, and the floor was real. Re-test after you remove a confound.
9. **Verify the server before you debug the client.** Sessions that appeared to
   lose their history were the engine crashing mid-turn; a single direct request
   to the endpoint would have ruled the server in or out in seconds. Test the
   half of the system you can interrogate directly, first.
10. **Do not advertise a capability you have not exercised.** A context length
   the server had never been asked to prefill was, in effect, an untested claim
   shipped to users — and the thing that actually limited it (free VRAM) was not
   the thing being configured.
11. **Negative results are the deliverable.** Of the twenty-odd things measured
   in this notebook, five helped. The rest are what makes the five trustworthy —
   and are the only reason the next person does not have to spend the same two
   days finding out.
