# What is actually left — GLM-5.3-Flash on one RTX PRO 6000

Research companion to [`RUNNING-GLM53-ON-ONE-GPU.md`](RUNNING-GLM53-ON-ONE-GPU.md)
(how to make it run) and
[`OPTIMIZING-GLM53-ON-ONE-GPU.md`](OPTIMIZING-GLM53-ON-ONE-GPU.md) (what we tried
and what failed). This document asks one question: **given hardware we cannot
change, what raises decode throughput, and by how much?**

Every estimate below uses constants measured on this machine, not published
numbers from other rigs.

> ### Revision note — 2026-09-02: scorecard
>
> This was written on 2026-09-01, when the best measured figure was 11.9 tok/s
> and its five recommendations were all unbuilt. Four of the five have since
> been settled by measurement. **The original text is preserved**, with a
> verdict box on each section, because the *prediction versus outcome* is the
> point.
>
> | § | recommendation (2026-09-01) | predicted | **verdict (2026-09-02)** |
> |---|---|---|---|
> | 1 | GPUDirect Storage as an additive path | dead | **DEAD — unchanged** |
> | 2 | a real gather kernel (+ CUDA graphs) | 14.8 tok/s | **DONE — 13.53, and the relative gain was right** |
> | 3 | hot-set pinning at 48% hit | 17.9 tok/s | **DONE — 61.5% hit, 16.5 tok/s** |
> | 4 | mixed-precision (2-bit) cold experts | 25.4 tok/s | **still untried; the estimate was ~4 tok/s optimistic** |
> | 5 | FreeToken hybrid CPU co-execution | 19.3 tok/s | **DEAD — measured 0.895x** |
>
> Two of the three biggest errors in the original are worth naming up front:
> - It treated `T_other` as ~8 ms once CUDA graphs were restored. **The measured
>   fixed cost is 34.8 ms/token.** Every projection built on the 8 ms figure is
>   an over-estimate, §4's included.
> - It assumed the winning gather kernel would have to be *ported from
>   FreeToken*. It did not; a ~40-line Triton kernel beat the ported target's
>   published rate on our shapes (52.1 vs 51.8 GB/s) and, more importantly, was
>   graph-capturable, which is where most of the gain actually came from.
>
> ### Second scorecard — later on 2026-09-02
>
> Four more branches closed after the table above was written, and **the biggest
> win of the day is not on it at all**, because this document only ever asked
> about decode:
>
> | branch | predicted / assumed | **verdict** |
> |---|---|---|
> | **prefill** | never asked about | **the largest quality-neutral win: 614 → 1729–1771 tok/s**, 10k TTFT 16.8 s → ~5.9 s, logged in `artifacts/prefill-fresh-2026-09-02.txt` ([OPTIMIZING §10](OPTIMIZING-GLM53-ON-ONE-GPU.md#10-prefill--the-term-the-decode-benchmarks-hid)) |
> | §6 n-gram stacked on MTP | "+10–25% on code traffic, low effort, worth trying" | **DEAD — both deeper-draft routes are blocked** (§6) |
> | re-profile the hot set on real mixed traffic | implied by §3's "unfinished business" | **DEAD — worse on everything**, 13.02 vs 15.77 prose (§3) |
> | native SM120 sparse-MLA attention backend | never asked about | **BLOCKED — NoPE shapes rejected at startup** ([OPTIMIZING §11.3](OPTIMIZING-GLM53-ON-ONE-GPU.md#113-two-adjacent-branches-that-closed)) |
>
> And one methodological correction that outranks any of them: the "bit-identical
> at T=0" evidence used for the prefill path was **invalid**, because baseline
> prefill on this stack is itself non-deterministic run to run. The replacement
> test — per-token prompt logprobs on fresh prompts, scored against the
> baseline's own drift — is in
> [OPTIMIZING §10.6](OPTIMIZING-GLM53-ON-ONE-GPU.md#106-correctness-the-claim-that-had-to-be-withdrawn).
>
> ### Third scorecard — later still on 2026-09-02
>
> | branch | assumed | **verdict** |
> |---|---|---|
> | **long context** | assumed to work because `--max-model-len` said so | **it did not.** The usable ceiling is **free VRAM**: a 180k prompt died with 3.4 GiB free; 304,800 tokens prefill with 6.2 GiB free. Context is now 327,680 with the KV cap derived from it ([OPTIMIZING §12](OPTIMIZING-GLM53-ON-ONE-GPU.md#12-long-context--the-ceiling-is-free-vram-not---max-model-len)) |
> | **hot slots as a throughput lever, at the top of the ladder** | cutting slots costs throughput | **over-claimed.** 48 → 14.92 tok/s, 54 → 14.99, inside the ±1 tok/s spread. The top of the hit-rate ladder is flat, so long-context headroom was nearly free |
> | **shrink `--max-num-batched-tokens` to buy VRAM** | the cheap alternative to giving up slots | **DEAD.** 4096 kills the engine on the first request even after the Xid 31 fix — a second, independent failure ([OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all)) |
> | **agent sessions "losing context"** | client truncation or a template bug | **it was the engine dying mid-turn.** All three `chat/completions` 500s coincide exactly with engine deaths ([OPTIMIZING §13.1](OPTIMIZING-GLM53-ON-ONE-GPU.md#131-the-model-forgets-its-context-was-the-engine-dying)) |
>
> This document ranks ways to raise decode tok/s. Not one of those four is a
> decode-throughput question, and together they mattered more to the deployment
> than anything in the ranking below.

---

## 0. The measured cost model

| constant | measured | how |
|---|---|---|
| expert bytes touched per token | 4.43 GiB | 42 MoE layers × top-8 of 288 × ~13.5 MiB |
| PCIe linear H2D | 56.4 GB/s | `ft bench bw` |
| PCIe realistic nvfp4 gather (FreeToken's kernel) | 51.8 GB/s | `ft bench bw` |
| **our Triton gather kernel** | **52.1 GB/s** | our microbenchmark, 4 MiB rows |
| our per-row `copy_` gather | 41.4 – 44.9 GB/s | our microbenchmark |
| PyTorch `index_select` gather | 10.8 – 23.1 GB/s | our microbenchmark, row-size dependent |
| CPU STREAM read | **64.6 GB/s** | `ft bench bw`, **dual-channel** DDR5 |
| VRAM bandwidth | ~1800 GB/s | vendor figure, not measured here |
| async copy engines | 2 | `cudaDeviceProp.asyncEngineCount` |
| ~~current best~~ | ~~11.9 tok/s~~ → **16.50 tok/s** | staging + pinning + MTP-2 + CUDA graphs |

### The fixed-cost term, which the original version of this document got wrong

The original decomposed 84 ms/token as `3.08 GB / 44.9 GB/s + 15.4 ms`, and
assumed the 15.4 ms residual would fall to ~8 ms once CUDA graphs were restored.
It does not. Varying the offload and fitting a line gives:

```
offload 102 GiB (63% of experts remote)  -> 78.3 ms/token
offload 125 GiB (77% remote)             -> 87.6 ms/token

linear fit:  T_token = 34.8 ms  +  68.6 ms x traffic_fraction
```

**34.8 ms is fixed and does not go away**, which is why the ceiling with zero
PCIe traffic is ~29 tok/s rather than the ~100 the original model implied.

Sanity-checking the fit against the shipped configuration: traffic fraction is
`125/163.27 × (1 − 0.6146) = 0.295`, giving a predicted **55.0 ms/token ≈ 18.2
tok/s**. We measure **60.6 ms/token = 16.5 tok/s**. The ~5 ms/token gap is
unexplained; candidates are the hot cache's own cost (the per-row branch and the
VRAM reads it adds), the different context length (the fit was taken at 512k, the
result at 262k), and MTP acceptance variance. Small enough that the model is
still useful for **ranking** options; large enough that every projection below
should be read as an **upper bound**.

**Everything worth doing attacks one of three terms:** raise `BW_gather`, shrink
`B_host`, or cut the 34.8 ms. Nothing else matters.

> **Scope correction (2026-09-02, later).** That sentence is true *of decode*,
> and this whole document is a decode document. It is not true of the workload.
> A prefill chunk routes to ~all 288 experts, which exceeds the staging cap, so
> prefill was running on zero-copy BAR reads at ~23 GB/s and cost **16.8 s
> before the first token of a 10k-token prompt** — a term with no place in the
> model above and a bigger effect on a real coding turn than anything in §§1–5.
> Fixed by grouping ([OPTIMIZING §10](OPTIMIZING-GLM53-ON-ONE-GPU.md#10-prefill--the-term-the-decode-benchmarks-hid)):
> 614 → ~1750 tok/s. **A cost model is only as good as the phases it was built
> from.**

---

## Table of contents

1. [DEAD: NVMe/GPUDirect Storage as an additive path](#1-dead-nvmegpudirect-storage-as-an-additive-path)
2. [DONE: a real gather kernel](#2-done-a-real-gather-kernel)
3. [DONE: hot-set pinning instead of arbitrary residency](#3-done-hot-set-pinning-instead-of-arbitrary-residency)
4. [UNTRIED: mixed-precision cold experts](#4-untried-mixed-precision-cold-experts)
5. [DEAD: CPU co-execution (FreeToken hybrid)](#5-dead-cpu-co-execution-freetoken-hybrid)
6. [Speculative decoding angles](#6-speculative-decoding-angles)
7. [NEW: the two levers that cost quality](#7-new-the-two-levers-that-cost-quality)
8. [Things that die on our numbers](#8-things-that-die-on-our-numbers)
9. [Where it ended up](#9-where-it-ended-up)

---

## 1. DEAD: NVMe/GPUDirect Storage as an additive path

> **Verdict 2026-09-02: unchanged. Still dead.**

**The idea:** the NVMe sits on its own PCIe lanes, so stream cold experts
NVMe→GPU via GPUDirect Storage *concurrently* with host RAM→GPU, and the two
bandwidths add.

**The topology is favourable.** Verified with `lspci -tv` and sysfs:

```
+-01.1-[01]--+-00.0  NVIDIA RTX PRO 6000       <- CPU root port 00:01.1, Gen5 x16
+-01.2-[02]----00.0  Samsung NVMe 9100 PRO     <- CPU root port 00:01.2, Gen5 x4
+-02.1-[03-14]-- ... 600-series chipset switch <- everything else lives here
```

Both devices are on **direct CPU lanes on separate root ports** — not behind the
chipset, so they do *not* share the chipset's x4 uplink. This is the best case
the AM5 platform can offer, and the idea still dies.

**Why it dies:** the two paths converge on the **GPU's own x16 ingress**, and we
are already at 92% of it.

```
GPU ingress ceiling (measured linear H2D)   56.4 GB/s
already achieved by our gather kernel       52.1 GB/s   (92%)
headroom for ANY additional source           4.3 GB/s
```

Adding a second *source* does not add ingress bandwidth. Bytes from NVMe and
bytes from DRAM contend for the same x16 link into the GPU. The NVMe's x4 lanes
are upstream of the bottleneck, not parallel to it.

The one place it retained value was inside the *hybrid* configuration (§5), where
GDS reads bypass DRAM and would free DRAM bandwidth for a CPU-side MoE path. §5
is now dead on independent grounds, so this residual value is gone too.

**Verdict: not worth it.** Effort is high (GDS needs the `nvidia-fs` kernel
module, which is *not installed* here — no `libcufile`, no `modinfo nvidia-fs`).

*Falsified if:* someone measures simultaneous GDS + host-DMA ingress on this card
exceeding ~57 GB/s aggregate. That would mean the x16 ingress figure is wrong,
and the whole analysis reopens.

---

## 2. DONE: a real gather kernel

> **Verdict 2026-09-02: BUILT, and the reasoning held.** The predicted mechanism
> — that restoring CUDA graphs would be worth more than the bandwidth — was
> correct. What was wrong was the plan: we did not port FreeToken's kernel.
>
> | | predicted | measured |
> |---|---|---|
> | gather bandwidth | 51.8 GB/s | **52.1 GB/s** (our Triton kernel, 4 MiB rows) |
> | relative gain from kernel + graphs | +24% | **+25.6%** (10.77 → 13.53) |
> | absolute result | 14.8 tok/s | 13.53 tok/s (from a lower baseline) |
> | value of restoring CUDA graphs alone | "worth more than the bandwidth" | **+13%** (11.77 eager → 13.53 captured) |

*(Original text, 2026-09-01:)*

We concluded "UVA memory isn't slow, the *gather kernel* is" after measuring
`index_select` at 10.8 GB/s versus per-row `copy_` at 44.9. That was right as far
as it went — but **44.9 is not the ceiling either.** FreeToken's purpose-built
gather kernel achieves **51.8 GB/s** on the identical hardware and identical
access pattern. Their kernel does what a generic path cannot: 16-byte vectorized
access, non-temporal loads/stores so streamed weights do not evict live data from
L2, worker-thread partitioning so the copy is coalesced regardless of how
scattered the *indices* are, and a sync-flag gate allowing the copy to be
pipelined against compute.

### Two wins, not one

**(a) +15% bandwidth.** 3.08 GB at 51.8 instead of 44.9 GB/s: 68.6 → 59.5 ms.

**(b) It restores CUDA graphs — worth more than the bandwidth.** Our staging
takes device-side indices to the host (`experts.tolist()`) to drive per-row
copies. That host sync is illegal during CUDA graph capture, which is why we had
to run `--enforce-eager` and eat the launch overhead of an entire eager forward
pass. **A kernel that consumes indices as a device tensor needs no sync and is
capturable.**

*(What actually happened:)*

We wrote our own Triton kernel instead of porting theirs, for three reasons that
turned out to matter more than the vectorized-access techniques above:

1. **Deduplication.** `topk_ids` repeats experts across the tokens verified in
   one MTP step. `torch.unique` has a data-dependent output shape and graph
   capture rejects it, so the kernel skips any row whose expert appeared at a
   lower row and the grid stays statically shaped.
2. **The hot-expert cache had to live inside the same kernel** (§3), choosing per
   row between a VRAM slot and a PCIe fetch via a device-resident slot table.
   Bolting that onto a ported kernel would have been harder than writing one.
3. **Block size is the whole tuning surface.** `BLOCK = 8192` measured fastest;
   4096 ties; ≥16384 falls to ~49 GB/s. That is the entire optimization, and it
   got us past the ported target's published figure on our shapes.

The prediction that our smaller experts might not reach 51.8 GB/s ("verify at
*our* expert size before committing") was the right caution and turned out not to
bite.

---

## 3. DONE: hot-set pinning instead of arbitrary residency

> **Verdict 2026-09-02: BUILT, and the estimate was conservative on hit rate and
> optimistic on throughput.**
>
> | | predicted | measured |
> |---|---|---|
> | hit rate at ~35% capacity | 48% (band 48–55%) | **74.3%** |
> | hit rate at the shipped 23% capacity | — | **61.46%** |
> | throughput | 17.9 tok/s | **16.50 tok/s** |
>
> The literature-derived skew factor (1.70x over uniform, from DeepSeek) was
> **too pessimistic** for this model measured per layer: at 25% capacity we see
> 2.56x uniform, not 1.70x. The throughput shortfall is the fixed-cost error in
> §0, not the hit rate.
>
> **The falsification test proposed below was never run as specified**, and that
> gap is still open — see *Unfinished business* at the end of this section.

*(Original text, 2026-09-01:)*

**The idea:** vLLM's offloader picks which experts stay in VRAM by *module
iteration order* — i.e. arbitrarily. 35% of experts are resident and they serve
35% of requests. If instead we pin the 35% most-frequently-*used* experts, the
same VRAM serves a much larger share.

**This is a different question from the LRU test we ran tonight.** Our LRU cache
measured **temporal** locality (does the next token reuse the last token's
experts?) and found it weak: 9% hit at 5.6% capacity. **Static frequency skew** is
a separate property and the literature says it is real.

- **DeepSeek at a 25% expert budget captures 42.6% of per-layer token
  activations** — 1.70x better than uniform
  ([Mixture of Neuron Experts](https://arxiv.org/pdf/2510.05781)).
- Routers "naturally converge toward a skewed, winner-take-most configuration"
  with "a long tail of under-utilized experts"
  ([MoE-Beyond](https://arxiv.org/pdf/2508.17137)).
- **The critical caveat:** MoE-Infinity finds activation "patterns are skewed
  within single requests" but "become **more uniform across multiple requests**"
  ([MoE-Infinity](https://arxiv.org/html/2401.14361v3)). A global hot set profiled
  over a broad corpus washes out.

*(What actually happened:)*

Measured per layer on 12,000 staged steps of real traffic, at
`cpu_offload_gb=125` (33 offloaded layers × 288 experts):

```
 33 slots/layer -> 42.65%      132 -> 82.99%
 66 slots/layer -> 61.46%      165 -> 89.62%
 99 slots/layer -> 73.98%
```

The hottest ~36% of a layer's experts serve 75% of that layer's fetches. Shipped
at 66 slots (23% capacity, 61.46% hit, ~29 GiB of VRAM), which is where the VRAM
wall is: 68 slots boots and OOMs mid-serving, 76 and 96 OOM at startup.

**The single most dangerous property of this technique** is that the profile must
be recorded at the same `cpu_offload_gb` it will be used at — residency is a
whole-layer decision, so a different offload means a different set of host-side
layers and the layer indices stop lining up. A 125-profile run at 127 measured
**15.09 vs 16.76 tok/s** with no error of any kind. Both the launcher and the
pinner now check and warn.

**And the measurement that nearly buried it:** an earlier look at the *aggregate*
(summed-across-layers) histogram concluded routing was near-uniform and that
caching was capped at ~1.15x. Summing across layers washes the per-layer skew
out, because different layers are hot on different experts. Same data, grouped
per layer, is this entire section. See
[OPTIMIZING §4.2](OPTIMIZING-GLM53-ON-ONE-GPU.md#42-the-skew-is-per-layer-and-the-aggregate-hides-it).

### Unfinished business

The original proposed a cheap decisive test:

> *Dump routing counts for two different real workloads and compute the overlap
> of their top-35% sets. High overlap (>70%) means pin globally. Low overlap
> means this only works per-workload, or not at all.*

~~**That test was never run.**~~ **Run 2026-09-02, and the answer is
uncomfortable: low overlap, and the "broader" profile is worse.**

A profile captured over a mixed workload (prose/code/chat/long/math at
T=0/0.7/1.0, plus a coding prompt and a 30k prompt) at offload 130 shares only
**44% of its per-layer top-64 set** with the shipped one-prose-prompt profile —
well below the >70% that would have said "pin globally". Its decode-only hit
estimate at 66 slots was ~50% against the shipped profile's 61.5%.

Served, it was worse on everything except VRAM margin:

| | mixed profile @ offload 130, 64 slots | shipped @ 125 |
|---|---|---|
| prose decode | 13.02 tok/s | **15.77** |
| coding prompt | 16.10 tok/s | **17.10** |
| prefill | 1698 tok/s | **1762** (the figure current at the time; see the logged 1729–1771 range) |
| VRAM margin | +4 GiB | — |

So the MoE-Infinity caveat is **confirmed in direction and inverted in
prescription**: hot sets really do differ across workloads, but the fix is *not*
"capture on a broader mix". Two thirds of that mix ran at T=0.7 or T=1.0, and
sampling randomizes tokens, which **flattens routing** — the profile records a
blurred version of the very skew it exists to capture. Five more GiB of experts
host-side does the rest.

**Re-record the profile on your own traffic — at T=0.** And note that a profile
captured with grouped prefill enabled carries a uniform +1 per expert per chunk
(ranking preserved, raw top-k shares ~17% low); subtract the per-layer minimum
first. The failure mode throughout is a quiet loss of throughput rather than an
error. Cost of this experiment: two 6-minute launches and 15 minutes of driving.

Also untested: whether **variable per-layer slot budgets** help. They do not —
water-filling the same total slots across layers gains **+0.07 pp**, because the
layers' distribution *shapes* are alike even though their hot *sets* differ.

---

## 4. UNTRIED: mixed-precision cold experts

> **Verdict 2026-09-02: still untried, and the estimate below is roughly 4–5
> tok/s optimistic** because it used `T_other = 8 ms` instead of the measured
> 34.8 ms fixed cost. Redone with the measured fit and the shipped hit rate:
>
> ```
> shipped:  34.8 + 68.6 x 0.295            = 55.0 ms  (predicted; 60.6 measured)
> with a 1.9x smaller cold format:
>           34.8 + 68.6 x 0.295 / 1.9      = 45.5 ms  ->  ~22 tok/s predicted
> ```
>
> So it is worth roughly **+20%**, not the +113% the original claimed — and it
> stacks on top of pinning rather than replacing it. Still the largest remaining
> lever on `B_host`, but no longer a route to 25–30 tok/s on its own.

**The idea:** hot experts stay NVFP4. Cold experts — the ones that must cross
PCIe — are stored at 2–3 bits. Fewer bytes per miss is a direct, linear cut to
the dominant term.

This is not speculative; it is a published technique. **HOBBIT** substitutes INT4
for FP16 experts and INT2 for INT8 experts specifically on cache miss, "reducing
loading latency without significant accuracy loss". **Mixture Compressor** for
Mixtral-8x7B allocates 4-bit to the top-8 most-used experts and 2-bit to the
remaining 8 (a 3.03-bit average), with a 2.54-bit variant giving hot experts
3-bit and the rest 2-bit ([Mixture Compressor](https://arxiv.org/pdf/2410.06270)).
The standard recipe classifies "the top 20–30% of experts by selection frequency
as hot and the rest as cold" — which is exactly the split §3 now computes for
free.

### Why it is still ranked below everything already built

- **It needs a kernel.** Marlin's NVFP4 MoE path has no mixed-bit-width variant.
  Dequantizing 2-bit cold experts into the 4-bit compute path means writing and
  validating a new dequant kernel — and bug 7 in the runbook is a standing
  reminder that a wrong MoE kernel on this hardware returns *plausible garbage
  with no error*.
- **It needs a calibration pass** over the checkpoint, then a re-quantized
  181 GiB artifact.
- **Quality is a real risk.** The literature is explicit that "uniformly reducing
  expert precision below 4 bits leads to non-negligible accuracy loss", and that
  experts differ sharply in quantization sensitivity. Selective application to
  rarely-used experts is the mitigation, but it must be measured — and this model
  already carries an unfixed `w13` scale-collapse defect that muddies any quality
  comparison.
- **It is now in direct competition with §7's FP8 dense weights**, which needs no
  new MoE kernel and attacks a term the profiler says is comparable in size.

*Falsified if:* a perplexity/benchmark check on the 2-bit-cold checkpoint shows
material regression versus the NVFP4 baseline.

---

## 5. DEAD: CPU co-execution (FreeToken hybrid)

> **Verdict 2026-09-02: measured and dead.** Not on GLM-5.3 — porting it there
> was the expensive part — but on **Qwen3.8-Flash-Next NVFP4, a checkpoint
> FreeToken already supports**, forcing each backend:
>
> ```
> ft serve --moe-backend offload   61.62 tok/s
> ft serve --moe-backend hybrid    55.15 tok/s   ->  0.895x
> ```
>
> **A 10% regression, not a 1.36x gain.** Hybrid computes host-resident experts
> on the CPU instead of fetching them; it pays only when CPU memory bandwidth
> clearly beats PCIe. Do **not** port GLM-5.3-Flash into FreeToken to get it.
>
> (Also worth noting from the same run: FreeToken's own offload path at 61.6
> tok/s is far below our vLLM fully-resident 173 tok/s for the same model.
> Offload is a last resort, never a speedup.)

*(Original text and reasoning, which was directionally right about the platform
and wrong about the payoff:)*

```
CPU-MoE 53.7 GB/s  vs  PCIe-gather 51.8 GB/s  ->  CPU/PCIe = 1.04x
FreeToken's own threshold to select hybrid    ->  2.0x
```

Their policy therefore picks plain `offload` on this box. The gain looked
non-zero because the paths overlap: 35.0 + 35.2 = 70.2 GB/s aggregate versus 51.8
alone, i.e. **1.36x** on the transfer term — but the overlap measurement also
shows *both* paths collapsing when run together, because they share DRAM, and in
practice the coordination cost exceeds what is left.

**The reason their published GLM-5.2 number (14.9 tok/s, same GPU) looked
unreachable to us is hardware:** their rig is a Xeon Platinum 8559C with
178 GB/s host bandwidth. Ours is a dual-channel desktop at 64.6 GB/s. Their CPU
path is ~2.8x ours. This is a platform gap, not a software one — and it is why
16.5 tok/s on this box is the interesting number rather than the raw figure.

A `glm5_next` config scaffold (args/config/registration, verified against the
real checkpoint) sits in `FreeToken/python/freetoken/models/glm5_next` if a
higher-bandwidth machine ever appears.

---

## 6. Speculative decoding angles

> **Verdict 2026-09-02: MTP-2 shipped and is worth ~+31%. ~~N-gram stacking
> remains untried.~~ N-gram stacking is now DEAD — measured and blocked.**
>
> Measured on the shipped config, coding prompt, T=0: **MTP-2 = 17.10 tok/s**,
> cumulative acceptance **64% of draft tokens = 1.28 accepted per step**
> (position 0: 75%, position 1: 53%), i.e. 2.28 tokens per forward pass. A
> second coding run at offload 130 measured 1.81 accepted per step on its own
> generation — acceptance is generation-specific, so quote it with the prompt.
>
> `ngram`, `ngram_gpu` and `suffix` all exist in this fork and the launcher
> takes `SPEC_METHOD` / `SPEC_EXTRA`, so the "low effort, just a config flag"
> estimate below was right about the effort. It was wrong about the outcome:
>
> - `num_speculative_tokens=5` needs **4.27 GiB of KV per request against the
>   4 GiB cap** in force at the time — startup `ValueError`. (The cap now
>   derives from the context and is 5.25 GiB, which is larger than 4.27;
>   **not retested**, and the acceptance argument below says it would not buy
>   much anyway.) Fundable from hot slots, if it were
>   worth it.
> - `num_speculative_tokens=4` (5 query tokens/step) dies in the
>   sparse-attention KPOOL indexer: DeepGEMM `csrc/apis/attention.hpp:320`,
>   `DG_HOST_ASSERT` on `block_kv` for arch 12, via `sparse_attn_indexer_kpool →
>   fp8_fp4_paged_mqa_logits`. MTP-2's 3 query tokens pass. Whether
>   `--block-size 256` (the knob from
>   [OPTIMIZING §7.2](OPTIMIZING-GLM53-ON-ONE-GPU.md#72-the-block_kv-constraint-and-why-mtp-depth-4-tripped-it))
>   clears it at that depth on the current stack was **not retested**.
> - At depth 2 a free drafter can only save the one MTP layer — ~2–3% of a step
>   — and prompt-lookup acceptance on generic code is below a trained MTP
>   head's. The upside was never large.
>
> **MTP-2 stays.** Do not spend launches on n-gram unless the indexer
> constraint lifts.

Decode is bandwidth-bound, so any scheme that yields more accepted tokens per
expert-fetch is a direct multiplier. But the MTP-4 experiment showed GLM's
specific failure mode: it re-runs *the same* MTP layer per draft step, so
acceptance collapsed 60.2% → 33.7% and throughput fell 19%.

Options that avoid a second weight set:

- **Lookahead decoding** — breaks the sequential dependency via Jacobi iteration,
  "functions without the need for a draft model or a data store"
  ([LMSYS](https://www.lmsys.org/blog/2023-11-21-lookahead-decoding/)). Zero extra
  VRAM, which matters here.
- **N-gram / prompt-lookup decoding** — best for "repetitive, input-grounded
  tasks". For agentic coding, where the model frequently reproduces spans from
  context, this is a good fit and is already in vLLM
  (`--speculative-config '{"method":"ngram",...}'`).

We already get 2.20 tokens/pass from MTP-2 at ~60% acceptance, so most of the
generic upside is banked. The remaining opportunity is **stacking n-gram lookup
on top of MTP for code workloads**.

**Important interaction:** with *staging*, deeper speculation fetches the
**union** of experts across verified tokens, so bytes grow with draft width even
when acceptance holds. The stager's own announce line reports **32 experts
staged** on its first captured multi-token step, against the 8 a single-token
decode would need. Any spec-decode change must therefore be measured as tok/s,
never as acceptance rate.

Estimated gain: **+10–25% on code-like traffic**, ~0 on prose. Low effort (a
config flag), so worth trying, but measure it.

---

## 7. NEW: the two levers that cost quality

Neither of these existed in the 2026-09-01 version, because the profiler run that
motivated the first and the measurement that killed the second both happened on
2026-09-02. They are the only two remaining changes with a large expected gain,
and **both change model outputs**.

### 7.1 FP8 for the 18.01 GiB of BF16 non-expert weights

The profiler says **18.98 ms of every decode step** is dense BF16 GEMM — the
non-expert weights the NVFP4 quantizer left untouched (attention projections
11.74 GiB, shared expert 2.02, dense MLP 1.58, embedding 1.18, lm_head 1.18).
Halving those bytes is worth ~5–6 ms/token.

Three findings bound it:

1. **"FP8 is free because FP8 is the native checkpoint format" is refuted by
   direct measurement on our own tensors.** The BF16 dense tensors are *not*
   FP8-representable — round-tripping through e4m3 has non-zero error — so they
   are genuine BF16 masters, not upcast FP8. Properly scaled FP8 (per-tensor
   **and** per-channel) gives **~31 dB SNR**, the expected figure for e4m3's
   three mantissa bits. A real, if mild, loss.
2. **No pre-quantized checkpoint fits.** A full-FP8 GLM-5.3-Flash is ~331 GB
   against 182 GiB host + 96 GiB VRAM ≈ 278 GB total. It cannot fit *at all* —
   ruling out `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` and
   `orcarouter/...-Uncensored-FP8` on capacity alone.
3. **No mixed variant exists.** `LibertAIDAI/GLM-5.3-Flash-NVFP4` is the same
   layout as ours (NVFP4 experts, everything else BF16, ~181 GiB). There is no
   FP4-experts + FP8-dense build to download.

So this is a quantize-it-yourself job with a measured quality cost. It is the
largest untried lever and it is not free.

### 7.2 Router pruning — +24%, and rejected

GLM routes top-8. The router's weight by rank, averaged over 3000 steps of real
traffic:

```
[0.269 0.186 0.137 0.109 0.091 0.078 0.069 0.061]
cumulative: [0.269 0.455 0.592 0.701 0.792 0.870 0.939 1.000]
```

Fetching only the top-6 measured **20.48 tok/s against 16.50 — +24%**. It is
implemented (`VLLM_MOE_DECODE_TOPK`), it selects by weight with `torch.topk` and
renormalizes (GLM sets `norm_topk_prob=True`) rather than slicing `[:, :k]`, and
its static output shape keeps CUDA graphs capturing.

**It is not shipped, and will not be.** Output visibly degrades: repetition loops
(*"Jupiter (planet) (planet) (planet)"*) and explanations drifting into word
salad. Simple facts survived (Paris; 17×23=391), which is exactly what makes it
dangerous on a spot check. The standing constraint on this deployment is that it
is used for critical work and must not trade quality for speed.

Going further down is worse and not even fast: from the weight-by-rank table,
top-4 discards 30% of the routed weight and still projects only ~17.7 tok/s.

**Consequence: the quality-neutral ceiling on this hardware is ~16–17 tok/s.**
25 tok/s requires stacking §7.1 and §7.2, both of which change model outputs.

---

## 8. Things that die on our numbers

| idea | why it dies |
|---|---|
| **GDS / NVMe additive bandwidth** | GPU x16 ingress is the shared bottleneck and we are at 92% of it. `nvidia-fs` isn't even installed. |
| **More copy engines / more streams** | `asyncEngineCount = 2`. Engines are not the constraint; the x16 link is. |
| **`cudaMemcpyBatchAsync`** | Exists in CUDA 13.2 and would cut per-copy launch overhead — but a device-index gather kernel subsumes it *and* is graph-capturable. |
| **Bigger LRU expert cache** | Measured **9.0% hit** at 5.6% capacity — barely above random. GLM-5.3 activates 8 of 288 (2.8%), far sparser than the Mixtral-class models where caching papers show wins. Static frequency skew (§3) is the property that pays; temporal locality is not. |
| **Variable per-layer slot budgets (water-filling)** | Measured **+0.07 pp**. Layer distribution *shapes* are alike; uniform allocation is already optimal. |
| **Native-FP4 MoE kernels (b12x / FlashInfer / CUTLASS)** | Measured at GLM dims: Marlin is **fastest at every batch size** (77.5 µs at M=1 vs 99.8 / 110.0 / 409.4). End-to-end, `b12x` served at **1.2–1.3 tok/s**. Marlin's *"no native support for FP4"* warning is unconditional and proves nothing. |
| **CPU co-execution / FreeToken hybrid** | Measured **0.895x**. Host RAM is 1.04x PCIe here; their own policy needs 2.0x. |
| **Raising `gpu-memory-utilization`** | 0.95 and 0.96 give byte-identical VRAM and the same 272,771-token KV cache. Buys nothing. |
| **Lowering `max-num-batched-tokens` for headroom** | 2048 and 4096 crash the engine. ~~Withdrawn~~ as an Xid 31 MMU fault that also hit at 8192 (fixed by `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + staging bounds checks), then **re-confirmed on the fixed build**: 4096 dies on the *first* request. Two independent failures; this one is unfixed, so the cheap route to VRAM headroom stays closed ([OPTIMIZING §11.4](OPTIMIZING-GLM53-ON-ONE-GPU.md#114-the-floor-that-was-not-an-artifact-after-all)). |
| **Deeper speculation (n-gram / MTP past depth 2)** | Depth 5 exceeds the 4 GiB KV cap at startup; depth 4 hits a DeepGEMM `block_kv` assert in the sparse-attention indexer on arch 12. At depth 2 a free drafter saves ~2–3% of a step. §6. |
| **Re-profiling the hot set on mixed-temperature traffic** | Measured **worse on everything** (13.02 vs 15.77 prose) for +4 GiB of margin. Sampling at T>0 flattens the routing skew the profile exists to capture. §3. |
| **The native SM120 sparse-MLA attention backend** | Refuses GLM-5.3 at startup — it expects DeepSeek's `qk_rope_head_dim=64` / 576-wide query, and this model is NoPE MLA. We run the SM90 kernel on Blackwell by fallback. |
| **Prefill via more VRAM or a faster link** | Not needed — prefill was slow because it *skipped* the fast path, not because the link is slow. Grouping the expert walk was worth **2.9x** with no hardware change. |
| **Reducing top-k below 6** | top-4 discards 30% of routed weight and still projects only 17.7 tok/s — and degrades quality further. |
| **Expert pruning / merging (offline)** | Changes model quality irreversibly and needs retraining or calibration for a 320B model. |
| **A smaller GLM-5.3 variant** | No documented smaller Flash variant found. |
| **Buying more RAM** | Even at 256 GB, GLM-5.3's best case projects ~55 tok/s while Qwen3.8-Flash-Next already delivers ~145–173 tok/s *because it fits*. When a model does not fit, the answer is a smaller model. |

---

## 9. Where it ended up

| # | change | effort | outcome |
|---|---|---|---|
| — | starting point (staging + MTP-2 + **eager**) | — | 11.77 tok/s |
| 1 | **device-index Triton gather kernel → CUDA graphs restored** | moderate | **13.53** ✅ |
| 2 | **frequency-ranked hot-expert cache, offload 125 + 66 slots** | low, plus one profiling run | **16.50** ✅ |
| 3 | ~~n-gram spec decode stacked on MTP (code traffic)~~ | trivial | ❌ **blocked** — KV cap at depth 5, DeepGEMM `block_kv` assert at depth 4 (§6) |
| 4 | FP8 dense weights (§7.1) | high — self-quantize | untried, ~31 dB SNR cost |
| 5 | mixed-precision cold experts (§4) | high — new kernel | untried, ~+20% |
| 6 | FreeToken hybrid CPU co-execution | very high | ❌ measured 0.895x |
| 7 | GPUDirect Storage (§1) | high | ❌ ingress-bound |
| 8 | re-profile the hot set on mixed traffic (§3) | low | ❌ **worse on everything** — 13.02 vs 15.77 prose |
| — | *router pruning top-6* | trivial | *20.48 — **rejected on quality*** |

**And the row this table did not have a column for.** Every entry above is
decode tok/s, which is why the day's largest quality-neutral win is missing from
it:

| # | change | effort | outcome |
|---|---|---|---|
| P | **grouped DMA staging during prefill, gated at ≥ 48 tokens** | moderate | **614 → 1729–1771 tok/s prefill** ✅ (10k TTFT 16.8 s → 5.82–5.96 s; decode unchanged; logged) |
| C | **context 262,144 → 327,680, KV cap derived from it, funded by 62 → 54 hot slots** | low | ✅ **a 304,800-token prompt prefills** (205 s, 1,487 tok/s), for a decode cost inside the run-to-run spread |
| M | masked-group prefill variant — stage only the routed experts | moderate | **0.36 s TTFT on the 14-token control** vs 0.85 s shipped, but withdrawn on a misattributed nondeterminism claim. **Untested since; a live candidate, not a closed branch** |
| S | **`VLLM_USE_BREAKABLE_CUDAGRAPH=0` + staging bounds checks** | trivial | ✅ **0 failures / 0 Xid** in two 40-request soaks, at unchanged speed |

**What none of this changes.** The wall is host memory bandwidth on a consumer
platform: **64.6 GB/s dual-channel** against FreeToken's reference 178 GB/s.
Every technique here works *around* that number. None of them move it, and no
software will.

**What it does change is which number you should be looking at.** This document
spent its whole length ranking ways to raise decode tok/s, and the largest
remaining win on the box turned out to be a phase it never modelled. Before
optimising the next one of these: measure time-to-first-token on a realistic
prompt, separately from the inter-token rate, and find out which one your users
are actually waiting on.

---

## Sources

- [MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/html/2401.14361v3) — request-level vs token-level locality; the "uniform across requests" caveat, still untested here
- [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/pdf/2410.06270) — per-expert mixed bit allocation, 3.03-bit and 2.54-bit configurations
- [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/pdf/2312.17238) — LRU expert caching and speculative expert prefetch
- [MoE-Beyond: Learning-Based Expert Activation Prediction on Edge Devices](https://arxiv.org/pdf/2508.17137) — router skew, long-tail expert utilisation
- [Mixture of Neuron Experts](https://arxiv.org/pdf/2510.05781) — 25% budget captures 42.6% of activations (we measure 64.1% per layer)
- [Mixed-Precision Quantization for Language Models](https://arxiv.org/pdf/2510.16805) — sub-4-bit accuracy loss, expert sensitivity heterogeneity
- [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://www.lmsys.org/blog/2023-11-21-lookahead-decoding/) — draft-model-free speculation
- [Speculative Decoding Production Guide](https://www.spheron.network/blog/speculative-decoding-production-guide/) — 0.6–0.75 acceptance for instruction following
- [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching](https://arxiv.org/html/2511.14102v1) — combining quantization with prefetch
- FreeToken source, `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh` — the vectorized gather kernel achieving 51.8 GB/s here
