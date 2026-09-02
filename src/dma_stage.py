# SPDX-License-Identifier: Apache-2.0
"""Gathered-DMA staging for UVA-offloaded MoE expert weights.

Why this exists
---------------
UVA offloading maps host memory into the GPU address space and lets the matmul
read it directly. That read is slow -- but *not* because the memory is remote.
It is the access pattern. Measured on RTX PRO 6000 / PCIe 5.0 x16, gathering
8 rows of 4 MiB from a host-mapped view:

    torch.index_select                  1.45 ms    23.1 GB/s
    per-row copy_ (cudaMemcpyAsync)     0.81 ms    41.4 GB/s
    this kernel                         0.64 ms    52.1 GB/s

``index_select`` is slow because its gather reads scattered elements with narrow
SM loads across the PCIe BAR. A per-row ``copy_`` lowers to the DMA engine and
does much better -- but it needs the expert ids on the HOST to drive the loop,
and that ``.tolist()`` sync is illegal inside CUDA graph capture, which forced
the whole engine into eager mode.

This kernel takes the indices as a DEVICE tensor. No sync, so it captures
cleanly, and the wide contiguous per-row access reaches 52 GB/s of the 56.4 GB/s
link -- 92% of ingress, and faster than the DMA path it replaces.

On GLM-5.3-Flash decode roughly 204 of every 207 ms per forward pass is this
transfer, so the transfer rate is very nearly the token rate.

Two things deliberately NOT done here
-------------------------------------
* **No pinned host handle per parameter.** An earlier version kept one to DMA
  from. It was unnecessary (copying straight from the UVA view is exactly as
  fast) and harmful: the extra reference pinned a superseded copy of every
  expert, taking host RSS from 1.08x to 2.13x of offloaded bytes and OOM-killing
  the worker at full scale.
* **No LRU cache.** Measured on GLM-5.3: a 16-slot/layer cache over 288 experts
  returned a **9.0% hit rate** -- barely above the 5.6% you would get at random.
  Expert routing here has almost no temporal locality, so caching costs
  bookkeeping and returns nothing. (Static frequency skew is a different
  property and may still be worth exploiting; that is not this code.)
"""

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

# Cap the staged working set so a pathological step cannot claim the card.
MAX_STAGED_EXPERTS = 64

# 8192 measured fastest (52.1 GB/s); 4096 ties, 16384+ falls to ~49.
_BLOCK = 8192


@triton.jit
def _gather_rows_cached(
    src, cache, dst, idx, slot, n, row_bytes, n_experts, BLOCK: tl.constexpr, MAXN: tl.constexpr
):
    """dst[i] = cache[slot[idx[i]]] if resident else src[idx[i]].

    Two sources, chosen per row by a DEVICE-resident slot table, so there is no
    host sync and the kernel still captures into a CUDA graph. That matters:
    forcing eager to do cache bookkeeping costs 13% (11.77 vs 13.53 tok/s
    measured), which would eat most of what the cache wins.

    A hit reads VRAM at ~1.8 TB/s instead of PCIe at ~52 GB/s, so a hit is
    effectively free; only misses cost link time.
    """
    r = tl.program_id(0)
    b = tl.program_id(1)
    e = tl.load(idx + r).to(tl.int64)
    # An id outside [0, n_experts) must never reach the address arithmetic
    # below. A padded or sentinel id (-1, or n_experts) would read past the
    # host tensor, or -- worse -- read garbage from past the slot table and
    # then dereference that garbage into the cache: an unmapped address, i.e.
    # an intermittent Xid 31 MMU fault that surfaces in whatever kernel runs
    # next. Such a row is simply not fetched; the expert map never points at it.
    if (e < 0) | (e >= n_experts):
        return

    # skip duplicate experts (same reason as the uncached kernel)
    js = tl.arange(0, MAXN)
    prev = tl.load(idx + js, mask=js < r, other=-1).to(tl.int64)
    if tl.sum((prev == e).to(tl.int32)) > 0:
        return

    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < row_bytes
    sl = tl.load(slot + e).to(tl.int64)
    if sl >= 0:
        v = tl.load(cache + sl * row_bytes + offs, mask=m, other=0)
    else:
        v = tl.load(src + e * row_bytes + offs, mask=m, other=0)
    tl.store(dst + r * row_bytes + offs, v, mask=m)


@triton.jit
def _gather_rows(src, dst, idx, n, row_bytes, n_experts, BLOCK: tl.constexpr, MAXN: tl.constexpr):
    """dst[i, :] = src[idx[i], :], with idx on the device.

    One program per (row, chunk). The chunk is a long contiguous run so the
    compiler widens the accesses; that width is what turns a BAR read from
    ~23 GB/s into ~52.

    ``idx`` comes straight from topk_ids and therefore REPEATS experts (the
    same expert is picked by several of the tokens verified in one MTP step).
    We cannot compact it -- torch.unique has a data-dependent output shape and
    CUDA graph capture rejects that -- so instead each program checks whether
    its expert already appeared at a lower row and returns without copying if
    so. The grid stays statically shaped while only unique bytes cross PCIe;
    ``build_local_expert_map`` points every expert at its FIRST occurrence, so
    the skipped slots are never read.
    """
    r = tl.program_id(0)
    b = tl.program_id(1)
    e = tl.load(idx + r).to(tl.int64)
    # An id outside [0, n_experts) must never reach the address arithmetic
    # below. A padded or sentinel id (-1, or n_experts) would read past the
    # host tensor, or -- worse -- read garbage from past the slot table and
    # then dereference that garbage into the cache: an unmapped address, i.e.
    # an intermittent Xid 31 MMU fault that surfaces in whatever kernel runs
    # next. Such a row is simply not fetched; the expert map never points at it.
    if (e < 0) | (e >= n_experts):
        return

    # duplicate check against the rows before us (n <= MAXN, so one vector op)
    js = tl.arange(0, MAXN)
    prev = tl.load(idx + js, mask=js < r, other=-1).to(tl.int64)
    if tl.sum((prev == e).to(tl.int32)) > 0:
        return

    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < row_bytes
    v = tl.load(src + e * row_bytes + offs, mask=m, other=0)
    tl.store(dst + r * row_bytes + offs, v, mask=m)


def is_offloaded(t: torch.Tensor | None) -> bool:
    """True if ``t`` is a device-addressable view of host memory (UVA offload).

    The offloader marks the Parameter, not the tensor the kernel receives, so
    the marker is invisible here. Probing the pointer with
    cudaPointerGetAttributes needs ctypes and an oversized attribute buffer
    (the struct has grown across CUDA releases; declaring it too small corrupts
    the heap). Instead: a UVA view is a CUDA tensor whose storage is host
    memory, and torch reports that faithfully through ``is_pinned`` on the
    underlying storage's host mapping -- which is what ``untyped_storage`` sees.
    """
    if t is None or not t.is_cuda:
        return False
    cached = _OFFLOAD_CACHE.get(t.data_ptr())
    if cached is None:
        cached = _probe_host_mapped(t)
        _OFFLOAD_CACHE[t.data_ptr()] = cached
    return cached


_OFFLOAD_CACHE: dict[int, bool] = {}
_ATTR_BUF_BYTES = 256  # cudaPointerAttributes has grown; oversize it deliberately
_CUDA_MEMORY_TYPE_HOST = 1
_get_attrs = None
_probe_broken = False


def _probe_host_mapped(t: torch.Tensor) -> bool:
    global _get_attrs, _probe_broken
    if _probe_broken:
        return False
    import ctypes

    try:
        if _get_attrs is None:
            # Resolve from the ALREADY-LOADED runtime. ctypes.CDLL("libcudart")
            # would map a second copy of the CUDA runtime into the process.
            lib = ctypes.CDLL(None)
            _get_attrs = lib.cudaPointerGetAttributes
            _get_attrs.restype = ctypes.c_int
            _get_attrs.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        buf = ctypes.create_string_buffer(_ATTR_BUF_BYTES)
        rc = _get_attrs(
            ctypes.cast(buf, ctypes.c_void_p), ctypes.c_void_p(t.data_ptr())
        )
        return rc == 0 and int.from_bytes(buf.raw[:4], "little") == _CUDA_MEMORY_TYPE_HOST
    except Exception as exc:  # pragma: no cover - driver dependent
        _probe_broken = True
        logger.warning("expert-offload probe unavailable (%s); staging off", exc)
        return False


class ExpertDMAStager:
    """Gathers the experts a step selected into VRAM, capture-safe.

    One shared instance for the whole model: MoE layers run sequentially and
    each is done with its staged experts before the next begins, so they share
    buffers. Per-layer stagers multiply the footprint by the layer count (45x
    here), which exhausts VRAM and fails as "No available memory for the cache
    blocks".
    """

    __slots__ = ("_buffers", "_announced", "_ev", "_ms", "_bytes", "_n",
                 "_hist", "_hist_steps", "_per_layer", "_cache", "_slot", "_pinned",
                 "_warned", "_profile")

    def __init__(self) -> None:
        self._buffers: dict[int, torch.Tensor] = {}
        self._announced = False
        # _maybe_pin runs once per layer (33x here), so a bare logger.warning
        # would repeat every message 33 times.
        self._warned: set[str] = set()
        # Profile is read ONCE, not re-opened per layer.
        self._profile: dict | None = None
        # CUDA-event timing of the gather itself (VLLM_MOE_TIME_GATHER=1).
        # Events are async: they record on the stream and are only read back
        # every N passes, so this does not serialize the pipeline the way
        # torch.cuda.synchronize() would.
        self._ev: list = []
        self._ms = 0.0
        self._bytes = 0
        self._n = 0
        # Per-expert activation counts, summed over layers. Static frequency
        # skew is a DIFFERENT property from the temporal locality an LRU cache
        # exploits (measured at 9% hit here); if the top-k experts by frequency
        # are stable across workloads, pinning them beats vLLM's current
        # module-order residency. Enabled by VLLM_MOE_SKEW_DUMP=<path>.
        self._hist: torch.Tensor | None = None
        self._hist_steps = 0
        self._per_layer: dict[int, torch.Tensor] = {}
        # Hot-expert cache, per layer. layer_id -> {tensor_key: VRAM cache},
        # plus a device-resident slot table the kernel reads to decide, per
        # expert, whether to hit VRAM or fetch over PCIe.
        self._cache: dict[int, dict[int, torch.Tensor]] = {}
        self._slot: dict[int, torch.Tensor] = {}
        self._pinned = False

    def _buffer(self, key: int, src: torch.Tensor) -> torch.Tensor:
        """Worst-case-sized buffer, allocated once.

        Deliberately MAX_STAGED_EXPERTS rows, not the current step's count.
        vLLM profiles peak memory on a dummy forward and hands every remaining
        byte to the KV cache; a buffer that grows later has nowhere to go and
        dies mid-request with "CUDA out of memory ... 248 MiB free".
        """
        buf = self._buffers.get(key)
        if buf is None:
            self._buffers[key] = buf = torch.empty(
                (MAX_STAGED_EXPERTS, *src.shape[1:]),
                dtype=src.dtype,
                device=src.device,
            )
        return buf

    def _warn_once(self, msg: str, *args) -> None:
        """Log ``msg`` the first time only; this runs once per layer."""
        if msg not in self._warned:
            self._warned.add(msg)
            logger.warning(msg, *args)

    def _maybe_pin(
        self,
        layer: int,
        tensors: list[torch.Tensor | None],
        stageable: list[bool],
        num_experts: int,
    ) -> None:
        """Populate this layer's hot-expert cache from a frequency profile.

        vLLM makes residency a WHOLE-LAYER decision: with cpu_offload_gb set,
        some layers are entirely in VRAM and the rest entirely on the host. So
        the resident share of expert *fetches* equals the resident share of
        layers -- 36.2% measured -- and which experts they are is decided by
        module order, i.e. arbitrarily.

        Routing is in fact strongly skewed WITHIN each layer (measured: the
        hottest 36% of a layer's experts serve 75% of its fetches). Pinning by
        frequency instead of by module order therefore converts the same VRAM
        into far more hits. The aggregate histogram hides this -- summing counts
        across layers washes the skew out, which is why an earlier look at the
        data wrongly concluded routing was near-uniform.

        Profile path comes from VLLM_MOE_HOT_PROFILE; absent, no cache is built
        and the plain gather runs, so this is inert unless asked for.
        """
        import json
        import os

        if layer in self._cache:
            return
        path = os.environ.get("VLLM_MOE_HOT_PROFILE")
        if not path or not os.path.exists(path):
            self._cache[layer] = {}
            return
        if self._profile is None:
            try:
                blob = json.load(open(path))
                self._profile = blob["layers"]
                # A profile is only valid at the cpu_offload_gb it was recorded
                # under: residency is a whole-layer decision, so a different
                # offload changes WHICH layers are host-side and the hot sets
                # stop lining up. A 125-profile run at 127 measured 15.09 vs
                # 16.76 tok/s -- a silent 10% loss with no error at all.
                want = os.environ.get("CPU_OFFLOAD_GB")
                got = blob.get("offload_gb")
                if want and got and str(want) != str(got):
                    self._warn_once(
                        "hot-expert profile was recorded at CPU_OFFLOAD_GB=%s "
                        "but this run uses %s; the pinned experts belong to a "
                        "different layer partition and the hit rate will "
                        "silently drop. Regenerate it at this offload.",
                        got, want,
                    )
            except Exception as exc:
                logger.warning("hot-expert profile unreadable (%s); pinning off", exc)
                self._profile = {}
        prof = self._profile

        # Layers are keyed by discovery order, which matches the order stage()
        # first sees them -- the same order the profile was written in.
        key_for = str(len(self._cache))
        counts = prof.get(key_for)
        if counts is None or len(counts) != num_experts:
            # Silence here once cost 19 points of hit rate: a profile written
            # in a different key order still parsed, so every layer got another
            # layer's hot set and nothing complained.
            self._warn_once(
                "hot-expert profile does not match this model: layer %s of %d "
                "profiled is missing or has %s experts, expected %d. Pinning "
                "is OFF for the mismatched layers -- regenerate with "
                "VLLM_MOE_SKEW_PERLAYER under the SAME cpu_offload_gb.",
                key_for, len(prof),
                "None" if counts is None else len(counts), num_experts,
            )
            self._cache[layer] = {}
            return

        budget = int(os.environ.get("VLLM_MOE_HOT_SLOTS", "0"))
        if budget <= 0:
            if prof:
                self._warn_once(
                    "hot-expert profile loaded but VLLM_MOE_HOT_SLOTS is unset; "
                    "pinning stays OFF and the profile does nothing"
                )
            self._cache[layer] = {}
            return
        budget = min(budget, num_experts)
        hot = sorted(range(num_experts), key=lambda i: -counts[i])[:budget]

        slot = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
        for i, e in enumerate(hot):
            slot[e] = i
        per_layer: dict[int, torch.Tensor] = {}
        for k, (t, do) in enumerate(zip(tensors, stageable)):
            if not do:
                continue
            c = torch.empty((budget, *t.shape[1:]), dtype=t.dtype, device=t.device)
            for i, e in enumerate(hot):
                c[i].copy_(t[e])
            per_layer[k] = c
        self._cache[layer] = per_layer
        self._slot[layer] = slot.to(tensors[0].device)
        if not self._pinned:
            self._pinned = True
            logger.info(
                "Hot-expert pinning ACTIVE: %d/%d experts per layer cached in VRAM "
                "(frequency-ordered, device-side slot table, graph-safe)",
                budget, num_experts,
            )

    def stage(
        self,
        tensors: list[torch.Tensor | None],
        experts: torch.Tensor,
        num_experts: int,
    ) -> list[torch.Tensor | None] | None:
        """Gather ``experts`` (1-D, device-resident) of each tensor into VRAM.

        Returns staged tensors in the same order, or None when staging does not
        apply -- the caller must then use the originals unchanged.
        """
        n = int(experts.numel())
        if n == 0 or n > MAX_STAGED_EXPERTS:
            return None

        # Stage only tensors that are (a) host-resident, so the copy buys
        # something, and (b) indexed by expert on dim 0. A per-tensor scalar
        # must pass through untouched -- indexing it by expert id reads garbage.
        stageable = [
            t is not None
            and t.ndim >= 1
            and t.shape[0] == num_experts
            and is_offloaded(t)
            for t in tensors
        ]
        # Reserve unconditionally, before any early return: profiling runs a full
        # prefill batch where n exceeds the cap, so an allocation placed after the
        # cap check never happens during profiling and the KV cache eats the VRAM.
        for key, (t, do) in enumerate(zip(tensors, stageable)):
            if do:
                self._buffer(key, t)
        if not any(stageable):
            return None

        layer = tensors[0].data_ptr()
        self._maybe_pin(layer, tensors, stageable, num_experts)

        idx = experts.to(torch.int32)

        import os

        _time = bool(os.environ.get("VLLM_MOE_TIME_GATHER"))
        if _time:
            _e0 = torch.cuda.Event(enable_timing=True)
            _e1 = torch.cuda.Event(enable_timing=True)
            _e0.record()

        import os

        if os.environ.get("VLLM_MOE_SKEW_PERLAYER"):
            # Per-LAYER counts. The aggregate histogram answers "how skewed is
            # routing on average"; pinning needs to know which experts to keep
            # in EACH layer, and layers may differ. Keyed by the w1 pointer,
            # which is stable for the life of the process.
            lid = tensors[0].data_ptr()
            per = self._per_layer.setdefault(
                lid,
                torch.zeros(num_experts, dtype=torch.int64, device=experts.device),
            )
            per.scatter_add_(
                0,
                experts.to(torch.int64).reshape(-1),
                torch.ones_like(experts.reshape(-1), dtype=torch.int64),
            )
            self._hist_steps += 1
            if self._hist_steps % 6000 == 0:
                self._dump_per_layer()

        if os.environ.get("VLLM_MOE_SKEW_DUMP"):
            if self._hist is None:
                self._hist = torch.zeros(
                    num_experts, dtype=torch.int64, device=experts.device
                )
            self._hist.scatter_add_(
                0,
                experts.to(torch.int64).reshape(-1),
                torch.ones_like(experts.reshape(-1), dtype=torch.int64),
            )
            self._hist_steps += 1
            if self._hist_steps % 4000 == 0:
                self._dump_skew(num_experts)
        out: list[torch.Tensor | None] = []
        for key, (t, do) in enumerate(zip(tensors, stageable)):
            if not do:
                out.append(t)
                continue
            buf = self._buffer(key, t)
            row_bytes = t[0].numel() * t.element_size()
            grid = (n, triton.cdiv(row_bytes, _BLOCK))
            cache = self._cache.get(layer, {}).get(key)
            slot = self._slot.get(layer)
            if cache is not None and slot is not None:
                _gather_rows_cached[grid](
                    t.view(torch.uint8).view(-1),
                    cache.view(torch.uint8).view(-1),
                    buf.view(torch.uint8).view(-1),
                    idx,
                    slot,
                    n,
                    row_bytes,
                    num_experts,
                    BLOCK=_BLOCK,
                    MAXN=MAX_STAGED_EXPERTS,
                )
            else:
                _gather_rows[grid](
                    t.view(torch.uint8).view(-1),
                    buf.view(torch.uint8).view(-1),
                    idx,
                    n,
                    row_bytes,
                    num_experts,
                    BLOCK=_BLOCK,
                    MAXN=MAX_STAGED_EXPERTS,
                )
            out.append(buf[:n])

        if _time:
            _e1.record()
            self._ev.append((_e0, _e1, sum(
                n * t[0].numel() * t.element_size()
                for t, do in zip(tensors, stageable) if do)))
            self._n += 1
            if self._n % 2000 == 0:
                torch.cuda.synchronize()
                ms = sum(a.elapsed_time(b) for a, b, _ in self._ev)
                by = sum(c for _, _, c in self._ev)
                logger.info(
                    "GATHER: %.1f ms over %d calls (%.3f ms/call), %.2f GB, "
                    "%.1f GB/s in situ",
                    ms, len(self._ev), ms / len(self._ev), by / 1e9,
                    by / (ms / 1000) / 1e9,
                )
                self._ev = []

        if not self._announced:
            self._announced = True
            logger.info(
                "ExpertDMAStager ACTIVE: %d experts/step, %d tensor(s), "
                "device-side indices (CUDA-graph safe)",
                n, sum(1 for x in stageable if x),
            )
        return out


    def _dump_per_layer(self) -> None:
        """Write per-layer expert counts for the hot-set pinning profile."""
        import json
        import os

        path = os.environ["VLLM_MOE_SKEW_PERLAYER"]
        # Keyed by DISCOVERY order, not pointer order. _maybe_pin reads these
        # back as str(len(self._cache)), i.e. the order stage() first sees each
        # layer. sorted() here ordered them by data_ptr instead, so the profile
        # and the pinner disagreed and every layer got another layer's hot set
        # -- measured hit rate 45% against a predicted 73%.
        out = {str(i): t.tolist() for i, t in enumerate(self._per_layer.values())}
        with open(path, "w") as fh:
            # offload_gb is what makes a profile reusable-or-not: both
            # serve-opt.sh and _maybe_pin compare against it to catch a profile
            # being used at an offload it was never captured at.
            json.dump({
                "steps": self._hist_steps,
                "layers": out,
                "offload_gb": os.environ.get("CPU_OFFLOAD_GB"),
            }, fh)
        logger.info(
            "MoE per-layer profile: %d layers, %d steps -> %s",
            len(out), self._hist_steps, path,
        )

    def _dump_skew(self, num_experts: int) -> None:
        """Write per-expert counts + the concentration curve."""
        import json
        import os

        path = os.environ["VLLM_MOE_SKEW_DUMP"]
        c = self._hist.float()
        total = float(c.sum().item())
        if total <= 0:
            return
        srt = torch.sort(c, descending=True).values / total
        cum = torch.cumsum(srt, 0)
        pts = {}
        for frac in (0.05, 0.10, 0.25, 0.35, 0.50):
            k = max(1, int(num_experts * frac))
            pts[f"top_{int(frac*100)}pct_capacity"] = round(float(cum[k - 1]), 4)
        payload = {
            "steps": self._hist_steps,
            "num_experts": num_experts,
            "coverage": pts,
            "uniform_reference": {
                f"top_{int(f*100)}pct_capacity": f for f in (0.05, 0.10, 0.25, 0.35, 0.50)
            },
            "counts": self._hist.tolist(),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        logger.info(
            "MoE skew @%d steps: top-25%% of experts capture %.1f%% of activations "
            "(uniform would be 25.0%%); top-35%% -> %.1f%%",
            self._hist_steps,
            100 * pts["top_25pct_capacity"],
            100 * pts["top_35pct_capacity"],
        )


_SHARED: "ExpertDMAStager | None" = None


def shared_stager() -> "ExpertDMAStager":
    global _SHARED
    if _SHARED is None:
        _SHARED = ExpertDMAStager()
    return _SHARED


def build_local_expert_map(
    experts: torch.Tensor, global_num_experts: int
) -> torch.Tensor:
    """Global expert id -> staged row, -1 for experts not staged.

    Built entirely on device (no host round-trip) so it stays capture-safe.
    This is vLLM's existing expert_map convention, so the Marlin kernels need no
    change: they already translate ids through such a table for expert
    parallelism.
    """
    n = experts.numel()
    # Each expert must resolve to its FIRST occurrence, because that is the only
    # row the kernel fills (later duplicates are skipped). A plain scatter
    # (m[experts] = arange) will NOT do: PyTorch leaves the winner undefined
    # when indices repeat, and it empirically returns last-ish occurrences.
    # scatter_reduce with amin is deterministic.
    order = torch.arange(n, dtype=torch.int32, device=experts.device)
    m = torch.full(
        (global_num_experts,), n, dtype=torch.int32, device=experts.device
    )
    # scatter_reduce_ does not bounds-check on CUDA: an id of -1 or
    # global_num_experts would silently write outside ``m`` and corrupt the
    # neighbouring allocation. Route invalid ids to index 0 carrying the
    # "absent" value n, which amin can never prefer over a real occurrence.
    valid = (experts >= 0) & (experts < global_num_experts)
    safe = torch.where(valid, experts, torch.zeros_like(experts)).to(torch.int64)
    order = torch.where(valid, order, torch.full_like(order, n))
    m.scatter_reduce_(0, safe, order, reduce="amin")
    return torch.where(
        m == n, torch.full_like(m, -1), m
    )
