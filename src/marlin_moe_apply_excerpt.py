# Excerpt from vLLM's MarlinExperts.apply (Apache-2.0, modified).
# This is the HOOK that makes offloaded experts fast: it gathers the experts
# this step actually routes to into VRAM before Marlin reads them, and adds an
# opt-in grouped path for prefill. Full file lives in the vLLM fork; only the
# added logic is reproduced here.

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        # --- gathered-DMA staging of UVA-offloaded experts --------------------
        # When the expert weights live in host memory behind a UVA view, the
        # matmul reads them over the PCIe BAR with narrow scattered SM loads and
        # gets a small fraction of the link. Instead, gather just the experts
        # this step selected into VRAM first, then run the kernel against those.
        # The gather is a wide contiguous per-row gather that reaches 92% of
        # ingress -- see the measured table in offloader/dma_stage.py.
        #
        # The staged tensors carry only the selected experts, so we hand the
        # kernel an expert_map -- exactly the translation table it already uses
        # for expert parallelism. No kernel change is needed.
        #
        # Skipped when an expert_map is already in play (real EP): composing
        # the two mappings is not worth the risk for a single-GPU path.
        _record_router(topk_weights)
        # --- optional router pruning: fetch fewer experts per token ---------
        # PCIe gather of routed experts is ~60% of a decode step here, and it
        # scales with how many DISTINCT experts a step touches. GLM routes
        # top-8; the lowest-weighted of those contribute least to the output,
        # so dropping them trades a little quality for a large traffic cut.
        # Opt-in and off by default: this changes model numerics.
        #
        # Selects by weight rather than slicing [:, :k] -- the router's output
        # order is not guaranteed to be weight-sorted, and slicing the wrong
        # ones would silently degrade quality. torch.topk here has a static
        # output shape, so CUDA graph capture is unaffected. Weights are
        # renormalized because GLM sets norm_topk_prob=True.
        import os as _os

        _k = int(_os.environ.get("VLLM_MOE_DECODE_TOPK", "0"))
        if 0 < _k < topk_ids.shape[1]:
            _w, _sel = torch.topk(topk_weights, _k, dim=-1)
            topk_ids = torch.gather(topk_ids, 1, _sel).contiguous()
            topk_weights = _w / _w.sum(dim=-1, keepdim=True)

        w1_scale, w2_scale = self.w1_scale, self.w2_scale
        g1_alphas, g2_alphas = self.g1_alphas, self.g2_alphas
        if _dma_staging_enabled() and expert_map is None:
            from vllm.model_executor.offloader.dma_stage import shared_stager

            stager = shared_stager()
            # Anything else indexed by expert must not stay globally indexed
            # while the staged tensors are locally indexed -- the kernel would
            # read the wrong rows and return plausible garbage with no error.
            # If we find such a tensor, skip staging entirely.
            others = (
                self.w1_bias, self.w2_bias, self.a1_gscale, self.a2_gscale,
                self.w1_zp, self.w2_zp, self.w13_g_idx, self.w2_g_idx,
                self.w13_g_idx_sort_indices, self.w2_g_idx_sort_indices,
            )
            unstaged_per_expert = any(
                t is not None and t.ndim >= 1 and t.shape[0] == global_num_experts
                for t in others
            )
            # NOT torch.unique: its output shape is data-dependent, which CUDA
            # graph capture forbids ("operation not permitted when stream is
            # capturing"). topk_ids.reshape(-1) is statically shaped for a given
            # captured batch. Duplicates cost nothing: the gather kernel skips
            # any id that appeared earlier in the list, and the expert map sends
            # every copy of an id to its FIRST occurrence.
            # Marlin itself looks ids up through expert_map with no bounds
            # check either. A row whose id is outside [0, E) is made inert:
            # id 0 with weight 0 contributes exactly nothing, whatever the
            # row was (CUDA-graph padding or a sentinel). Static shapes only.
            _ok = (topk_ids >= 0) & (topk_ids < global_num_experts)
            topk_ids = torch.where(_ok, topk_ids, torch.zeros_like(topk_ids))
            topk_weights = torch.where(
                _ok, topk_weights, torch.zeros_like(topk_weights)
            )
            selected = topk_ids.reshape(-1)
            staged = (
                None
                if unstaged_per_expert
                else stager.stage(
                    [w1, w2, w1_scale, w2_scale, g1_alphas, g2_alphas],
                    selected,
                    global_num_experts,
                )
            )
            if staged is not None:
                from vllm.model_executor.offloader.dma_stage import (
                    build_local_expert_map,
                )

                (w1, w2, w1_scale, w2_scale, g1_alphas, g2_alphas) = staged
                expert_map = build_local_expert_map(selected, global_num_experts)
            elif (
                _prefill_groups_enabled()
                and not unstaged_per_expert
                and self._lora_context is None
                and staged is None
                # Only worth it for real prefill batches. Grouping stages all
                # 288 experts per layer at DMA speed; a short prompt routes to
                # far fewer (14 tokens -> ~95), which zero-copy reads for less.
                # Crossover is ~40 tokens. A variant that masked unneeded
                # experts out of each group was withdrawn after it showed
                # run-to-run differences -- later found to be the baseline's
                # own prefix-cache nondeterminism, not the mask -- so it was
                # never proven either way; this gate is the simpler,
                # measured choice.
                # shape[0] is static per captured graph, so this is capture-safe.
                and topk_ids.shape[0] >= 48
            ):
                from vllm.model_executor.offloader.dma_stage import (
                    MAX_STAGED_EXPERTS,
                    build_local_expert_map,
                )

                # stage() declined because this batch touches more experts
                # than one staging buffer holds (a prefill chunk). Walk the
                # expert range in buffer-sized groups: stage the group, run
                # Marlin with an expert_map that hides every other expert, and
                # accumulate. moe_sum(..., expert_map) writes the sum over
                # PRESENT experts only, so the per-group outputs add up to the
                # exact full result; a token whose experts all fall outside the
                # group contributes zero for that group.
                # Same reduction as the single pass: every (token, expert) pair
                # belongs to exactly one group, and the second Marlin GEMM
                # writes only the sorted VALID rows of the per-pair buffer
                # (intermediate_cache3), so filling that buffer one group at a
                # time and running the one fp32 moe_sum at the end reduces the
                # same rows the single pass would. Measured equivalence is by
                # per-token prompt logprobs on FRESH prompts (this stack's
                # prefill is not run-to-run deterministic -- prefix-cache
                # state reuse shifts bf16 rounding -- so greedy-text
                # comparison proves nothing): grouped vs baseline p90 <=0.0007
                # nats, >1 nat on ~0.1% of tokens, matching the baseline's own
                # fresh-vs-cached spread. An earlier draft summed five bf16
                # partial outputs instead; that is a real extra rounding step
                # and was replaced. cache1 overlays cache3 in the same
                # storage, but a group's GEMM1 only writes rows of ITS pairs,
                # which its own GEMM2 then overwrites -- other groups' finished
                # rows are never touched.
                G = MAX_STAGED_EXPERTS
                dev = topk_ids.device
                M, K = hidden_states.size()
                topk = topk_ids.size(1)
                quant_type = ScalarType.from_id(self.quant_type_id)
                # fused_marlin_moe's block-size heuristic, evaluated for a
                # G-expert local group exactly as it would be per call.
                m_est = math.ceil(M * G / global_num_experts)
                for block_size_m in [8, 16, 32, 48, 64]:
                    if m_est * topk / G / block_size_m < 0.9:
                        break
                if self.input_dtype is not None and self.input_dtype.itemsize == 1:
                    block_size_m = max(block_size_m, 16)
                complete = True
                for lo in range(0, global_num_experts, G):
                    hi = min(lo + G, global_num_experts)
                    ids_g = torch.arange(lo, hi, dtype=torch.int32, device=dev)
                    st = stager.stage(
                        [w1, w2, w1_scale, w2_scale, g1_alphas, g2_alphas],
                        ids_g,
                        global_num_experts,
                    )
                    if st is None:
                        complete = False
                        break
                    (gw1, gw2, gs1, gs2, ga1, ga2) = st
                    map_g = build_local_expert_map(ids_g, global_num_experts)
                    sorted_ids, expert_ids, npad = moe_align_block_size(
                        topk_ids,
                        block_size_m,
                        global_num_experts,
                        map_g,
                        ignore_invalid_experts=True,
                    )
                    _fused_marlin_moe(
                        hidden_states=hidden_states,
                        w1=gw1,
                        w2=gw2,
                        bias1=self.w1_bias,
                        bias2=self.w2_bias,
                        w1_scale=gs1,
                        w2_scale=gs2,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        num_topk=topk,
                        quant_type=quant_type,
                        apply_router_weight_on_input=apply_router_weight_on_input,
                        expert_map=map_g,
                        block_size_m=block_size_m,
                        sorted_token_ids=sorted_ids,
                        expert_ids=expert_ids,
                        num_tokens_post_padded=npad,
                        activation=activation,
                        activation_func=self.activation,
                        activation_config=self.activation_config,
                        input_global_scale1=self.a1_gscale,
                        input_global_scale2=self.a2_gscale,
                        global_scale1=ga1,
                        global_scale2=ga2,
                        g_idx1=self.w13_g_idx,
                        g_idx2=self.w2_g_idx,
                        sort_indices1=self.w13_g_idx_sort_indices,
                        sort_indices2=self.w2_g_idx_sort_indices,
                        w1_zeros=self.w1_zp,
                        w2_zeros=self.w2_zp,
                        intermediate_cache13=workspace2,
                        intermediate_cache2=workspace13,
                        output=None,
                        input_dtype=self.input_dtype,
                        is_k_full=self.is_k_full,
                    )
                if complete:
                    # moe_sum's C++ op wants the per-pair buffer as [M, topk, K]
                    # (the single pass does the same .view before calling it).
                    cache3 = _resize_cache(workspace2, (M * topk, K)).view(-1, topk, K)
                    self.moe_sum(cache3, output, topk_ids, None)
                    return
        # ---------------------------------------------------------------------

