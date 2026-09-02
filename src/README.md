# Source

The optimization is two pieces of code.

| file | what it is |
|---|---|
| `dma_stage.py` | **New file.** The Triton dedup gather kernel (device-side indices, CUDA-graph-capturable), the hot-expert VRAM cache with its frequency profile, and the offload detection. This is the whole optimization. |
| `marlin_moe_apply_excerpt.py` | **Excerpt** of the hook inside vLLM's `MarlinExperts.apply` that calls the stager, plus the opt-in grouped-prefill path. Reproduced from a modified vLLM file (Apache-2.0). |

Both target a **specific vLLM fork** (`v0.29.0.dev0+glm53`) and a specific
checkpoint. They are published to show the mechanism and the measurements, not
as a drop-in patch — the surrounding APIs move quickly. Read
`../OPTIMIZING-GLM53-ON-ONE-GPU.md` for why each piece is shaped the way it is.

Environment variables they respond to are documented in `../config/serve-opt.sh`.
