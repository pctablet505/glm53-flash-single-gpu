#!/usr/bin/env bash
# GLM-5.3-Flash ABLITERATED NVFP4 -> GGUF -> llama.cpp server
# Usage:
#   ~/Projects/glm53-run.sh convert   # NVFP4 safetensors -> GGUF (needs ~180G free)
#   ~/Projects/glm53-run.sh serve     # llama-server on port 8002, GPU+CPU expert offload
set -euo pipefail

LLAMA=~/Projects/glm53-llama
CONV=~/Projects/glm53-conv-venv/bin/python
SRC=~/Models/GLM-5.3-Flash-ABLITERATED-NVFP4
OUT=~/Models/GLM-5.3-Flash-ABLITERATED-NVFP4-GGUF
CUDA_HOME=$HOME/Projects/glm53-venv/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUT"

case "${1:-}" in
convert)
    # Lossless NVFP4 passthrough: converter detects quant_method=modelopt NVFP4
    # and repacks to GGML_TYPE_NVFP4 without requantizing.
    "$CONV" "$LLAMA/convert_hf_to_gguf.py" "$SRC" \
        --outfile "$OUT/glm53-flash-abliterated-nvfp4.gguf" \
        --outtype bf16 \
        2>&1 | tee /tmp/glm53_convert.log
    ;;
serve)
    # -ncmoe 250: keep 250 of 288 routed experts per layer on GPU? No —
    # --n-cpu-moe N = push N experts/layer to CPU DRAM. Tune:
    #   VRAM budget: 96G - KV - activations ~= 80G for weights.
    # Start conservative: all experts on CPU (--cpu-moe), attention+shared on GPU.
    # Then dial --n-cpu-moe down until VRAM ~90G.
    exec "$LLAMA/build/bin/llama-server" \
        -m "$OUT/glm53-flash-abliterated-nvfp4.gguf" \
        --n-cpu-moe 288 \
        -ngl 99 \
        -c 8192 \
        -fa on \
        --mtp 3 \
        --host 127.0.0.1 --port 8002 \
        "$@"
    ;;
*)
    echo "usage: $0 convert|serve [extra llama-server args]"
    exit 1
    ;;
esac
