#!/usr/bin/env bash
# One-time setup on a fresh GPU instance: a uv venv matching the local smoke env, then the models.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x .venv/bin/python ] || uv venv .venv --python 3.12 -q
uv pip install -q --python .venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install -q --python .venv/bin/python "transformers==5.20.2" accelerate scipy hf_transfer "lm-eval==0.4.13" \
  "flash-linear-attention==0.5.2"  # Triton kernels for Qwen3.5's linear attention
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3.5-4B", "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]:
    print(snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"]))
EOF
.venv/bin/python -c "import torch, transformers; print('READY', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
