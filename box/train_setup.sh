#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r box/train-requirements.txt
.venv/bin/python -m pip freeze > box/train-environment.lock.txt
.venv/bin/python - <<'PY'
import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig
print({'torch': torch.__version__, 'cuda_build': torch.version.cuda})
assert torch.version.cuda is not None, 'Install a CUDA-enabled PyTorch wheel first'
assert tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8), 'Blackwell requires CUDA 12.8+'
PY
# Download only. This step does not initialize a GPU and needs no lease.
HF_HOME="${HF_HOME:-$PWD/.hf}" .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from auditor_ml.modeling import MODEL_ID, MODEL_REVISION
print(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
    ignore_patterns=['original/*', '*.pt', '*.msgpack', '*.h5']))
PY
