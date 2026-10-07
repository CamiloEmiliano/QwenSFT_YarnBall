# QwenSFT_YarnBall: Financial Intelligence SFT Training Engine

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CamiloEmiliano%2Fyarnball--sft-blue)](https://huggingface.co/datasets/CamiloEmiliano/yarnball-sft)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--7B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dedicated Supervised Fine-Tuning (SFT) repository for training a unified **`Qwen2.5-7B-Instruct`** financial intelligence model for the **YarnBall** GraphRAG platform.

---

## 1. Overview & Capabilities

This model is fine-tuned on the point-in-time grounded [CamiloEmiliano/yarnball-sft](https://huggingface.co/datasets/CamiloEmiliano/yarnball-sft) corpus (12,470 records across all 11 GICS sectors) to master 5 core capabilities:

1. **Task A (`<|extract_sec_graph|>`)**: Converts SEC Form 10-K disclosures into validated OpenCypher graph triples with materiality tiers.
2. **Task B (`<|extract_news_event|>`)**: Extracts real-time breaking market event edges with 5-axis directional polarity and temporal validity.
3. **Task C (`<|text_to_cypher|>`)**: High-precision natural language queries translated to executable Cypher queries.
4. **Task D (`<|contagion_reasoning|>`)**: Multi-hop supply chain and credit contagion modeling using step-by-step `<think>` Chain-of-Thought.
5. **Task E (`<|portfolio_recommendation|>`)**: Institutional risk hedging (zero-cost collars, put spreads, swaptions) and asset reallocation recommendations with `<think>` Chain-of-Thought.

---

## 2. Model & Training Hyperparameters

| Hyperparameter | Specification | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-7B-Instruct` | Native multilingual, high-throughput ChatML instruction following. |
| **Precision** | `bfloat16` | Native Ampere / Hopper / Ada mixed precision (no quantization artifacts). |
| **Attention** | `FlashAttention-2` | Maximize throughput and sequence length (2,048 tokens). Falls back to PyTorch SDPA. |
| **LoRA Rank (r)** | `32` | High rank for complex graph syntax and reasoning representation. |
| **LoRA Alpha (alpha)** | `64` | Scaling factor (alpha / r = 2.0). |
| **LoRA Dropout** | `0.05` | Regularization against overfitting on corporate boilerplates. |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All linear projections adapted. |
| **Loss Masking** | Completion-Only (`DataCollatorForCompletionOnlyLM`) | Cross-entropy loss computed strictly after `<\|im_start\|>assistant\n`. |
| **Epochs** | `3` | Optimal convergence across 12,470 point-in-time records. |
| **Batch Size** | `4` per device × `4` grad accum | Effective global batch size: 16. |
| **Learning Rate** | `2.0e-4` (Cosine schedule, 5% warmup) | Stable convergence for LoRA rank 32. |

---

## 3. Quickstart & Verification

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dry-Run Smoke Test (2 Steps on 20 Samples)
Before launching a paid GPU container on Vast.ai, run a dry run to verify tokenizer masking and loss computation:
```bash
python3 train.py --dry-run
```

---

## 4. Cloud Training Execution (Vast.ai)

### Recommended GPU Specifications:
- **GPU**: 1x NVIDIA A100 (80GB SXM) or 1x RTX 4090 / A6000 Ada (24GB/48GB).
- **Vast.ai Filters**: `datacenter: true` and `verified: true` (for security and 1+ Gbps fiber network).
- **Estimated Run Time**: ~30–45 minutes (~$0.80–$1.50 total cost).

### Automated Datacenter Workflow (vast_runner.py):
The included `vast_runner.py` enforces strict institutional security (`datacenter=true`, `verified=true`):

```bash
# 1. Search top verified datacenter offers (A100 SXM4 80GB)
python vast_runner.py search --gpu A100_SXM4

# 2. Dry-run provisioning check (no credit spent)
python vast_runner.py launch --gpu A100_SXM4 --dry-run

# 3. Provision instance (with confirmation prompt)
python vast_runner.py launch --gpu A100_SXM4 --max-price 2.00

# 4. Check instance status & SSH command
python vast_runner.py status
python vast_runner.py ssh

# 5. Destroy instance once training completes (prevents idle billing)
python vast_runner.py destroy <INSTANCE_ID>
```


### Execution Inside Vast.ai Container:
```bash
# 1. Clone repository
git clone https://github.com/CamiloEmiliano/QwenSFT_YarnBall.git
cd QwenSFT_YarnBall

# 2. Export Hugging Face token (for private dataset pull and adapter push)
export HF_TOKEN="<your_huggingface_token>"

# 3. Launch training
bash launch_vast.sh
```

Upon completion, the LoRA adapter weights and tokenizer are automatically pushed to:
`https://huggingface.co/CamiloEmiliano/yarnball-qwen-7b-lora`