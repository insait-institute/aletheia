<h1 align="center">
  <img src="figures/logo.png" width="50" align="center" alt="Aletheia Logo"> 
  Aletheia: What Makes RLVR For Code Verifiers Tick?
</h1>

<p align="center">
  <a href="arxiv.org"><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Aletheia-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Aletheia-yellow?style=flat" alt="HuggingFace"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-CC BY--NC--SA 4.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <b>Vatsal Venkatkrishna, Indraneil Paul, Iryna Gurevych</b>
</p>

---

## Key Features

- 🔬 **Controlled Testbed**: Execution-grounded evaluation across disparate policy models and covariate shifts
- 🧠 **Multiple Training Methods**: GRPO-Think, GRPO-Instruct, and DPO-Think implementations
- 📊 **Comprehensive Evaluation**: Evaluation scripts for the Aletheia benchmark datasets
- 📦 **Model Zoo**: Fine-tuned verifiers at 1.5B, 7B, and 14B scales

---

## 🎯 Abstract

Multi-domain thinking verifiers trained via Reinforcement Learning from Verifiable Rewards (RLVR) are a prominent fixture of the Large Language Model (LLM) post-training pipeline, owing to their ability to robustly rate and rerank model outputs. However, the adoption of such verifiers towards code generation has been comparatively sparse, with execution feedback constituting the dominant signal. Nonetheless, code verifiers remain valuable toward judging model outputs in scenarios where execution feedback is hard to obtain and are a potentially powerful addition to the code generation post-training toolbox. To this end, we create and open-source **Aletheia**, a controlled testbed that enables execution-grounded evaluation of code verifiers' robustness across disparate policy models and covariate shifts. We examine components of the RLVR-based verifier training recipe widely credited for its success: (1) intermediate thinking traces, (2) learning from negative samples, and (3) on-policy training. While experiments show the optimality of RLVR, we uncover important opportunities to simplify the recipe. Particularly, despite code verification exhibiting positive training- and inference-time scaling, on-policy learning stands out as the key component at small verifier sizes, and thinking-based training emerges as the most important component at larger scales.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Conda or similar environment manager
- CUDA 12.6 or higher 

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/vatsal-kr/aletheia.git
   cd aletheia
   ```

2. **Set up the environment**:
   ```bash
   # Create and activate conda environment
   conda create -n aletheia python=3.12
   conda activate aletheia
   
   # Install dependencies
   pip install -r requirements.txt
   ```
---

## 🏋️ Training

We provide training scripts for three different methods: **GRPO**, **GRPO-Instruct**, and **DPO-Think**. All methods support three model sizes: `1_5b`, `7b`, and `14b`. We make our trained models available on [HuggingFace](https://huggingface.co/Aletheia-Bench). 

### GRPO (Group Relative Policy Optimization)

Standard GRPO training with thinking traces:

```bash
cd src

# Configuration
model='1_5b'      # Options: 1_5b, 7b, 14b
b_tr=4096         # Reasoning budget (max completion length)

# Launch training
accelerate launch \
  --config_file=configs/deepspeed/config.yaml \
  grpo_training.py \
  model=grpo/${model} \
  method=grpo \
  grpo_params.max_completion_length=${b_tr}
```

### GRPO-Instruct

GRPO training without thinking traces (instruction-only):

```bash
accelerate launch \
  --config_file=configs/deepspeed/config.yaml \
  grpo_training.py \
  model=grpo_instruct/${model} \
  method=grpo_instruct \
  gen_params.max_completion_length=${b_tr}
```

### DPO-Think

Direct Preference Optimization with thinking traces:

```bash
accelerate launch \
  --config_file=configs/deepspeed/config.yaml \
  thinkdpo_training.py \
  model=thinkdpo/${model} \
  method=thinkdpo \
  gen_params.max_completion_length=${b_tr}
```

---

## 📊 Evaluation

Evaluate trained models on the Aletheia benchmark datasets using the provided evaluation script:

```bash
cd src

# Basic evaluation
python evaluate.py \
  --eval_llm /path/to/checkpoint \
  --dataset Heldout \
  --K 16 \
```

### Evaluation Metrics

The evaluation script generates K responses and calculates:
- **SC Accuracy**: The self consistency accuracy. A model is correct for a given prompt if it generates the correct answer most of the time (majority voting) across K responses.
- **BoN Accuracy**: The best of N accuracy. A model is correct for a given prompt if it generates the correct answer at-least once across N responses.
---

## 🎁 Datasets

The **Aletheia** dataset collection is available on [HuggingFace](https://huggingface.co/Aletheia-Bench) and includes:
- **Aletheia-Train**: A dataset of 2-5 candidate code snippets to solve a coding problem, each with exactly one correct code
- **Aletheia-DPO**: A companion dataset to Aletheia-Train, containing "chosen" and "rejected" responses for each instance. The chosen response identifies the correct code snippet, while the rejected responses do not.
- **Aletheia-Heldout**: A completely in-distribution test set
- **Aletheia-Strong**: An OOD test set where the candidates are generated by stronger models
- **Aletheia-Hard**: An OOD test set where the comparison between candidates is more difficult
- **Aletheia-Adv**: An OOD test set where the candidates are adversarially modified to exploit common LLM biases
---

## 📁 Project Structure

```
aletheia/
├── src/
│   ├── grpo_training.py          # GRPO training script (Think and Instruct)
│   ├── thinkdpo_training.py      # DPO-Think training script
│   ├── evaluate.py               # Evaluation script
│   ├── rewards.py                # Reward formulations
│   ├── prompts.py                # Prompt templates
│   └── configs/
│       └── deepspeed/            # DeepSpeed configuration
│       └── method/               # Global method configurations
│           └── grpo.yaml
│           └── grpo_instruct.yaml
│           └── thinkdpo.yaml
│       └── model/                # Model specific configurations
│           └──grpo/
│           │   └── 1_5b.yaml
│           │   └── 7b.yaml
│           │   └── 14b.yaml
│           └──grpo_instruct/
│           │   └── ...
│           └──thinkdpo/
│           │   └── ...
├── figures/
│   └── logo.png                  # Project logo
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{aletheia2025,
  title={Aletheia: What Makes RLVR For Code Verifiers Tick?},
  author={Venkatkrishna, Vatsal and Paul, Indraneil and Gurevych, Iryna},
  journal={arXiv preprint},
  year={2025}
}
```

*Note: Full citation details coming soon.*

---

## 📄 License

This project is licensed under the CC BY-NC-SA License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This repository contains experimental research software and is published for the sole purpose of giving additional background details on the respective publication. Use at your own risk.

---


<p align="center">
  Made with ❤️ by the Aletheia Team
</p> 
