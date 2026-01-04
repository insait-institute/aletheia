# Aletheia: Ablating the Success of RLVR for Code Verifiers

> **Abstract:** Verifiers are a prominent fixture of the Large Language Model (LLM) post-training pipeline owing to their ability to guide policy model outputs. Recently, verifiers have repurposed the generative capabilities of pretrained models in a manner that can be scaled with inference compute. Recent work has converged on leveraging Reinforcement Learning from Verifiable Rewards (RLVR) to train robust multi-domain generative verifiers. While possessing a demonstrably high performance ceiling, this recipe imposes exacting demands on training infrastructure and data collection pipelines. Moreover, these methods have not yet been applied to verifiers for code generation. To this end, we create **Aletheia**, a testbed that enables reliable, execution-grounded evaluation of coding verifiers across a vast array of policy models, difficulty levels, and adversarial settings. We ablate the prevalent RLVR-based verifier training recipe along three axes widely credited for its success: (1) Generating intermediate ``thinking'' traces, (2) learning from negative samples, and (3) on-policy training. While our experiments across three model scales and disparate out-of-domain generalization settings show that all the constituent components of the RLVR positively contribute to its final performance, we uncover important opportunities to simplify the recipe. Particularly, while the verification task demonstrates consistently strong training- and inference-time scaling trends, on-policy learning stands out as the crucial component at smaller verifier sizes, and thinking-based training emerges as the most important component at larger scales.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training 
GRPO-Think
```bash
cd src
model='1_5b' # one of 1_5b, 7b,  14b
b_tr=4096 # reasoning budget
accelerate launch --config_file=configs/deepspeed/config.yaml grpo_training.py model=grpo/dsqwen_${model} method=grpo grpo_params.max_completion_length=${b_tr}
``` 

GRPO-Instruct
```bash
cd src
model='1_5b' # one of 1_5b, 7b,  14b
b_tr=4096 # reasoning budget
accelerate launch --config_file=configs/deepspeed/config.yaml grpo_training.py model=grpo_instruct/qwen_${model} method=grpo_instruct gen_params.max_completion_length=${b_tr}
``` 

ThinkDPO
```bash
cd src
model='1_5b' # one of 1_5b, 7b,  14b
b_tr=16384 # reasoning budget
accelerate launch --config_file=configs/deepspeed/config.yaml thinkdpo_training.py model=thinkdpo/dsqwen_1_5b method=thinkdpo gen_params.max_completion_length=${b_tr}
``` 

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 