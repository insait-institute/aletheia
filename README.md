# Aletheia: What Makes RLVR For Code Verifiers Tick?

> **Abstract:** Multi-domain thinking verifiers trained via Reinforcement Learning from Verifiable Rewards (RLVR) are a prominent fixture of the Large Language Model (LLM) post-training pipeline, owing to their ability to robustly rate and rerank model outputs. However, the adoption of such verifiers towards code generation has been comparatively sparse, with execution feedback constituting the dominant signal. Nonetheless, code verifiers remain valuable toward judging model outputs in scenarios where execution feedback is hard to obtain and are a potentially powerful addition to the code generation post-training toolbox. To this end, we create and open-source **Aletheia**, a controlled testbed that enables execution-grounded evaluation of code verifiers' robustness across disparate policy models and covariate shifts. We examine components of the RLVR-based verifier training recipe widely credited for its success: (1) intermediate thinking traces, (2) learning from negative samples, and (3) on-policy training. While experiments show the optimality of RLVR, we uncover important opportunities to simplify the recipe. Particularly, despite code verification being amenable to training- and inference-time scaling, on-policy learning stands out as the key component at smaller verifier sizes, and thinking-based training emerges as the most important component at larger scales.

## Getting Started
Ensure you have git-lfs installed and initialized before cloning the repository.
```
git lfs install
```

To setup the environment, simply run
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