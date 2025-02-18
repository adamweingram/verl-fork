<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>Paper</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>Wechat</b></a> | <a href="https://x.com/verl_project"><b>Twitter</b></a>

</p>

## Custom Instructions

## Setup
1. Set up environment
  ```bash
  spack env activate .
  spack concretize -f && spack install -j32
  ```
2. Initialize a virtual environment using `uv`
  ```bash
  uv venv --seed
  ```
3. Activate the virtual environment
  ```bash
  source ./.venv/bin/activate
  ```
4. Updates
  ```bash
  uv pip install --upgrade pip setuptools wheel
  ```
4. Install `vllm`. Note that we need the specific version or we'll get an error when we try to run veRL later.
  ```bash
  uv pip install vllm==0.6.3
  ```
5. Install items first
  ```bash
  uv pip install torch torchvision torchaudio torchaudio transformers
  ```
6. Deal with Flash Attention specifically (cannot use isolated build environment, so use `--no-build-isolation`). We have to do this manually because the `pyproject.toml` file cannot include the `--no-build-isolation` flag.
  ```bash
  uv pip install -U --no-build-isolation flash-attn==2.6.3
  ```
7. Sync and Lock the environment
  ```bash
  uv sync --extra gpu
  uv lock
  ```
8. Install dependencies in the virtual environment using `uv` and the `pyproject.toml`
  ```bash
  uv pip install -r pyproject.toml --extra gpu
  ```
<!-- 9. Install `verl` in the virtual environment
  ```bash
  uv pip install -e .
  ``` -->

### Prepare Datasets
1. Download the `gsm8k` dataset
  ```bash
  mkdir -p ./data
  git clone https://huggingface.co/datasets/openai/gsm8k ./data/gsm8k
  mv ./data/gsm8k/main/test-00000-of-00001.parquet ./data/gsm8k/test.parquet
  mv ./data/gsm8k/main/train-00000-of-00001.parquet ./data/gsm8k/train.parquet
  ```
2. Pre-process the `gsm8k` dataset
  ```bash
  source .venv/bin/activate
  python ./examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k
  ```

## Key Features

- **FSDP** and **Megatron-LM** for training.
- **vLLM** and **TGI** for rollout generation, **SGLang** support coming soon.
- huggingface models support
- Supervised fine-tuning
- Reinforcement learning from human feedback with [PPO](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer), [GRPO](https://github.com/volcengine/verl/tree/main/examples/grpo_trainer), and [ReMax](https://github.com/volcengine/verl/tree/main/examples/remax_trainer)
  - Support model-based reward and function-based reward (verifiable reward)
- flash-attention, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [long context](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh)
- scales up to 70B models and hundreds of GPUs
- experiment tracking with wandb, swanlab and mlflow

## Upcoming Features
- Reward model training
- DPO training
- DeepSeek integration with Megatron backend
- SGLang integration

## Getting Started

Checkout this [Jupyter Notebook](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer/verl_getting_started.ipynb) to get started with PPO training with a single 24GB L4 GPU (**FREE** GPU quota provided by [Lighting Studio](https://lightning.ai/hlin-verl/studios/verl-getting-started))!

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Running a PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO and GRPO](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)

## Performance Tuning Guide
The performance is essential for on-policy RL algorithm. We write a detailed performance tuning guide to allow people tune the performance. See [here](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more details.

## vLLM v0.7 testing version
We have released a testing version of veRL that supports vLLM>=0.7.0. Please refer to [this document](https://github.com/volcengine/verl/docs/README_vllm0.7.md) for installation guide and more information.

## Contribution Guide
Contributions from the community are welcome!

### Code formatting
We use yapf (Google style) to enforce strict code formatting when reviewing PRs. To reformat you code locally, make sure you installed **latest** `yapf`
```bash
pip3 install yapf --upgrade
```
Then, make sure you are at top level of verl repo and run
```bash
bash scripts/format.sh
```

## Citation and acknowledgement

If you find the project helpful, please cite:
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and supported by Anyscale, Bytedance, LMSys.org, Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, and University of Hong Kong.

## Awesome work using verl
- [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://arxiv.org/abs/2410.21236)
- [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME/)
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of DeepSeek R1 Zero recipe for reasoning tasks
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning agent training framework
- [Logic R1](https://github.com/Unakar/Logic-RL): a reproduced DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset.
- [deepscaler](https://github.com/agentica-project/deepscaler): iterative context scaling with GRPO
- [critic-rl](https://github.com/HKUNLP/critic-rl): Teaching Language Models to Critique via Reinforcement Learning

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in MLSys/LLM reasoning/multimodal alignment.
