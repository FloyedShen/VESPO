<div align="center">

# VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.10693)
[![Github](https://img.shields.io/badge/VESPO-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/FloyedShen/VESPO)

</div>

<div align="center">
  <p>
    <a href="#-overview">📖 Overview</a> •
    <a href="#-main-results">📊 Main Results</a> •
    <a href="#-getting-started">🚀 Getting Started</a> •
    <a href="#-citation">📝 Citation</a> 
  </p>
</div>

## 📖 Overview

Off-policy updates are inevitable in RL for LLMs due to rollout staleness, asynchronous training, and training-inference mismatches. VESPO incorporates variance reduction into a variational formulation and derives a closed-form reshaping kernel that operates directly on sequence-level importance weights without token-level approximation or length normalization.

<p align="center">
  <img src="figures/overview.png" width="30%" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/overview_b.png" width="34%" />
</p>

*The proposal Q\* balances proximity to both the behavior policy μ and the target π under an importance weight budget. **(Right)** Training reward with staleness N=4: VESPO remains stable while GRPO and SAPO collapse. VESPO scales to 64× staleness and fully asynchronous training without divergence.*

<p align="center">
  <img src="figures/phi_and_f_comparison.png" width="50%" />
</p>

*From a REINFORCE perspective, what matters is the effective coefficient on ∇log π, i.e., φ(W)=W·f'(W) (bottom row). VESPO's gamma-shaped kernel provides separate control over positive and negative advantages, offering more flexibility than hard clipping or fixed normalization.*

## 📊 Main Results

VESPO's robustness extends to fully asynchronous training, where rollout and training run on separate node groups with multi-step policy lag.

<p align="center">
  <img src="figures/fully_async_2x4.png" width="85%" />
</p>

*Training dynamics under fully asynchronous training on Qwen3-30B-A3B-Base. VESPO maintains stable training and achieves the highest reward and benchmark accuracy.*

> [!NOTE]
> For complete results across different staleness ratios (N=4 to 64), model scales, and ablation studies, please refer to our [paper](https://arxiv.org/abs/2602.10693).

## 🚀 Getting Started

The core VESPO policy loss is in [core_algos.py](recipe/vespo/code/core_algos.py). Training scripts are under [recipe/vespo/run/](recipe/vespo/run/).

**1. Install** — follow the [veRL documentation](https://verl.readthedocs.io/en/latest/start/install.html) to set up the environment.

**2. Prepare data**

```bash
cd recipe/vespo/tools
python preprocess_datasets.py
```

**3. Train**

Edit the model and data paths in the script, then launch with a Ray cluster:

```bash
# Synchronous (N=8, 32 GPUs)
bash recipe/vespo/run/sync/vespo_N_8.sh

# Fully asynchronous (48 rollout + 16 train GPUs)
bash recipe/vespo/run/fully_async/vespo_S_1.0_N_4.sh
```

> [!TIP]
> Additional synchronous scripts for other staleness ratios are available under `recipe/vespo/run/sync/` (N=16, 32, 64).

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{shen2026vespovariationalsequencelevelsoft,
  title={VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training},
  author={Guobin Shen and Chenxiao Zhao and Xiang Cheng and Lei Huang and Xing Yu},
  year={2026},
  eprint={2602.10693},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.10693},
}
```

## Attribution

Our implementation is based on a recent version of [veRL](https://github.com/volcengine/verl).
