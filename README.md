# UniPROT: Uniform Prototype Selection via Partial Optimal Transport with Submodular Guarantees

**AISTATS 2026** · [arXiv:2604.10952](https://arxiv.org/abs/2604.10952)

UniPROT is a novel subset selection framework that minimises the optimal transport (OT) distance between a **uniformly weighted** prototypical distribution and the target distribution. Existing methods often rely on implicit importance scores skewed towards majority classes, leading to low-quality prototypes for minority classes. UniPROT reformulates the OT marginal constraints to yield a **partial optimal transport-based submodular objective**, enabling a greedy algorithm with a $(1 - 1/e)$ approximation guarantee relative to the original super-additive maximisation problem. It consistently improves minority-class representation in imbalanced classification benchmarks without compromising majority-class accuracy, and delivers robust performance gains in both fine-tuning and pre-training regimes for large language models under domain imbalance.

---

## Overview

Given a source pool $\mathcal{X}$ and a target distribution $\mu_T$, UniPROT selects a subset $S \subseteq \mathcal{X}$ of size $k$ that **minimises the optimal transport distance** between the _uniformly weighted_ prototypical distribution $\mu_S = \frac{1}{k}\sum_{i \in S} \delta_{x_i}$ and $\mu_T$:

$$\min_{S \subseteq \mathcal{X},\, |S|=k} \; \mathrm{OT}(\mu_S,\, \mu_T)$$

Directly solving this is intractable because the uniform-weight constraint makes the objective **super-additive**, ruling out standard submodular guarantees. UniPROT addresses this by reformulating the OT marginal constraints to obtain an equivalent **partial optimal transport (POT)** objective:

$$F(S) = \max_{\Gamma \geq 0} \;\langle C_{S,:}, \Gamma \rangle \quad \text{s.t.} \quad \Gamma\,\mathbf{1} \leq \tfrac{1}{k}\mathbf{1}_k,\;\; \mathbf{1}_k^\top \Gamma = \mu_T$$

where $C$ is a pairwise cost matrix and $\Gamma$ is a partial transport plan. This reformulated objective is **submodular**, enabling a greedy algorithm with a provable $(1 - 1/e)$ approximation guarantee relative to the original super-additive maximisation. The greedy procedure supports both an exact re-solve and a fast closed-form approximation at each step.

---

## Repository Structure

```
UniPROT/
├── UniPROT/                        # Core UniPROT implementation
│   ├── UniPROT.py                  #   Greedy prototype selector (exact & approx)
│   ├── sinkhorn.py                 #   Partial OT solvers (custom + POT wrappers)
│   └── fairOT.py                   #   Standalone driver / quick-start script
│
├── baselines/                      # Comparison methods
│   ├── main.py                     #   TinyImageNet baseline experiment entry point
│   ├── SPOTgreedy.py               #   SPOT-Greedy subset selection
│   ├── evaluation.py               #   Cost/similarity matrices and data splits
│   ├── features.py                 #   HuggingFace feature extraction
│   ├── loader.py                   #   TinyImageNet DataLoader
│   └── MMDcritic/                  #   MMD-Critic baseline
│
├── image/                          # Vision experiments
│   ├── exp.py                      #   Main image experiment (all methods & datasets)
│   ├── exp_ablation.py             #   Ablation: regularisation / method variants
│   ├── exp_ablation_other_datasets.py
│   ├── datasets_viz.py             #   Dataset visualisation utilities
│   └── proto_selection_evals/      #   Flickr30K prototype evaluation scripts
│
└── llm-experiments/                # LLM fine-tuning experiments
    ├── train/
    │   ├── train.py                #   Main training entry point
    │   ├── uniprot.py              #   UniPROT data selection for LLMs
    │   ├── facility_location.py    #   Facility-location baseline
    │   ├── SPOTgreedy.py           #   SPOT-Greedy baseline
    │   ├── subset_trainer_distributed.py  # Distributed subset trainer
    │   └── ...
    ├── data/                       #   Dataset loaders and prompt templates
    ├── math_eval/                  #   Math reasoning evaluation suite
    ├── superglue_eval/             #   SuperGLUE evaluation suite
    └── scripts/                    #   Shell scripts for full runs
```

---

## Installation

```bash
# Core dependencies
pip install torch torchvision transformers peft
pip install POT scikit-learn numpy matplotlib tqdm scipy

# Optional: GPU-accelerated similarity matrices
pip install cupy-cuda12x          # match your CUDA version

# Optional: experiment tracking
pip install wandb
```

---

## Quick Start

### Prototype selection with UniPROT

```python
import torch
from UniPROT.UniPROT import UniPROT

# Build a cosine-similarity matrix  (n_source × n_target)
S = torch.randn(1000, 500)

# Select 50 prototypes using the fast approximation
selector = UniPROT(regularization=0.1, device="cpu")
indices, objectives = selector.prototype_selection(
    S, num_prototypes=50, method="approx", epsilon=0.001
)
# indices — LongTensor of selected row indices into S
```

### Sinkhorn / partial OT solvers

```python
from UniPROT.sinkhorn import pot_partial_library, pot_partial_extended

# POT-based solver (recommended)
plan = pot_partial_library(cost_matrix, reg=0.1, k=50)

# Custom extended-dummy-row solver
plan = pot_partial_extended(cost_matrix, reg=0.1, k=50)
```

---

## Experiments

### Image experiments

```bash
cd image

# Full comparison: UniPROT vs SPOT-Greedy vs MMD-Critic vs uniform random
# Datasets: MNIST, Letter, USPS, TinyImageNet, Flickr30K
# Evaluation: 1-NN accuracy under varying class skew (10 %–100 %)
python exp.py

# Ablation over regularisation strength and method variants
python exp_ablation.py
```

Key hyperparameters (edit at the top of `exp.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_prototypes` | 500 | Fixed prototype budget |
| `regularization` | 0.1 | UniPROT entropic regularisation $\varepsilon$ |
| `skew_levels` | [10,30,50,70,100] | Target class-skew percentages |
| `n_runs` | 5 | Independent random repetitions |

Results are saved to `image/plots/`.

### Baseline (TinyImageNet)

```bash
# Set dataset path, then run SPOT-Greedy + random baselines
export TINYIMAGENET_ROOT=/path/to/tiny-imagenet-200
python baselines/main.py
```

### LLM fine-tuning

```bash
cd llm-experiments

# SuperGLUE (RTE, MRPC, CoLA, QQP …) with UniPROT data selection
python train/train.py \
  --model_name_or_path microsoft/phi-2 \
  --train_files superglue-rte \
  --output_dir ./output/phi2_rte \
  --data_selection_method uniprot \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --regularization 0.1

# Convenience shell scripts
bash scripts/run_superglue.sh
bash scripts/run_math.sh
```

Supported `--data_selection_method` values: `none`, `facility_location`, `greedy_diversity`, `uniprot`.

---

## Citation

If you use UniPROT in your research, please cite:

```bibtex
@inproceedings{chanda2026uniprot,
  title     = {UniPROT: Uniform Prototype Selection via Partial Optimal Transport with Submodular Guarantees},
  author    = {Chanda, Prateek and Agrawal, Prayas and Gurumoorthy, Karthik S. and Ramakrishnan, Ganesh and Mishra, Bamdev and Jawanpuria, Pratik},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2026},
  eprint    = {2604.10952},
  archivePrefix = {arXiv},
}
```
