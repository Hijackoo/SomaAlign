# SomaAlign: Reliability-Aware Multimodal Representation Learning for Soft Robotic Grasping

[中文说明](README_zh.md)

[Project Page](https://hijackoo.github.io/SomaAlign/)

SomaAlign is a multimodal representation learning framework for soft robotic
perception. It aligns synchronized RGB, second-view images, depth, tactile
pressure, IMU, and language observations in a shared 1024-dimensional embedding
space, enabling retrieval and comparison across heterogeneous sensor streams.

## Overview

Soft robotic systems combine sensors with very different signal structures and
noise characteristics. Applying the same alignment strength to every modality
pair can amplify unreliable observations and distort the shared representation.
SomaAlign addresses this problem with a progressive three-stage framework:

```text
Stage I                         Stage II                         Stage III
Reliability Estimation         Adaptive Alignment              Semantic Refinement
──────────────────────         ────────────────────             ───────────────────

Masked reconstruction          Reliability-aware               Conflict-aware
+ signal quality modeling      contrastive learning             semantic consistency
                                                               + geometry preservation

             └──────────────────────┴──────────────────────┘
                         Shared multimodal embedding space
```

**Key idea:** first estimate how trustworthy each modality is, then use that
information to guide cross-modal alignment, and finally refine semantic
consistency without collapsing the geometry of the learned representation.

### Stage I: Modality Reliability Estimation

SomaAlign uses modality-specific masked reconstruction to characterize signal
quality. Self-reconstruction measures the reliability of each observation
without requiring one sensor to predict another sensor with a fundamentally
different representation.

### Stage II: Reliability-Aware Feature Alignment

The modality encoders are aligned through pairwise contrastive learning.
Reliability estimates adapt the strength of each modality-pair objective so
that clean signals receive sharper alignment while uncertain signals are
handled more conservatively.

### Stage III: Conflict-Aware Semantic Refinement

The aligned representation is refined using sample-level cross-modal conflict.
SomaAlign strengthens agreement for compatible observations while protecting
the global feature geometry when modalities provide conflicting evidence.

## Repository Features

- Inference support for RGB, second-view images, depth, tactile pressure, IMU,
  and text.
- Unified 2F and 3F tactile-sensor dataset adapters.
- Strict checkpoint loading with model-key and architecture validation.
- Tensor-level embedding extraction for individual modalities.
- Cross-modal retrieval evaluation with paired and description-level positive
  sets.
- User-controlled evaluation splits, random subsets, and explicit sample lists.

## Datasets

- [Soma](https://huggingface.co/datasets/Why427/Soma): synchronized multimodal
  observations for two-finger and three-finger soft robotic grasping.
- [Soma-Success](https://huggingface.co/datasets/Why427/Soma_Success): diverse
  grasping trials for grasp-outcome evaluation.

The files for both datasets will be made publicly available upon paper
acceptance.

## Project Structure

```text
SomaAlign/
├── inference.py                     # Model construction and embedding extraction
├── eval_retrieval.py                # Cross-modal retrieval evaluation
├── modal_config.py                  # Modality and preprocessing configuration
├── requirements.txt                 # Runtime dependencies
├── data/
│   ├── SomaData.py                  # Unified 2F + 3F dataset adapter
│   ├── SomaData2F.py                # 2F tactile-sensor dataset
│   └── SomaData3F.py                # 3F tactile-sensor dataset
├── ImageBind/
│   ├── bpe/                         # Text tokenizer vocabulary
│   └── models/                      # Multimodal encoder implementation
├── dataset_splits/                  # Split-manifest documentation
└── tests/                           # Data-free release checks
```

## Quick Start

### 1. Installation

Create a Python 3.10 environment and install a PyTorch build suitable for your
CPU or CUDA environment before installing the remaining dependencies:

```bash
conda create -n somaalign python=3.10 -y
conda activate somaalign

# Choose the appropriate command for your platform from pytorch.org.
pip install torch torchvision
pip install -r requirements.txt
```

The release has been validated with PyTorch 2.6 and torchvision 0.21.

### 2. Prepare a Checkpoint

SomaAlign accepts a direct PyTorch state dictionary or a checkpoint containing
`state_dict` or `model_state_dict`. Checkpoints must match the released encoder
architecture. Pretrained weights will be released upon paper acceptance.

### 3. Extract Embeddings

```python
import torch

from inference import build_model, encode, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device)
load_checkpoint(model, "/path/to/somaalign_checkpoint.pth")

# One RGB temporal window: [frames, channels, height, width]
rgb = torch.randn(3, 3, 224, 224)
embedding = encode(model, {"rgb": rgb}, device)["rgb"]  # [1, 1024]
```

Supported input names are `rgb`, `photo`, `depth`, `tactile`, `imu`, and
`text`. Image-like temporal batches use `[B,S,C,H,W]`, IMU uses `[B,C,T]`, and
tokenized text uses `[B,T]`.

## Data Organization

The provided adapters expect the following high-level layout:

```text
<data_root>/
├── <class_name>/
│   └── <trial>/
│       └── <sample>/
│           ├── RGB/
│           ├── Photo/
│           ├── Depth/
│           ├── Tactile/
│           └── IMU/
├── 标签.xlsx                  # Default; override with --label_file_2f/3f
└── text_annotations.json
```

For portable user-defined splits, provide sample IDs relative to each 2F or 3F
data root. The loader also accepts zero-based indices in its deterministic
lexicographic directory order; see [split formats](dataset_splits/README.md).

```json
{
  "sample_ids": {
    "train": ["1_Object/1/0"],
    "val": ["1_Object/2/0"],
    "test": ["1_Object/3/0"]
  }
}
```

## Cross-Modal Retrieval Evaluation

```bash
python eval_retrieval.py \
  --checkpoint /path/to/somaalign_checkpoint.pth \
  --data_root_2f /path/to/2f \
  --data_root_3f /path/to/3f \
  --split_file_2f /path/to/split_2f.json \
  --split_file_3f /path/to/split_3f.json \
  --output results/retrieval.json
```

By default, the evaluator uses all samples in the selected split. Users can
choose a reproducible random subset or provide exact split-relative indices:

```bash
# Seeded random subset
python eval_retrieval.py ... --sample_count 500 --seed 42

# Explicit user-defined subset
python eval_retrieval.py ... --sample_indices_file indices.json

```

To generate a seeded 70/15/15 split instead of supplying manifests:

```bash
python eval_retrieval.py \
  --checkpoint /path/to/somaalign_checkpoint.pth \
  --data_root_2f /path/to/2f \
  --data_root_3f /path/to/3f \
  --generate_split --seed 42 \
  --output results/retrieval.json
```

This sample-level generated split is for running the example; it is not a
prescribed paper evaluation split. For a trial-level split, provide your own
manifests with all samples from each trial in the same partition.

If your label files have other names, set `--label_file_2f` and
`--label_file_3f` accordingly.

Similarity ranking is evaluated in bounded query chunks, so full-split
evaluation does not allocate a complete square similarity matrix. Adjust the
memory/speed trade-off with `--retrieval_query_batch_size` (default: `512`).

Sensory-to-sensory retrieval uses synchronized observations as positives.
Sensory-to-text retrieval uses a deduplicated description gallery, while
text-to-sensory retrieval treats all observations sharing the same structured
description as valid positives. Recall@1/5/10 is reported for every directed
modality pair and macro-averaged across tasks.

Evaluation results depend on the split and gallery composition. When reporting
results, include the split manifest, selection method, sample count, and random
seed.

## Verification

Run the data-free API and retrieval checks with:

```bash
python -m unittest discover -s tests -v
```

## Citation

Preprint: [arXiv:2611.02794](https://arxiv.org/abs/2611.02794).

```bibtex
@misc{wang2026somaalign,
  title         = {SomaAlign: Reliability-Aware Multimodal Representation Learning for Soft Robotic Grasping},
  author        = {Haoyuan Wang and Shuai Peng and Yongkai Liao and Yuxiao Sun and Wei Luo and Xinxing Chen and Jian Huang},
  year          = {2026},
  eprint        = {2611.02794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.02794}
}
```

## Acknowledgements

Acknowledgements are omitted during double-blind review.

## License

SomaAlign is distributed under the [CC BY-NC-SA 4.0 license](LICENSE).
