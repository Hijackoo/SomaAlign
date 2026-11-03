# SomaAlign：面向软体机器人抓取的可靠性感知多模态表征学习

[English](README.md)

[项目主页](https://hijackoo.github.io/SomaAlign/)

SomaAlign 是一个面向软体机器人感知的多模态表征学习框架。它将同步采集的 RGB、
第二视角图像、深度、触觉压力、IMU 和语言观测映射到统一的 1024 维嵌入空间，
支持异构传感器之间的跨模态检索与语义比较。

## 项目概览

软体机器人通常同时使用多种信号结构和噪声特性差异显著的传感器。如果对所有
模态对采用完全相同的对齐强度，不可靠观测可能被进一步放大，并破坏共享表征。
SomaAlign 通过渐进式三阶段框架解决这一问题：

```text
Stage I                         Stage II                         Stage III
可靠性估计                      自适应特征对齐                   语义细化
──────────────────────         ────────────────────             ───────────────────

掩码重建                        可靠性感知对比学习               冲突感知语义一致性
+ 信号质量建模                                                   + 全局几何保持

             └──────────────────────┴──────────────────────┘
                            共享多模态嵌入空间
```

**核心思想：**先估计不同模态观测的可信程度，再利用可靠性指导跨模态对齐，最后在
不破坏整体表征几何的前提下进一步提高语义一致性。

### Stage I：模态可靠性估计

SomaAlign 使用模态专用的掩码重建任务刻画信号质量。自重建不要求一种传感器预测
另一种表征形式完全不同的传感器，因此可以减少跨模态语义差异对可靠性估计的干扰。

### Stage II：可靠性感知特征对齐

各模态编码器通过成对对比学习映射到共享空间。可靠性估计用于调整不同模态对的
对齐强度：高质量信号采用更精确的对齐约束，不确定信号则采用更稳健的约束。

### Stage III：冲突感知语义细化

SomaAlign 根据样本级跨模态冲突进一步优化对齐表征。对于相互兼容的观测增强语义
一致性；当不同模态提供冲突信息时，则限制过度对齐并保护全局特征几何。

## 仓库功能

- 支持 RGB、第二视角图像、深度、触觉压力、IMU 和文本六种模态推理。
- 提供统一的 2F 和 3F 触觉传感器数据适配器。
- 提供带模型参数与架构校验的严格 checkpoint 加载。
- 支持单模态张量级嵌入提取。
- 支持同步正样本和描述级多正样本的跨模态检索评测。
- 支持用户自定义 split、随机子集和显式样本列表。

## 数据集

- [Soma](https://huggingface.co/datasets/Why427/Soma)：面向双指和三指软体机器人抓取的
  同步多模态观测数据集。
- [Soma-Success](https://huggingface.co/datasets/Why427/Soma_Success)：面向抓取结果评估的
  多样化抓取试验数据集。

两个数据集的文件均将在论文录用后公开。

## 项目结构

```text
SomaAlign/
├── inference.py                     # 模型构建与嵌入提取
├── eval_retrieval.py                # 跨模态检索评测
├── modal_config.py                  # 模态与预处理配置
├── requirements.txt                 # 运行依赖
├── data/
│   ├── SomaData.py                  # 统一 2F + 3F 数据适配器
│   ├── SomaData2F.py                # 2F 触觉传感器数据
│   └── SomaData3F.py                # 3F 触觉传感器数据
├── ImageBind/
│   ├── bpe/                         # 文本分词词表
│   └── models/                      # 多模态编码器实现
├── dataset_splits/                  # 数据划分格式说明
└── tests/                           # 无数据发布检查
```

## 快速开始

### 1. 安装环境

建议创建 Python 3.10 环境。请先安装与 CPU 或 CUDA 环境匹配的 PyTorch，再安装
其余依赖：

```bash
conda create -n somaalign python=3.10 -y
conda activate somaalign

# 请根据 pytorch.org 的说明选择适合当前平台的安装命令。
pip install torch torchvision
pip install -r requirements.txt
```

本代码已在 PyTorch 2.6 和 torchvision 0.21 环境中验证。

### 2. 准备 Checkpoint

SomaAlign 支持直接的 PyTorch state dictionary，也支持包含 `state_dict` 或
`model_state_dict` 的 checkpoint。权重需要与本仓库提供的编码器结构完全匹配。
预训练权重将在论文录用后发布。

### 3. 提取嵌入

```python
import torch

from inference import build_model, encode, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device)
load_checkpoint(model, "/path/to/somaalign_checkpoint.pth")

# 单个 RGB 时序窗口：[帧数, 通道数, 高, 宽]
rgb = torch.randn(3, 3, 224, 224)
embedding = encode(model, {"rgb": rgb}, device)["rgb"]  # [1, 1024]
```

支持的输入名称为 `rgb`、`photo`、`depth`、`tactile`、`imu` 和 `text`。
图像类时序 batch 使用 `[B,S,C,H,W]`，IMU 使用 `[B,C,T]`，分词后的文本使用
`[B,T]`。

## 数据组织

本仓库的数据适配器采用以下整体目录结构：

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
├── 标签.xlsx                  # 默认名称，可通过 --label_file_2f/3f 指定
└── text_annotations.json
```

建议使用相对于 2F 或 3F 数据根目录的样本 ID 自定义划分；也支持按稳定的目录字典序
定义从零开始的索引。格式详见 [split 说明](dataset_splits/README.md)。

```json
{
  "sample_ids": {
    "train": ["1_Object/1/0"],
    "val": ["1_Object/2/0"],
    "test": ["1_Object/3/0"]
  }
}
```

## 跨模态检索评测

```bash
python eval_retrieval.py \
  --checkpoint /path/to/somaalign_checkpoint.pth \
  --data_root_2f /path/to/2f \
  --data_root_3f /path/to/3f \
  --split_file_2f /path/to/split_2f.json \
  --split_file_3f /path/to/split_3f.json \
  --output results/retrieval.json
```

默认评估所选 split 中的全部样本。用户也可以抽取可复现的随机子集，或者精确指定
split 内的样本索引：

```bash
# 固定随机种子的随机子集
python eval_retrieval.py ... --sample_count 500 --seed 42

# 用户自定义的显式索引
python eval_retrieval.py ... --sample_indices_file indices.json

```

如果不提供 manifest，也可以通过完整命令生成指定随机种子的 70/15/15 划分：

```bash
python eval_retrieval.py \
  --checkpoint /path/to/somaalign_checkpoint.pth \
  --data_root_2f /path/to/2f \
  --data_root_3f /path/to/3f \
  --generate_split --seed 42 \
  --output results/retrieval.json
```

自动生成的是便于运行例程的样本级划分，并非论文指定的评测划分。如需按 trial 隔离，
请自行提供划分文件，并把同一 trial 的所有样本放在同一分区。

标签文件名不同时，分别使用 `--label_file_2f` 和 `--label_file_3f` 指定。

相似度排序按查询分块计算，因此完整 split 评测不会一次分配完整的方形相似度矩阵。
可通过 `--retrieval_query_batch_size` 调节显存/内存与速度的权衡，默认值为 `512`。

传感器到传感器的检索以同步观测作为正样本；传感器到文本采用去重后的描述
gallery；文本到传感器则将具有相同结构化描述的所有观测视为有效正样本。程序会
报告每个有向模态对的 Recall@1/5/10，并在所有任务之间计算宏平均。

评测结果会受到数据划分和 gallery 构成的影响。公开结果时应同时提供 split
manifest、样本选择方式、样本数量和随机种子。

## 代码检查

运行不依赖数据和 checkpoint 的 API 与检索测试：

```bash
python -m unittest discover -s tests -v
```

## 引用

预印版：[arXiv:2611.02794](https://arxiv.org/abs/2611.02794)。

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

## 致谢

双盲评审期间暂不公开致谢信息。

## 许可证

SomaAlign 采用 [CC BY-NC-SA 4.0 许可证](LICENSE)发布。
