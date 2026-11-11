# DINOv3 中文教程 🦖🦖🦖

本教程面向想在本机运行 **Meta AI 的 DINOv3** 项目的读者,涵盖三块内容:

1. **环境怎么搭** —— 依赖、Python/PyTorch 版本、安装步骤
2. **数据集怎么设置** —— ImageNet-1k / ImageNet-22k / ADE20K / NYU / COCO 的目录结构与元数据生成
3. **怎么运行** —— 加载预训练模型、推理、评估、训练

> 项目仓库结构:`Z:\1.CY-SPACE\CSJ\Dinov3\dinov3`(即 `dinov3` 代码库本身)。
> 后文中所有命令的「当前目录」都指这个仓库根目录,命令里的 `REPO_DIR` 也指它。

---

## 目录

- [0. 项目概览](#0-项目概览)
- [1. 环境搭建](#1-环境搭建)
- [2. 数据集设置](#2-数据集设置)
- [3. 模型权重下载](#3-模型权重下载)
- [4. 如何运行](#4-如何运行)
- [5. 常见问题](#5-常见问题)

---

## 0. 项目概览

DINOv3 是一族视觉基础模型(Vision Foundation Model),基于自监督学习,能产出**高质量的稠密特征**,在分类、分割、深度估计、目标检测等任务上无需微调即可取得优秀效果。

代码库目录结构(重点部分):

```
dinov3/
├── dinov3/                    # 核心 Python 包
│   ├── data/                  # 数据集与数据加载
│   │   └── datasets/          #   ImageNet / ImageNet22k / ADE20K / NYU / CocoCaptions
│   ├── configs/               # 配置(默认配置 + 各训练/评估 YAML)
│   ├── train/                 # 训练入口与元架构(SSL / 多蒸馏)
│   ├── eval/                  # 评估(kNN / 线性分类 / 分割 / 深度 / 文本)
│   ├── models/                # ViT 与 ConvNeXt 模型定义
│   ├── hub/                   # torch.hub 模型加载入口
│   └── run/                   # 提交器(submitit/SLURM)
├── notebooks/                 # 上手 notebook(PCA、前景分割、匹配、跟踪等)
├── conda.yaml                 # conda 环境定义(推荐)
├── requirements.txt           # pip 依赖
├── pyproject.toml             # 项目元数据与 lint 配置
└── setup.py                   # 安装脚本
```

**关键前置条件:**

- 操作系统:**官方只支持 Linux**(代码依赖 submitit / SLURM / `torch.distributed` / FSDP)。Windows 本机建议用 **WSL2** 运行。
- Python:**3.11**
- PyTorch:**>= 2.7.1**(README 要求),且**强烈建议装 CUDA 版**。
- 训练需要多卡(小模型 ViT-L 也要 32 张 GPU),单卡只能跑推理、小规模评估或极小的 toy 配置。

---

## 1. 环境搭建

### 1.1 准备 Python 环境

推荐用 `micromamba`(或 `conda`)创建名为 `dinov3` 的环境:

```bash
# 进入仓库根目录(包含 conda.yaml 的目录)
cd /path/to/dinov3

# 方式 A(推荐):用仓库自带的 conda 环境定义
micromamba env create -f conda.yaml
micromamba activate dinov3
```

`conda.yaml` 中定义的核心依赖如下(供你核对或手动安装):

```
python=3.11
omegaconf, pandas, scikit-learn, scikit-learn-intelex
submitit, termcolor
torch, torchvision, torchmetrics
ftfy, iopath, regex          # dino.txt 文本对齐需要
```

### 1.2 用 pip 手动安装(方式 B)

如果不用 conda,也可以先装好 PyTorch 再装其它依赖:

```bash
# 1) 先按官网安装匹配你 CUDA 版本的 PyTorch
#    https://pytorch.org/get-started/locally/
pip install torch torchvision

# 2) 安装其余依赖
pip install -r requirements.txt
```

`requirements.txt` 内容:

```
ftfy
omegaconf
regex
scikit-learn
submitit
termcolor
torch
torchmetrics
torchvision
```

### 1.3 安装 dinov3 包本身(可选)

评估与训练脚本都假设 `dinov3` 包在 Python 的模块搜索路径里。两种做法:

```bash
# 做法 1:直接 pip install(以可编辑模式)
pip install -e .

# 做法 2:不安装,每次运行时用 PYTHONPATH=. 前缀
PYTHONPATH=. python ...
```

> 强烈建议用 `PYTHONPATH=.` 前缀,这也是 README 里的标准写法。

### 1.4 Windows 用户特别说明

本机是 Windows 11。请优先考虑:

1. 安装 **WSL2 + Ubuntu**,在 WSL 里建环境(最省心,官方路径);
2. 或者用 Docker 拉一个带 CUDA 的 PyTorch 镜像;
3. 若坚持 Windows 原生跑,只能做**纯推理**(加载 backbone 前向一次),训练/分布式评估大概率会因 `torch.distributed` 的 NCCL 后端、`submitit` 等问题失败。

---

## 2. 数据集设置

数据集的通用约定:**用字符串描述数据集**,格式为

```
<数据集名>:split=<划分>:root=<数据根目录>:extra=<元数据目录>
```

例如:`ImageNet:split=TRAIN:root=/data/imagenet:extra=/data/imagenet`。
代码里解析该字符串的入口是 `dinov3/data/loaders.py` 的 `_parse_dataset_str`,支持的数据集名:`ImageNet`、`ImageNet22k`、`ADE20K`、`CocoCaptions`、`NYU`。

### 2.1 ImageNet-1k(ILSVRC2012)

**目录结构**(`<ROOT>` 为数据集根目录):

```
<ROOT>/
├── train/
│   └── n01440764/                       # 每个类别一个子目录
│       └── n01440764_10026.JPEG
│   └── ...                              # 共 1000 个类别目录
├── val/
│   └── n01440764/ILSVRC2012_val_00000293.JPEG
│   └── ...
├── test/
│   └── ILSVRC2012_test_00000001.JPEG
│   └── ...
└── labels.txt                           # 每行: <class_id>,<class_name>
```

**元数据文件**(`<EXTRA>` 目录,可与 `<ROOT>` 相同)需要以下 `.npy`:

```
<EXTRA>/class-ids-TRAIN.npy
<EXTRA>/class-ids-VAL.npy
<EXTRA>/class-names-TRAIN.npy
<EXTRA>/class-names-VAL.npy
<EXTRA>/entries-TEST.npy
<EXTRA>/entries-TRAIN.npy
<EXTRA>/entries-VAL.npy
```

**生成元数据**——用下面这段 Python 代码一次性生成(前提是 `dinov3` 包已可导入):

```python
from dinov3.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

### 2.2 ImageNet-22k

ImageNet-22k 使用「tar 打包 + block 日志」格式,官方代码默认路径较特殊,README 明确建议:**按你的本地数据布局自行改写** `dinov3/data/datasets/image_net_22k.py`。

核心逻辑(供改写参考):

- 数据根目录下每个类别一个 `<class_id>.tar`,以及 `blocks/<class_id>.log`(记录 block 偏移与文件名);
- 需要生成 `entries.npy` 和 `class-ids.npy` 两个元数据文件(在 `<EXTRA>` 目录);
- 通过 `ImageNet22k(...).dump_extra()` 生成。

> 训练命令里用 `ImageNet22k:root=<ROOT>:extra=<EXTRA>` 引用。

### 2.3 ADE20K(语义分割)

用于线性分割评估/推理。请先下载 ADE20K 数据,并按 `dinov3/eval/segmentation/configs/` 里配置的 `datasets.root` 指定根目录。数据集字符串形如 `ADE20K:split=<TRAIN|VAL>`。

### 2.4 NYUv2 Depth(深度估计)

详见 `DATASETS.md`,两种方式:

**方式一(推荐):直接从 BinsFormer 下载现成数据**。下载后确保目录结构为:

```
<DEPTH_DATASETS_ROOT>/NYU/
├── basement/[...]
├── basement_0001a/[...]
├── bathroom/[...]
├── ...                                  # 各类场景子目录
├── study_room_0005b/[...]
├── nyu_test.txt
└── nyu_train.txt
```

> 注意:若下载的目录名是小写 `nyu`,要重命名为大写 `NYU`。

**方式二:按 BTS 教程自己制作** 训练集/测试集,并下载 train/test 划分文件:

```bash
export DEPTH_DATASETS_ROOT=${HOME}/datasets
mkdir -p ${DEPTH_DATASETS_ROOT}/NYU
# 按 BTS README 制作 train/test 集后,下载划分文件:
wget https://github.com/cleinc/bts/blob/master/train_test_inputs/nyudepthv2_train_files_with_gt.txt -O ${DEPTH_DATASETS_ROOT}/NYU/nyu_train.txt
wget https://github.com/cleinc/bts/blob/master/train_test_inputs/nyudepthv2_test_files_with_gt.txt -O ${DEPTH_DATASETS_ROOT}/NYU/nyu_test.txt
```

### 2.5 CocoCaptions(文本对齐 / dino.txt)

`dino.txt` 文本对齐示例用 `CocoCaptions` 数据集做演示(论文实际用的是私有数据集)。数据集类在 `dinov3/data/datasets/coco_captions.py`,参考数据来源见 README。按需改写该类以匹配你的本地路径。

---

## 3. 模型权重下载

DINOv3 的权重需要**申请访问权限**:到 [DINOv3 下载页](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) 提交申请,通过后邮件会收到完整权重 URL 列表。

- **务必用 `wget` 下载**,不要用浏览器直接下。
- 拿到 URL 后,可以:
  1. 下载到本地,再通过 `weights=<本地路径>` 加载;或
  2. 直接把 URL 传给 `torch.hub.load()` 的 `weights` 参数自动下载。

主要模型(预训练于 LVD-1689M 网络数据):ViT-S/16、ViT-S+/16、ViT-B/16、ViT-L/16、ViT-H+/16、ViT-7B/16(distilled),以及 ConvNeXt Tiny/Small/Base/Large。

此外,所有 backbone 也都在 **HuggingFace Hub**(`facebook/dinov3-*` 集合)和 **timm**(>=1.0.20)、**transformers**(>=4.56.0)中可用,无需申请即可下载。

---

## 4. 如何运行

### 4.1 用 PyTorch Hub 加载 backbone(最简单,适合上手)

```python
import torch

REPO_DIR = "<dinov3 仓库本地路径>"

# 加载 ViT-B/16 backbone(weights 填本地路径或官方 URL)
dinov3_vitb16 = torch.hub.load(
    REPO_DIR, 'dinov3_vitb16', source='local', weights="<CHECKPOINT/URL/OR/PATH>"
)

# 其他可选入口:
# dinov3_vits16 / dinov3_vits16plus / dinov3_vitb16 / dinov3_vitl16
# dinov3_vith16plus / dinov3_vit7b16
# dinov3_convnext_tiny/small/base/large
```

### 4.2 用 HuggingFace Transformers 加载

```python
from transformers import pipeline
from transformers.image_utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction",
)
features = feature_extractor(image)
```

### 4.3 图像预处理(重要!)

LVD-1689M 权重(网络图片)用 **ImageNet 标准归一化**:

```python
import torch
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
```

SAT-493M 权重(卫星影像)用另一组均值方差:

```python
v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
```

### 4.4 训练

**所有训练命令都必须以 `PYTHONPATH=.` 开头**(或先 `pip install -e .`)。

#### 快速上手:在 ImageNet-1k 上训练 ViT-L/16

官方使用 SLURM 集群(4 节点 × 8 卡 = 32 卡):

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 4 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <输出目录> \
  train.dataset_path=ImageNet:root=<数据目录>:extra=<元数据目录>
```

> 训练时长约 14 小时,结果 k-NN 82.0%、线性评估 83.5%。
> 每 12500 迭代会在 `eval` 目录保存 teacher 权重供评估。

#### 不用 SLURM,直接用 python/torchrun 启动

没有 SLURM 时,可以直接跑(单机多卡用 `torchrun`):

```bash
PYTHONPATH=. python dinov3/train/train.py \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <输出目录> \
  train.dataset_path=ImageNet:root=<数据目录>:extra=<元数据目录>
```

> 注意:`dinov3.run.submit` 依赖 submitit + SLURM;本地单机请走上面的直接启动方式,并用 `torchrun --nproc_per_node=<N>` 做多卡。

#### ViT-7B 三阶段训练(私有数据,仅供了解)

官方 ViT-7B 分三阶段:**预训练 → Gram 锚定 → 高分辨率适配**,均用 `--nodes 32`(256 卡)。对应配置:

- `dinov3/configs/train/dinov3_vit7b16_pretrain.yaml`
- `dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml`
- `dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml`

#### 蒸馏 / 多蒸馏(可选)

- ConvNeXt 蒸馏配置在 `dinov3/configs/train/distillation_convnext/`
- 多蒸馏测试配置:`dinov3/configs/train/multi_distillation_test.yaml`(加 `--multi-distillation` 开关)

### 4.5 评估

训练时定期保存 teacher 权重,对 checkpoint 做评估:

#### k-NN 分类(ImageNet-1k)

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/knn.py \
  model.config_file=<输出目录>/config.yaml \
  model.pretrained_weights=<输出目录>/teacher_checkpoint.pth \
  output_dir=<输出目录> \
  train.dataset=ImageNet:split=TRAIN:root=<数据目录>:extra=<元数据目录> \
  eval.test_dataset=ImageNet:split=VAL:root=<数据目录>:extra=<元数据目录>
```

#### 逻辑回归分类

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/log_regression.py \
  model.config_file=<输出目录>/config.yaml \
  model.pretrained_weights=<输出目录>/teacher_checkpoint.pth \
  output_dir=<输出目录> \
  train.dataset=ImageNet:split=TRAIN:root=<数据目录>:extra=<元数据目录> \
  eval.test_dataset=ImageNet:split=VAL:root=<数据目录>:extra=<元数据目录>
```

#### 线性分类(带数据增强)

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/linear.py \
  model.config_file=<输出目录>/config.yaml \
  model.pretrained_weights=<输出目录>/teacher_checkpoint.pth \
  output_dir=<输出目录> \
  train.dataset=ImageNet:split=TRAIN:root=<数据目录>:extra=<元数据目录> \
  train.val_dataset=ImageNet:split=VAL:root=<数据目录>:extra=<元数据目录>
```

#### 语义分割(ADE20K,线性头)

```bash
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
  model.dino_hub=dinov3_vit7b16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=<ADE20K 根目录> \
  --output-dir <输出目录>
```

训练结束后在输出目录会生成:`segmentation_config.yaml`、`model_final.pth`、`results-semantic-segmentation.csv`。

#### 深度估计(NYUv2 Depth)

```bash
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/depth/run.py \
  model.dino_hub=dinov3_vit7b16 \
  config=dinov3/eval/depth/configs/config-nyu.yaml \
  datasets.root=<NYU 根目录> \
  --output-dir <输出目录>
```

#### 文本对齐(dino.txt)

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/text/train_dinotxt.py \
  --nodes 4 \
  trainer_config_file="<文本对齐配置路径,如 dinov3/eval/text/configs/dinov3_vitl_text.yaml>" \
  --output-dir <输出目录>
```

### 4.6 加载预训练「头」(分类/深度/检测/分割)

在 backbone 之外,仓库还提供在特定数据集上训好的「头」,用 torch.hub 一并加载:

```python
import torch
REPO_DIR = "<dinov3 仓库本地路径>"

# 图像分类(ViT-7B + ImageNet)
cls = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_lc', source="local",
                     weights="<HEAD 权重>", backbone_weights="<BACKBONE 权重>")

# 深度估计(ViT-7B + SYNTHMIX)
depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local",
                         weights="<HEAD 权重>", backbone_weights="<BACKBONE 权重>")

# 目标检测(ViT-7B + COCO2017)
detector = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_de', source="local",
                          weights="<HEAD 权重>", backbone_weights="<BACKBONE 权重>")

# 语义分割(ViT-7B + ADE20K)
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local",
                           weights="<HEAD 权重>", backbone_weights="<BACKBONE 权重>")
```

### 4.7 Notebook 上手(零训练门槛)

仓库自带多个 notebook,建议先跑这些熟悉模型用法(可在 Google Colab 直接运行):

| Notebook | 作用 |
|---|---|
| `notebooks/pca.ipynb` | 展示 patch 特征的 PCA 可视化(论文里的彩虹图) |
| `notebooks/foreground_segmentation.ipynb` | 基于 DINOv3 特征训练线性前景分割 |
| `notebooks/dense_sparse_matching.ipynb` | 两张图的 patch 稠密/稀疏匹配 |
| `notebooks/segmentation_tracking.ipynb` | 视频分割跟踪 |
| `notebooks/dinotxt_segmentation_inference.ipynb` | dino.txt 零样本分割 |

---

## 5. 常见问题

**Q1:训练报错说找不到 `dinov3` 模块?**
→ 命令前加 `PYTHONPATH=.`,或先 `pip install -e .`。

**Q2:Windows 上能不能跑?**
→ 官方只支持 Linux。推理(前向)可行,训练/分布式评估建议用 WSL2 或 Docker。

**Q3:`torch.__version__` 版本要求?**
→ 训练代码断言 `>= 2.1`,但 README 要求 `>= 2.7.1`,建议按 `>= 2.7.1` 装 CUDA 版。

**Q4:下载权重用浏览器失败了?**
→ 官方明确要求用 `wget`,浏览器下载可能被拒绝。

**Q5:ImageNet 元数据 `.npy` 怎么来?**
→ 运行 `ImageNet(split=..., root=..., extra=...).dump_extra()`,见 [2.1 节](#21-imagenet-1kilsvrc2012)。

**Q6:没有 SLURM 集群怎么办?**
→ 用 `python dinov3/train/train.py` 或 `torchrun` 直接启动,跳过 `dinov3.run.submit`。

**Q7:单卡能训练吗?**
→ 小模型 toy 配置或许能跑通,但论文配置(如 ViT-L)需要 32 卡,单卡显存/算力都不够。单卡建议只做推理和评估。

---

## 附:快速命令速查

```bash
# 建环境
micromamba env create -f conda.yaml && micromamba activate dinov3

# 生成 ImageNet-1k 元数据
PYTHONPATH=. python -c "from dinov3.data.datasets import ImageNet; [ImageNet(split=s, root='<ROOT>', extra='<EXTRA>').dump_extra() for s in ImageNet.Split]"

# 加载 backbone(推理)
PYTHONPATH=. python -c "import torch; m = torch.hub.load('.', 'dinov3_vitb16', source='local', weights='<权重路径>')"

# 训练 ViT-L/16(单机直接启动示例)
PYTHONPATH=. python dinov3/train/train.py \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir ./out \
  train.dataset_path=ImageNet:root=<数据目录>:extra=<元数据目录>
```
