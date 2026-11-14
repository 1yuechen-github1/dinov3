# DINOv3：本地预训练 ViT-L/16，再用自己的数据微调

本文只讲一个目标：**使用本仓库在本地对自己的图片做 DINOv3 自监督预训练，然后加载预训练 backbone 做下游微调**。

不把“加载官方权重做推理”、ImageNet 评测、ADE20K/NYUv2 评测和 SLURM 集群复现当作你的主流程；这些内容与“用自己的数据训练”不是一回事。

## 先说清楚：仓库代码能做什么

仓库已经包含 DINOv3 的 ViT、teacher-student 自监督训练器、数据加载器和评估代码。训练入口是：

```text
dinov3/train/train.py
```

训练配置在：

```text
dinov3/configs/train/
dinov3/configs/ssl_default_config.yaml
```

但是，官方配置主要针对大规模集群和 ImageNet/22k 数据。你要在本地使用自己的数据，必须准备：

1. 一个能被仓库数据加载器识别的数据集描述；
2. 与之匹配的 Dataset 类或元数据文件；
3. 一份适合本机显存和图片数量的训练配置。

**不能只把 ImageNet 路径替换成自己的文件夹，然后直接得到可靠训练结果。**

## 你的完整流程

```text
自己的无标注图片
        ↓
DINOv3 自监督预训练 ViT-L/16
        ↓  保存 teacher_checkpoint.pth
自己的有标注数据
        ↓
冻结 backbone 做线性头微调，或解冻部分/全部 backbone 做全量微调
        ↓
得到分类、分割、检测或回归模型
```

预训练阶段不需要类别标签；微调阶段才需要你的任务标签。

## 第一阶段：自己的数据做自监督预训练

### 1. 数据组织（你的情况：A）

你的图片全部直接放在同一个文件夹，不需要类别子目录：

```text
my_pretrain_data/
├── img001.jpg
├── img002.jpg
├── img003.png
└── img004.jpeg
```

可以有多个层级的子目录，但不是必须的。预训练阶段不读取文件名和类别标签，只读取图片本身。支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.tif`、`.tiff`。

### 2. 数据加载器

现有数据集解析入口：

```text
dinov3/data/loaders.py
dinov3/data/datasets/
```

仓库现在已经提供了 `ImageFolder`，用于递归读取普通图片文件夹；它不读取类别标签，正适合自监督预训练。代码位于 `dinov3/data/datasets/image_folder.py`，并已在 `dinov3/data/loaders.py` 注册。

```text
ImageFolder:root=D:/data/my_pretrain
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.tif`、`.tiff`，会递归扫描所有子目录。DINO 的多裁剪增强由训练数据管线负责。

### 3. 训练配置

以 ViT-L/16 配置为起点：

```text
dinov3/configs/train/vitl_im1k_lin834.yaml
```

复制一份到自己的配置文件，再修改：

- `train.dataset_path`：改成自己的 Dataset 字符串；
- `train.output_dir`：改成 checkpoint 输出目录；
- batch size、`num_workers`、训练迭代数：按显存和数据规模缩小；
- global crop/local crop 尺寸：按图片分辨率调整；
- 学习率和 warmup：按总 batch size 重新设定。

官方 ViT-L 配置是为多节点多卡准备的，不能原样当作单机配置。显存不足时，优先降低 batch size、裁剪尺寸和迭代数；必要时使用梯度累积或梯度检查点。

### 4. 启动训练

训练入口：

```text
python dinov3/train/train.py \
  --config-file dinov3/configs/train/my_vitl16.yaml \
  --output-dir <输出目录> \
  train.dataset_path=ImageFolder:root=<自己的数据目录>
```

例如你的图片在 `D:\data\my_pretrain`，Windows PowerShell 命令是：

```powershell
python dinov3/train/train.py `
  --config-file dinov3/configs/train/my_vitl16.yaml `
  --output-dir D:/dinov3_runs/vitl16_pretrain `
  train.dataset_path=ImageFolder:root=D:/data/my_pretrain
```

注意：路径中不要再写 `:split=TRAIN`，因为这种图片文件夹数据集没有预定义的 TRAIN/VAL 划分。若以后需要验证集，应分别准备两个目录并启动两个数据集配置。

单机多卡时使用 `torchrun` 启动同一个训练入口；`dinov3.run.submit` 是给 submitit/SLURM 集群提交任务的，不是本地训练的必要步骤。

训练输出中重点关注：

```text
teacher_checkpoint.pth
student_checkpoint.pth
config.yaml
```

后续微调通常使用 teacher checkpoint。

## 第二阶段：用自己的标注数据微调

微调方式取决于任务：

### 图像分类

可以参考：

```text
dinov3/eval/linear.py
dinov3/eval/log_regression.py
```

这类代码默认是“冻结 DINO backbone，只训练线性分类头”。如果你要真正微调 ViT-L，需要在此基础上取消 backbone 冻结，并设置更小的 backbone 学习率。

### 语义分割

参考：

```text
dinov3/eval/segmentation/
```

将数据集类、类别数、标签读取方式和配置中的 `datasets.root` 改成自己的数据格式。

### 深度估计、检测或回归

分别参考：

```text
dinov3/eval/depth/
dinov3/eval/detection/
```

同样需要替换数据集读取和任务头配置；不能直接使用 NYU、COCO 配置期待你的数据自动适配。

## “预训练”和“微调”的区别

| 阶段 | 是否需要标签 | 训练什么 | 输入数据 |
|---|---:|---|---|
| 自监督预训练 | 不需要 | teacher/student ViT 和投影头 | 大量无标注图片 |
| 线性评估 | 需要 | 只训练线性头 | 有标注图片 |
| 全量微调 | 需要 | 任务头 + 部分或全部 ViT | 有标注图片 |

## 需要避免的误导

- `torch.hub.load(... weights=...)` 是加载已有权重做推理，不是从你的数据开始预训练。
- ImageNet 的 `.npy` 元数据只适用于对应的 ImageNet 数据布局；自己的数据不要照抄。
- `dinov3.run.submit` 和 `--nodes 4` 是集群提交方式，不代表本地必须 4 节点 32 卡。
- ViT-L/16 很大；单卡本地可以用于调通流程，但正式预训练通常需要多卡、较大数据量和很长训练时间。
- 只用很少的医学图片直接“预训练 DINOv3”通常不稳定。更实际的方案是加载已有 ViT-L/16 权重，再用你的数据微调；如果坚持从零预训练，需要显著扩大无标注数据并重新调参。

## 下一步需要确定的三件事

要把本仓库改成真正可运行的“你的数据 → ViT-L/16 预训练 → 微调”流程，还需要明确：

1. 你的图片目录结构和图片数量；
2. 预训练使用单卡还是多卡、显存大小；
3. 微调任务是分类、分割、检测、深度估计还是别的任务。

确定后，应先实现/验证 `MyDataset`，再复制并缩小 ViT-L 配置，最后跑一个很短的 smoke test，确认数据加载、loss 和 checkpoint 都正常，再开始长时间训练。
