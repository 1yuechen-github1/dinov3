# -*- coding: utf-8 -*-
"""
DINOv3 ViT-L/16 图像分类 —— 冻结 backbone + 线性探针（linear probe）

适合小数据集（几百 ~ 几千张标注图片）。只训练最后一个线性分类头，
backbone 冻结，不易过拟合。

数据目录结构（ImageFolder 格式，每个子目录一个类别）：
    data_root/
      ├── 类别A/
      │     ├── img1.jpg
      │     └── img2.jpg
      └── 类别B/
            └── img3.jpg

依赖：torch（GPU 版）、transformers、torchvision
运行示例（在有 GPU 的机器上）：
    python linear_probe.py --data D:/your/images --model ./dinov3-vitl16
"""

import argparse
import faulthandler
import platform
import random
from pathlib import Path

# 在导入原生扩展之前启用，Windows 访问冲突时尽量输出 Python 调用栈。
faulthandler.enable(all_threads=True)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_curve, auc)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(model, loader, device, debug_cuda=False):
    """用冻结的 backbone 提取所有图片的 CLS 特征。"""
    model.eval()
    feats, labels = [], []
    for batch_idx, (images, ys) in enumerate(loader, start=1):
        print(f"[features {batch_idx}/{len(loader)}] images loaded", flush=True)
        images = images.to(device)
        if debug_cuda and device == "cuda":
            torch.cuda.synchronize()
        print(f"[features {batch_idx}/{len(loader)}] forward start", flush=True)
        out = model.forward_features(images) if hasattr(model, "embed_dim") else model(pixel_values=images)
        if debug_cuda and device == "cuda":
            torch.cuda.synchronize()
        cls = out["x_norm_clstoken"] if isinstance(out, dict) else out.last_hidden_state[:, 0]
        feats.append(cls.cpu())
        labels.append(ys)
        print(f"[features {batch_idx}/{len(loader)}] copied to CPU", flush=True)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


class FeatureDataset(Dataset):
    """把已提取的特征包装成 Dataset，供训练线性头用。"""

    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i], self.labels[i]


def stratified_split(dataset, val_ratio: float, seed: int):
    """按类别做分层划分，保证每个类别在训练/验证集里都有。"""
    labels = torch.tensor([y for _, y in dataset])
    classes = labels.unique().tolist()
    train_idx, val_idx = [], []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=False).view(-1).tolist()
        random.Random(seed).shuffle(idx)
        if len(idx) < 2:
            raise ValueError(f"类别 {c} 至少需要两张图片以划分训练/验证集")
        n_val = min(len(idx) - 1, max(1, int(len(idx) * val_ratio)))
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def collect_predictions(head, loader, device):
    head.eval(); ys_all, pred_all, prob_all = [], [], []
    for feats, ys in loader:
        logits = head(feats.to(device)); prob = logits.softmax(1).cpu()
        ys_all.append(ys); pred_all.append(prob.argmax(1)); prob_all.append(prob)
    return torch.cat(ys_all).numpy(), torch.cat(pred_all).numpy(), torch.cat(prob_all).numpy()


@torch.no_grad()
def collect_loss(head, loader, device, criterion):
    head.eval(); total_loss = 0.0; total = 0
    for feats, ys in loader:
        ys = ys.to(device)
        loss = criterion(head(feats.to(device)), ys)
        total_loss += loss.item() * ys.numel(); total += ys.numel()
    return total_loss / max(total, 1)


def metric_report(y_true, y_pred, y_prob, class_names, output_dir, prefix, make_plots=True, write_files=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }
    import json
    if write_files:
        (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if not make_plots:
            raise ImportError
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names) * .7)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180); plt.close()
        plt.figure(figsize=(7, 6))
        for c, name in enumerate(class_names):
            if len(np.unique(y_true == c)) == 2:
                fpr, tpr, _ = roc_curve((y_true == c).astype(int), y_prob[:, c]); plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / f"{prefix}_roc.png", dpi=180); plt.close()
        plt.figure(figsize=(7, 6))
        for c, name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve((y_true == c).astype(int), y_prob[:, c]); plt.plot(recall, precision, label=name)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / f"{prefix}_pr.png", dpi=180); plt.close()
    except ImportError:
        print("未安装 matplotlib/seaborn，跳过图像生成", flush=True)
    print(f"[{prefix}] Accuracy={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f} Weighted-F1={metrics['weighted_f1']:.4f}", flush=True)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0), flush=True)
    return metrics


def save_error_cases(y_true, y_pred, dataset, val_indices, class_names, output_dir, prefix):
    rows = []
    for i, (y, p) in enumerate(zip(y_true, y_pred)):
        path, _ = dataset.samples[val_indices[i]]
        if y == p:
            kind = "TP"
        else:
            # In multiclass classification a wrong sample is FN for its true
            # class and FP for the predicted class; TN is the remaining
            # one-vs-rest population and cannot be assigned to one image.
            kind = "FN+FP"
        rows.append(f"{kind}\t{class_names[y]}\t{class_names[p]}\t{path}")
    (output_dir / f"{prefix}_samples.tsv").write_text("type\ttrue\tpredicted\tpath\n" + "\n".join(rows), encoding="utf-8")


def train_linear_head(model, train_loader, val_loader, device, epochs, lr, num_classes, class_names, output_dir):
    """训练一个线性分类头（logistic regression），带早停。"""
    feature_dim = model.embed_dim if hasattr(model, "embed_dim") else model.config.hidden_size
    head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_score, best_state, final_state, patience = -1.0, None, None, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_accuracy": [], "val_macro_f1": []}
    for epoch in range(epochs):
        head.train()
        total, correct, loss_sum = 0, 0, 0.0
        for feats, ys in train_loader:
            feats, ys = feats.to(device), ys.to(device)
            logits = head(feats)
            loss = criterion(logits, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * ys.numel()
            total += ys.numel()
            correct += (logits.argmax(1) == ys).sum().item()

        train_acc = correct / total
        yv, pv, probv = collect_predictions(head, val_loader, device)
        val_metrics = metric_report(yv, pv, probv, class_names, output_dir, "val_current", make_plots=False, write_files=False)
        val_score = val_metrics["macro_f1"]
        val_loss = collect_loss(head, val_loader, device, criterion)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(loss_sum / max(total, 1))
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_macro_f1"].append(val_score)

        if val_score > best_score:
            best_score, best_state, patience = val_score, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 0
            torch.save({"head": best_state, "classes": class_names, "best_macro_f1": best_score}, output_dir / "best_model.pth")
        else:
            patience += 1
        print(f"epoch {epoch + 1:3d}/{epochs}  train_acc={train_acc:.4f}  val_macro_f1={val_score:.4f}")

        # if patience >= 20:  # 20 个 epoch 无提升就早停
        #     print("early stopping")
        #     break

        final_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_score, best_state, final_state, history


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    correct, total = 0, 0
    for feats, ys in loader:
        feats, ys = feats.to(device), ys.to(device)
        correct += (head(feats).argmax(1) == ys).sum().item()
        total += ys.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="标注图片根目录（ImageFolder 结构）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="HuggingFace 模型目录")
    source.add_argument("--weights", help="官方 teacher_checkpoint.pth")
    parser.add_argument("--config", help="预训练 yaml；默认查找权重上级目录的 config.yaml，否则使用仓库 ViT-L 配置")
    parser.add_argument("--output", default="linear_head.pth", help="分类头输出路径")
    parser.add_argument("--metrics-dir", default="classification_metrics", help="评估报告和图像输出目录")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=None,
                        help="排查 attention 原生算子崩溃时使用 eager")
    parser.add_argument("--debug-cuda", action="store_true",
                        help="在数据传输和模型前向后同步 CUDA，缩小崩溃位置范围")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("val-ratio 必须在 0 到 1 之间，epochs 和 batch-size 必须为正数")

    print(f"Python: {platform.python_version()}, torch: {torch.__version__}, "
          f"torch CUDA: {torch.version.cuda}", flush=True)
    print("[startup] checking CUDA", flush=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"using device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    set_seed(args.seed)

    # 1. 图片预处理（与官方一致：resize 224 + 归一化）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. 加载数据集
    print("[data] scanning images", flush=True)
    dataset = datasets.ImageFolder(args.data, transform=transform)
    print(f"共 {len(dataset)} 张图片，{len(dataset.classes)} 个类别：{dataset.classes}")

    # 3. 加载冻结的 backbone
    print("[model] loading weights on CPU", flush=True)
    model_kwargs = {}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    config_path = None
    if args.weights:
        from omegaconf import OmegaConf
        from dinov3.models import build_model
        repo_root = Path(__file__).resolve().parent
        config_path = Path(args.config) if args.config else next(
            (p / "config.yaml" for p in Path(args.weights).resolve().parents if (p / "config.yaml").is_file()),
            repo_root / "dinov3/configs/train/pretrain_vitl16.yaml",
        )
        print(f"[model] config: {config_path}", flush=True)
        cfg = OmegaConf.merge(
            OmegaConf.load(repo_root / "dinov3/configs/ssl_default_config.yaml"),
            OmegaConf.load(config_path),
        )
        cfg.student.fp8_enabled = False
        model, _ = build_model(cfg.student, only_teacher=True, img_size=224, device="meta")
        model.to_empty(device="cpu")
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("teacher", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if any(k.startswith("backbone.") for k in state):
            state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        # Training checkpoints may contain auxiliary storage-token and
        # qkv bias-mask buffers that are not needed for frozen CLS features.
        # Load the backbone compatibly and fail only on real tensor shape errors.
        msg = model.load_state_dict(state, strict=False)
        unexpected = [k for k in msg.unexpected_keys
                      if k != "storage_tokens" and not k.endswith(".bias_mask")]
        if unexpected or msg.missing_keys:
            raise RuntimeError(
                "预训练权重与当前 ViT 配置结构不匹配。\n"
                f"missing keys: {msg.missing_keys[:20]}\n"
                f"unexpected keys: {unexpected[:20]}"
            )
        print(f"[model] ignored auxiliary checkpoint keys: {len(msg.unexpected_keys)}", flush=True)
        del checkpoint, state
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    model.requires_grad_(False)
    print(f"[model] moving weights to {device}", flush=True)
    model.to(device)
    if args.debug_cuda and device == "cuda":
        torch.cuda.synchronize()
    print("[model] ready", flush=True)

    # 4. 提取所有图片特征
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("[features] loading first batch", flush=True)
    feats, labels = extract_features(model, loader, device, args.debug_cuda)
    print(f"特征形状: {feats.shape}")

    # 5. 分层划分 train/val
    feat_dataset = FeatureDataset(feats, labels)
    train_ds, val_ds = stratified_split(feat_dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 6. 训练线性头
    head, best_acc, best_state, final_state, history = train_linear_head(
        model, train_loader, val_loader, device, args.epochs, args.lr,
        len(dataset.classes), dataset.classes, Path(args.metrics_dir)
    )
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    import json
    (metrics_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5)); plt.plot(history["epoch"], history["train_loss"], label="train loss"); plt.plot(history["epoch"], history["val_loss"], label="val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(metrics_dir / "loss_curve.png", dpi=180); plt.close()
    except ImportError:
        print("未安装 matplotlib，跳过 loss 曲线", flush=True)
    # Evaluate each saved model once, after training is complete.
    head.load_state_dict(best_state)
    yb, pb, probb = collect_predictions(head, val_loader, device)
    best_metrics = metric_report(yb, pb, probb, dataset.classes, metrics_dir, "best")
    save_error_cases(yb, pb, dataset, val_ds.indices, dataset.classes, Path(args.metrics_dir), "best")

    # 7. 保存线性头
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Final model and final validation report (best_model.pth is selected by Macro-F1).
    head.load_state_dict(final_state)
    yv, pv, probv = collect_predictions(head, val_loader, device)
    final_metrics = metric_report(yv, pv, probv, dataset.classes, metrics_dir, "final")
    save_error_cases(yv, pv, dataset, val_ds.indices, dataset.classes, metrics_dir, "final")
    torch.save({"head": head.state_dict(), "classes": dataset.classes,
                "weights": args.weights, "model": args.model,
                "config": str(config_path) if config_path else None,
                "feature_dim": feats.shape[1], "best_val_acc": best_acc,
                "image_size": 224, "metrics": best_metrics}, out_path)
    torch.save({"head": head.state_dict(), "classes": dataset.classes,
                "best_macro_f1": best_acc, "metrics": final_metrics}, out_path.parent / "final_model.pth")
    print(f"\n完成！验证集最佳准确率: {best_acc:.4f}")
    print(f"线性头已保存到: {out_path}")


if __name__ == "__main__":
    main()
