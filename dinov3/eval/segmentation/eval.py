# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
from pathlib import Path
import json
import numpy as np
from PIL import Image

import torch

import dinov3.distributed as distributed
from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
)
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms
from dinov3.hub.segmentors import dinov3_vit7b16_ms
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


@torch.no_grad()
def evaluate_segmentation_model(
    segmentation_model: torch.nn.Module,
    test_dataloader,
    device,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    autocast_dtype,
    inference_mode="whole",
    reduce_zero_label=False,
    debug_dir=None,
):
    segmentation_model = segmentation_model.to(device)
    segmentation_model.eval()
    all_metric_values = []
    debug_count = 0
    metric_logger = MetricLogger(delimiter="  ")

    for batch_img, (_, gt) in metric_logger.log_every(test_dataloader, 10, header="Validation: "):
        batch_img = [img.to(device).to(dtype=autocast_dtype) for img in batch_img]
        gt = gt.to(device)[0]
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        if gt.ndim != 2:
            raise ValueError(f"Expected 2D class-id mask, got {tuple(gt.shape)}")
        aggregated_preds = torch.zeros(1, num_classes, gt.shape[-2], gt.shape[-1])
        for img_idx, img in enumerate(batch_img):
            prediction = make_inference(
                img,
                segmentation_model.module,
                inference_mode=inference_mode,
                decoder_head_type=decoder_head_type,
                rescale_to=gt.shape[-2:],
                n_output_channels=num_classes,
                crop_size=(eval_res, eval_res),
                stride=(eval_stride, eval_stride),
                apply_horizontal_flip=(img_idx and img_idx >= len(batch_img) / 2),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
            aggregated_preds += prediction.detach().to(device="cpu", dtype=aggregated_preds.dtype)
            del prediction
        aggregated_preds = (aggregated_preds / len(batch_img)).argmax(dim=1, keepdim=True).to(device)
        if debug_dir and debug_count < 10 and distributed.is_main_process():
            folder = Path(debug_dir)
            folder.mkdir(parents=True, exist_ok=True)
            truth = gt.cpu().numpy()
            predicted = aggregated_preds[0, 0].cpu().numpy()
            Image.fromarray(truth.astype(np.uint8)).save(folder / f"{debug_count:02d}_true_ids.png")
            Image.fromarray(predicted.astype(np.uint8)).save(folder / f"{debug_count:02d}_pred_ids.png")
            # Use one shared palette for truth and prediction (class 0 is black).
            palette = np.array([[0,0,0],[230,70,70],[70,200,90],[70,120,240],[240,200,50]], dtype=np.uint8)
            if num_classes <= len(palette):
                Image.fromarray(palette[np.clip(truth, 0, len(palette)-1)]).save(folder / f"{debug_count:02d}_true_color.png")
                Image.fromarray(palette[predicted]).save(folder / f"{debug_count:02d}_pred_color.png")
            base = test_dataloader.dataset.dataset if hasattr(test_dataloader.dataset, "dataset") else None
            mean = torch.tensor([.485,.456,.406]).view(3,1,1)
            std = torch.tensor([.229,.224,.225]).view(3,1,1)
            original = (batch_img[0][0].float().cpu() * std + mean).clamp(0,1)
            Image.fromarray((original.permute(1,2,0).numpy()*255).astype(np.uint8)).save(folder / f"{debug_count:02d}_image.png")
            stats = {"true_counts": dict(zip(*[v.tolist() for v in np.unique(truth, return_counts=True)])),
                     "pred_counts": dict(zip(*[v.tolist() for v in np.unique(predicted, return_counts=True)]))}
            (folder / f"{debug_count:02d}_counts.json").write_text(json.dumps(stats, indent=2))
            debug_count += 1
        intersect_and_union = calculate_intersect_and_union(
            aggregated_preds[0, 0],
            gt,
            num_classes=num_classes,
            reduce_zero_label=reduce_zero_label,
        )
        all_metric_values.append(intersect_and_union)
        del img, gt, aggregated_preds, intersect_and_union

    all_metric_values = torch.stack(all_metric_values)
    if distributed.is_enabled():
        all_metric_values = torch.cat(distributed.gather_all_tensors((all_metric_values)))
    final_metrics = calculate_segmentation_metrics(
        all_metric_values,
        metrics=["mIoU", "dice", "fscore"],
    )
    final_metrics = {k: round(v.cpu().item() * 100, 2) for k, v in final_metrics.items()}
    logger.info(final_metrics)
    return final_metrics


def test_segmentation(backbone, config):
    # 1- construct a segmentation decoder
    if config.load_from == "dinov3_vit7b16_ms":  # torch hub descriptor
        # Load public m2f head checkpoints
        logger.info("Loading the 7B backbone and the M2F adapter with torchhub")
        segmentation_model = dinov3_vit7b16_ms(autocast_dtype=config.model_dtype.autocast_dtype, check_hash=True)
    else:
        segmentation_model = build_segmentation_decoder(
            backbone,
            config.decoder_head.backbone_out_layers,
            config.decoder_head.type,
            hidden_dim=config.decoder_head.hidden_dim,  # Only used for instantiating a M2F head
            num_classes=config.decoder_head.num_classes,
            autocast_dtype=config.model_dtype.autocast_dtype,
            dropout=config.decoder_head.dropout,
        )
        state_dict = torch.load(config.load_from, map_location="cpu")["model"]
        _, _ = segmentation_model.load_state_dict(state_dict, strict=False)
    device = distributed.get_rank()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(segmentation_model.to(device), device_ids=[device])

    # 2- dataloader for testing
    eval_res = config.eval.crop_size
    eval_stride = config.eval.stride
    transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
        use_tta=config.eval.use_tta,
        tta_ratios=config.transforms.eval.tta_ratios,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    test_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=transforms,
        )
    )

    test_sampler_type = None
    if distributed.is_enabled():
        test_sampler_type = SamplerType.DISTRIBUTED

    test_dataloader = make_data_loader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=test_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=config.num_workers > 0,
    )

    # 3- make inference
    return evaluate_segmentation_model(
        segmentation_model=segmentation_model,
        test_dataloader=test_dataloader,
        device=device,
        eval_res=eval_res,
        eval_stride=eval_stride,
        decoder_head_type=config.decoder_head.type,
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        inference_mode=config.eval.mode,
        reduce_zero_label=config.eval.reduce_zero_label,
        debug_dir=Path(config.output_dir) / "prediction_check",
    )
