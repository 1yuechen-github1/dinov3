# Copyright (c) Meta Platforms, Inc. and affiliates.

"""A simple recursive image-folder dataset for self-supervised pretraining."""

import os
from typing import Any, Callable, List, Optional

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class ImageFolder(ExtendedVisionDataset):
    """Read images recursively from ``root``; labels are intentionally ignored."""

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(
        self,
        *,
        root: str,
        split: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        search_root = os.path.join(root, split) if split and os.path.isdir(os.path.join(root, split)) else root
        self.samples: List[str] = []
        for dirpath, _, filenames in os.walk(search_root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in self._EXTENSIONS:
                    self.samples.append(os.path.join(dirpath, filename))
        self.samples.sort()
        if not self.samples:
            raise RuntimeError(f'No images found under "{search_root}"')

    def get_image_data(self, index: int) -> bytes:
        with open(self.samples[index], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return None

    def __len__(self) -> int:
        return len(self.samples)
