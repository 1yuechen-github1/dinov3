import os
from typing import Any, Callable, Optional
from PIL import Image
from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class CustomSinus(ExtendedVisionDataset):
    """Custom semantic segmentation dataset: images/<split> and masks/<split>.

    Mask pixels must be integer class ids: 0 background, 1 mucosa, 2 sidewall.
    """
    def __init__(self, *, root: str, split: str, transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 image_decoder: Decoder = ImageDataDecoder,
                 target_decoder: Decoder = DenseTargetDecoder) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform,
                         target_transform=target_transform, image_decoder=image_decoder,
                         target_decoder=target_decoder)
        image_dir = os.path.join(root, "images", split)
        mask_dir = os.path.join(root, "masks", split)
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"需要目录: {image_dir} 和 {mask_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = []
        for name in sorted(os.listdir(image_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() in extensions:
                mask = os.path.join(mask_dir, stem + ".png")
                if not os.path.isfile(mask):
                    raise FileNotFoundError(f"找不到对应 mask: {mask}")
                self.samples.append((os.path.join(image_dir, name), mask))
        if not self.samples:
            raise RuntimeError(f"没有找到图片: {image_dir}")

    def get_image_data(self, index: int) -> bytes:
        with open(self.samples[index][0], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        with open(self.samples[index][1], "rb") as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.samples)
