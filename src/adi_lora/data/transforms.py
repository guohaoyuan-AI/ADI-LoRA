"""Image transforms for interpolation-shift experiments.

The project setting is low-resolution CIFAR images adapted to an ImageNet-pretrained
ViT.  Images are resized to the ViT input size with an explicitly controlled
interpolation kernel.  This is the experimental variable, so do not silently
change it through torchvision defaults.
"""

from __future__ import annotations

from typing import Literal

from torchvision import transforms
from torchvision.transforms import InterpolationMode

InterpolationName = Literal["bicubic", "bilinear", "nearest"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_interpolation_mode(name: str) -> InterpolationMode:
    normalized = str(name).strip().lower()
    if normalized == "bicubic":
        return InterpolationMode.BICUBIC
    if normalized == "bilinear":
        return InterpolationMode.BILINEAR
    if normalized == "nearest":
        return InterpolationMode.NEAREST
    raise ValueError(f"Unsupported interpolation={name!r}; use bicubic, bilinear, or nearest.")


def build_image_transform(
    image_size: int = 224,
    interpolation: str = "bicubic",
    train: bool = False,
):
    """Build deterministic resize/normalize transforms.

    The first FR-PEFT stage intentionally avoids augmentation.  Adding random
    crops, RandAugment, MixUp, CutMix, or interpolation-consistency augmentation
    would confound whether LP + delta interpolation itself improves robustness.
    """
    _ = train  # reserved for future controlled augmentation; currently unused by design.
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size)), interpolation=get_interpolation_mode(interpolation)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
