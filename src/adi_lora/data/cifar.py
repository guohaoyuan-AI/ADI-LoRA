"""CIFAR dataloaders for FR-PEFT interpolation-shift validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from .transforms import build_image_transform


@dataclass(frozen=True)
class CIFARLoaders:
    train: DataLoader
    val: Dict[str, DataLoader]
    test: Dict[str, DataLoader]
    train_indices: List[int]
    val_indices: List[int]


def _stratified_split(targets: List[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Return train/val indices with class-balanced validation samples.

    This avoids depending on sklearn at runtime and keeps the split deterministic.
    """
    rng = random.Random(int(seed))
    per_class: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        per_class.setdefault(int(target), []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for _, indices in sorted(per_class.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * float(val_ratio))))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _worker_init_fn(worker_id: int):
    # Keep dataloader workers deterministic across seeds.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def build_cifar10_loaders(
    root: str,
    image_size: int = 224,
    train_interpolation: str = "bicubic",
    val_interpolations: Optional[Iterable[str]] = None,
    test_interpolations: Optional[Iterable[str]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
    pin_memory: bool = True,
) -> CIFARLoaders:
    if val_interpolations is None:
        val_interpolations = ["bicubic", "bilinear"]
    if test_interpolations is None:
        test_interpolations = ["bicubic", "bilinear", "nearest"]
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")

    base_train = CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=build_image_transform(image_size, train_interpolation, train=True),
    )
    train_indices, val_indices = _stratified_split(list(base_train.targets), val_ratio=val_ratio, seed=seed)

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_loader = DataLoader(
        Subset(base_train, train_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    val_loaders: Dict[str, DataLoader] = {}
    for interp in val_interpolations:
        dataset = CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=build_image_transform(image_size, interp, train=False),
        )
        val_loaders[str(interp).lower()] = DataLoader(
            Subset(dataset, val_indices),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )

    test_loaders: Dict[str, DataLoader] = {}
    for interp in test_interpolations:
        dataset = CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=build_image_transform(image_size, interp, train=False),
        )
        test_loaders[str(interp).lower()] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )

    return CIFARLoaders(
        train=train_loader,
        val=val_loaders,
        test=test_loaders,
        train_indices=train_indices,
        val_indices=val_indices,
    )



def build_cifar100_loaders(
    root: str,
    image_size: int = 224,
    train_interpolation: str = "bicubic",
    val_interpolations: Optional[Iterable[str]] = None,
    test_interpolations: Optional[Iterable[str]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = False,
    pin_memory: bool = True,
) -> CIFARLoaders:
    """CIFAR-100 loaders with the same interpolation-shift protocol as CIFAR-10."""
    if val_interpolations is None:
        val_interpolations = ["bicubic", "bilinear"]
    if test_interpolations is None:
        test_interpolations = ["bicubic", "bilinear", "nearest"]
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")

    base_train = CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=build_image_transform(image_size, train_interpolation, train=True),
    )
    train_indices, val_indices = _stratified_split(list(base_train.targets), val_ratio=val_ratio, seed=seed)

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_loader = DataLoader(
        Subset(base_train, train_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    val_loaders: Dict[str, DataLoader] = {}
    for interp in val_interpolations:
        dataset = CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=build_image_transform(image_size, interp, train=False),
        )
        val_loaders[str(interp).lower()] = DataLoader(
            Subset(dataset, val_indices),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )

    test_loaders: Dict[str, DataLoader] = {}
    for interp in test_interpolations:
        dataset = CIFAR100(
            root=root,
            train=False,
            download=download,
            transform=build_image_transform(image_size, interp, train=False),
        )
        test_loaders[str(interp).lower()] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )

    return CIFARLoaders(
        train=train_loader,
        val=val_loaders,
        test=test_loaders,
        train_indices=train_indices,
        val_indices=val_indices,
    )
