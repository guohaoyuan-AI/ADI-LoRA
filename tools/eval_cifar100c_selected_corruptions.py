from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config
from adi_lora.models.peft import set_delta_scale
from adi_lora.utils.seed import set_seed


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def find_latest_delta_run(outputs_root: Path, seed: int):
    candidates = []

    for summary_path in outputs_root.rglob("summary.csv"):
        run_dir = summary_path.parent

        try:
            rows = read_csv(summary_path)
        except Exception:
            continue

        if not rows:
            continue

        row = rows[-1]

        if row.get("method") != "delta_lora":
            continue

        try:
            row_seed = int(float(row.get("seed", -1)))
        except Exception:
            continue

        if row_seed != int(seed):
            continue

        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        ckpts = sorted(list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt")))
        if not ckpts:
            continue

        final_ckpts = [p for p in ckpts if "final" in p.name.lower()]
        ckpt_path = final_ckpts[-1] if final_ckpts else ckpts[-1]

        resolved_config = run_dir / "resolved_config.json"
        if not resolved_config.exists():
            continue

        candidates.append((run_dir.stat().st_mtime, run_dir, summary_path, ckpt_path, resolved_config))

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x[0])[-1]


class CIFAR100CDataset(Dataset):
    def __init__(self, data_path: Path, labels_path: Path, severity: int, image_size: int = 224):
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.severity = int(severity)
        self.image_size = int(image_size)

        if self.severity < 1 or self.severity > 5:
            raise ValueError("severity must be in [1, 5]")

        data = np.load(self.data_path)
        labels = np.load(self.labels_path)

        if data.shape[0] >= 50000:
            start = (self.severity - 1) * 10000
            end = self.severity * 10000
            data = data[start:end]

            if labels.shape[0] >= 50000:
                labels = labels[start:end]
            elif labels.shape[0] == 10000:
                labels = labels
            else:
                raise ValueError(f"Unexpected labels shape: {labels.shape}")
        else:
            if labels.shape[0] != data.shape[0]:
                raise ValueError(f"Data/labels mismatch: data={data.shape}, labels={labels.shape}")

        self.data = data
        self.labels = labels.astype("int64")

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        img = self.data[idx]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        pil = Image.fromarray(img).convert("RGB")
        x = self.transform(pil)
        y = int(self.labels[idx])
        return x, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += int(targets.numel())

    elapsed = max(time.time() - start, 1e-6)
    peak_mem = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

    return {
        "acc": total_correct / max(total_seen, 1),
        "loss_ce": total_loss / max(total_seen, 1),
        "num_samples": total_seen,
        "eval_time_sec": elapsed,
        "throughput_img_s": total_seen / elapsed,
        "peak_mem_mb": peak_mem,
    }


def parse_corruptions(cifar100c_root: Path, corruptions_arg: str, exclude_arg: str):
    available = sorted([p.stem for p in cifar100c_root.glob("*.npy") if p.stem != "labels"])

    exclude = set()
    if exclude_arg.strip():
        exclude = {x.strip() for x in exclude_arg.split(",") if x.strip()}

    if corruptions_arg.strip().lower() == "all":
        selected = [c for c in available if c not in exclude]
    else:
        selected = [x.strip() for x in corruptions_arg.split(",") if x.strip()]

    missing = [c for c in selected if not (cifar100c_root / f"{c}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing corruption files: {missing}")

    return available, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar100c-root", required=True)
    parser.add_argument("--outputs-root", default="outputs/fr_peft_cifar100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--corruptions", default="all")
    parser.add_argument("--exclude-corruptions", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project = Path.cwd()
    cifar100c_root = Path(args.cifar100c_root)
    outputs_root = project / args.outputs_root
    out_dir = Path(args.out)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = cifar100c_root / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")

    available, selected_corruptions = parse_corruptions(
        cifar100c_root,
        corruptions_arg=args.corruptions,
        exclude_arg=args.exclude_corruptions,
    )

    found = find_latest_delta_run(outputs_root, args.seed)
    if found is None:
        raise SystemExit(
            f"No delta_lora seed{args.seed} checkpoint found under {outputs_root}."
        )

    _, run_dir, summary_path, ckpt_path, resolved_config_path = found

    summary = read_csv(summary_path)[-1]
    cfg = json.loads(resolved_config_path.read_text(encoding="utf-8"))

    selected_alpha = float(summary.get("selected_alpha", 1.0))
    seed = int(float(summary.get("seed", args.seed)))

    print("[AVAILABLE_CORRUPTIONS]", available)
    print("[SELECTED_CORRUPTIONS]", selected_corruptions)
    print("[SOURCE RUN]", run_dir)
    print("[CHECKPOINT]", ckpt_path)
    print("[SELECTED_ALPHA]", selected_alpha)
    print("[SEVERITY]", args.severity)

    set_seed(seed, deterministic=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg).to(device)
    inject_peft_from_config(model, cfg)
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("[LOAD] missing:", len(missing), "unexpected:", len(unexpected))

    all_rows = []
    cmp_rows = []

    for corr in selected_corruptions:
        data_path = cifar100c_root / f"{corr}.npy"
        print(f"[EVAL] {corr}, severity={args.severity}")

        dataset = CIFAR100CDataset(
            data_path=data_path,
            labels_path=labels_path,
            severity=args.severity,
            image_size=224,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        set_delta_scale(model, 1.0)
        r_lora = evaluate(model, loader, device)
        r_lora.update({
            "dataset": "CIFAR100-C",
            "seed": seed,
            "method": "lora_alpha1_same_checkpoint",
            "alpha": 1.0,
            "corruption": corr,
            "severity": args.severity,
            "data_file": str(data_path),
            "source_run_dir": str(run_dir.relative_to(project)),
        })
        all_rows.append(r_lora)

        set_delta_scale(model, selected_alpha)
        r_delta = evaluate(model, loader, device)
        r_delta.update({
            "dataset": "CIFAR100-C",
            "seed": seed,
            "method": "delta_lora_selected",
            "alpha": selected_alpha,
            "corruption": corr,
            "severity": args.severity,
            "data_file": str(data_path),
            "source_run_dir": str(run_dir.relative_to(project)),
        })
        all_rows.append(r_delta)

        cmp_rows.append({
            "dataset": "CIFAR100-C",
            "seed": seed,
            "corruption": corr,
            "severity": args.severity,
            "selected_alpha": selected_alpha,
            "lora_alpha1_acc": r_lora["acc"],
            "delta_selected_acc": r_delta["acc"],
            "acc_gain": r_delta["acc"] - r_lora["acc"],
            "lora_alpha1_loss": r_lora["loss_ce"],
            "delta_selected_loss": r_delta["loss_ce"],
            "loss_change": r_delta["loss_ce"] - r_lora["loss_ce"],
            "num_samples": r_lora["num_samples"],
        })

    write_csv(out_dir / "cifar100c_eval.csv", all_rows)
    write_csv(out_dir / "cifar100c_comparison.csv", cmp_rows)

    mean_gain = sum(float(r["acc_gain"]) for r in cmp_rows) / max(len(cmp_rows), 1)

    md = []
    md.append("# CIFAR-100-C selected corruption evaluation")
    md.append("")
    md.append(f"- seed: {seed}")
    md.append(f"- severity: {args.severity}")
    md.append(f"- selected_alpha: {selected_alpha}")
    md.append(f"- num_selected_corruptions: {len(selected_corruptions)}")
    md.append(f"- selected_corruptions: {', '.join(selected_corruptions)}")
    md.append(f"- source_run_dir: {run_dir.relative_to(project)}")
    md.append("")
    md.append("| Corruption | LoRA α=1 Acc | Δ-LoRA Acc | Gain |")
    md.append("|---|---:|---:|---:|")

    for r in cmp_rows:
        md.append(
            f"| {r['corruption']} | "
            f"{100 * float(r['lora_alpha1_acc']):.2f} | "
            f"{100 * float(r['delta_selected_acc']):.2f} | "
            f"{100 * float(r['acc_gain']):+.2f} |"
        )

    md.append("")
    md.append(f"- mean_acc_gain_pp: {100 * mean_gain:.2f}")
    md.append("")
    md.append("Protocol:")
    md.append("- Same checkpoint comparison.")
    md.append("- α=1.0 is the LoRA baseline.")
    md.append("- selected α is Δ-LoRA / ADI-LoRA.")
    md.append("- No training is performed.")

    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tar_path = Path(str(out_dir) + ".tar.gz")
    if tar_path.exists():
        tar_path.unlink()

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)

    print((out_dir / "README.md").read_text(encoding="utf-8"))
    print("[TAR]", tar_path)


if __name__ == "__main__":
    main()
