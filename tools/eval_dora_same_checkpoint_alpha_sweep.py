from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from adi_lora.data import build_cifar10_loaders, build_cifar100_loaders
from adi_lora.engine.evaluate import evaluate
from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config, _as_bool
from adi_lora.models.peft import set_delta_scale
from adi_lora.utils.seed import set_seed


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def subset_loader(loader, max_batches: int | None):
    if max_batches is None or max_batches <= 0:
        return loader

    def gen():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            yield batch
    return gen()


def build_loaders(cfg: dict, seed: int):
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    dataset_name = str(data_cfg.get("name", "CIFAR10")).lower()

    if dataset_name in {"cifar100", "cifar-100"}:
        builder = build_cifar100_loaders
    elif dataset_name in {"cifar10", "cifar-10"}:
        builder = build_cifar10_loaders
    else:
        raise ValueError(f"Unsupported dataset.name={data_cfg.get('name')}")

    return builder(
        root=data_cfg.get("root", "./outputs/data"),
        image_size=int(data_cfg.get("image_size", 224)),
        train_interpolation=data_cfg.get("train_interpolation", "bicubic"),
        val_interpolations=data_cfg.get("val_interpolations", ["bicubic", "bilinear"]),
        test_interpolations=data_cfg.get("test_interpolations", ["bicubic", "bilinear", "nearest"]),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        download=_as_bool(data_cfg.get("download", False)),
        pin_memory=torch.cuda.is_available(),
    )


def select_alpha(alpha_rows: list[dict], cfg: dict) -> tuple[float, str]:
    rule = cfg.get("delta_interpolation", {}).get(
        "selection_rule",
        "val_bilinear_acc_with_bicubic_guard",
    )
    guard_pp = float(cfg.get("delta_interpolation", {}).get("bicubic_guard_pp", 0.5))
    if not alpha_rows:
        return 1.0, rule

    max_bicubic = max(float(r["val_bicubic_acc"]) for r in alpha_rows)

    for r in alpha_rows:
        bicubic = float(r["val_bicubic_acc"])
        bilinear = float(r["val_bilinear_acc"])
        r["val_bilinear_drop"] = bicubic - bilinear
        r["bicubic_guard_pass"] = int(bicubic >= max_bicubic - guard_pp / 100.0)
        r["selection_score"] = bilinear
        r["nearest_used_for_selection"] = 0

    candidates = [r for r in alpha_rows if int(r["bicubic_guard_pass"]) == 1]
    if not candidates:
        candidates = alpha_rows

    # 规则与 run_fr_peft.py 保持一致：
    # 1) val bilinear acc 最高；
    # 2) bilinear drop 更小；
    # 3) 若仍并列，选更大 alpha，遵循最小干预原则。
    candidates = sorted(
        candidates,
        key=lambda r: (
            -float(r["val_bilinear_acc"]),
            float(r["val_bilinear_drop"]),
            -float(r["alpha"]),
        ),
    )

    selected_alpha = float(candidates[0]["alpha"])
    for r in alpha_rows:
        r["selected"] = int(abs(float(r["alpha"]) - selected_alpha) < 1e-12)
        r["selection_rule"] = rule
        r["alpha_selection_source"] = "val_bicubic_and_val_bilinear"
        r["nearest_used_for_alpha_selection"] = 0
        r["corruption_used_for_alpha_selection"] = 0
        r["checkpoint_selection_rule"] = "final_checkpoint_only"
        r["nearest_used_for_checkpoint_selection"] = 0
    return selected_alpha, rule


@torch.no_grad()
def eval_one(model, loader, device, cfg, split, interpolation, alpha, reference_model=None, max_eval_batches=None):
    set_delta_scale(model, alpha)
    return evaluate(
        model=model,
        loader=subset_loader(loader, max_eval_batches),
        device=device,
        cfg=cfg,
        split=split,
        interpolation=interpolation,
        alpha=alpha,
        reference_model=reference_model,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="DoRA run dir containing resolved_config.json and checkpoints/")
    ap.add_argument("--checkpoint", default=None, help="optional checkpoint path; default: latest final_*seed*.pth under run-dir/checkpoints")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alphas", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-eval-batches", type=int, default=None)
    ap.add_argument("--eval-all-test-alphas", action="store_true", help="also evaluate test Bicubic/Bilinear/Nearest for every alpha")
    args = ap.parse_args()

    project = Path.cwd()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project / run_dir

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "resolved_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing resolved_config.json: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if args.seed is not None:
        seed = int(args.seed)
    else:
        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            rows = read_csv(summary_path)
            seed = int(float(rows[-1].get("seed", cfg.get("seed", 42))))
        else:
            seed = int(cfg.get("seed", 42))

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    if args.checkpoint is None:
        ckpts = sorted((run_dir / "checkpoints").glob("final_*seed*.pth"))
        if not ckpts:
            ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))
        if not ckpts:
            raise SystemExit(f"No checkpoint found under {run_dir / 'checkpoints'}")
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = project / ckpt_path

    print("[RUN_DIR]", run_dir)
    print("[CFG]", cfg_path)
    print("[CKPT]", ckpt_path)
    print("[SEED]", seed)
    print("[ALPHAS]", alphas)

    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(cfg, seed)

    model = build_model_from_config(cfg).to(device)
    inject_report = inject_peft_from_config(model, cfg)
    print("[INJECT]", inject_report)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[LOAD] missing", len(missing), "unexpected", len(unexpected))
    if missing:
        print("[LOAD missing examples]", missing[:10])
    if unexpected:
        print("[LOAD unexpected examples]", unexpected[:10])

    reference_model = build_model_from_config(cfg).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    # 1) validation alpha sweep: only Bicubic/Bilinear, no Nearest.
    alpha_rows = []
    for alpha in alphas:
        val_bicubic = eval_one(
            model, loaders.val["bicubic"], device, cfg,
            split="val", interpolation="bicubic", alpha=alpha,
            reference_model=None, max_eval_batches=args.max_eval_batches,
        )
        val_bilinear = eval_one(
            model, loaders.val["bilinear"], device, cfg,
            split="val", interpolation="bilinear", alpha=alpha,
            reference_model=None, max_eval_batches=args.max_eval_batches,
        )
        alpha_rows.append({
            "source_run_dir": str(run_dir.relative_to(project)) if str(run_dir).startswith(str(project)) else str(run_dir),
            "checkpoint": str(ckpt_path),
            "seed": seed,
            "alpha": alpha,
            "val_bicubic_acc": float(val_bicubic["acc"]),
            "val_bilinear_acc": float(val_bilinear["acc"]),
            "val_bicubic_loss": float(val_bicubic.get("loss_ce", 0.0)),
            "val_bilinear_loss": float(val_bilinear.get("loss_ce", 0.0)),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
        })

    selected_alpha, selection_rule = select_alpha(alpha_rows, cfg)
    write_csv(out_dir / "alpha_selection_same_checkpoint.csv", alpha_rows)

    # 2) test evaluation.
    # 默认只测 alpha=1.0 和 selected alpha，避免浪费时间；需要全 alpha test 可加 --eval-all-test-alphas。
    test_alphas = alphas if args.eval_all_test_alphas else sorted(set([1.0, selected_alpha]))
    test_rows = []
    for alpha in test_alphas:
        for interp, loader in loaders.test.items():
            row = eval_one(
                model, loader, device, cfg,
                split="test", interpolation=interp, alpha=alpha,
                reference_model=reference_model,
                max_eval_batches=args.max_eval_batches,
            )
            row.update({
                "source_run_dir": str(run_dir.relative_to(project)) if str(run_dir).startswith(str(project)) else str(run_dir),
                "checkpoint": str(ckpt_path),
                "seed": seed,
                "method": "dora_same_checkpoint_alpha_eval",
                "peft_type": "dora",
                "alpha": alpha,
                "selected": int(abs(alpha - selected_alpha) < 1e-12),
                "nearest_used_for_alpha_selection": 0,
                "corruption_used_for_alpha_selection": 0,
                "checkpoint_selection_rule": "final_checkpoint_only",
                "nearest_used_for_checkpoint_selection": 0,
            })
            test_rows.append(row)
    write_csv(out_dir / "test_eval_same_checkpoint.csv", test_rows)

    val_by_alpha = {float(r["alpha"]): r for r in alpha_rows}
    response_rows = []
    for alpha in test_alphas:
        nearest_row = next(
            (
                r for r in test_rows
                if abs(float(r["alpha"]) - float(alpha)) < 1e-12 and r.get("interpolation") == "nearest"
            ),
            {},
        )
        val_row = val_by_alpha.get(float(alpha), {})
        response_rows.append({
            "dataset": "CIFAR-100",
            "method": "ADI-DoRA",
            "peft_type": "dora",
            "seed": seed,
            "alpha": alpha,
            "checkpoint_type": "final",
            "checkpoint_id": str(ckpt_path),
            "val_bicubic_acc_pct": float(val_row.get("val_bicubic_acc", "nan")) * 100,
            "val_bilinear_acc_pct": float(val_row.get("val_bilinear_acc", "nan")) * 100,
            "test_nearest_acc_pct": float(nearest_row.get("acc", "nan")) * 100,
            "loss_ce": float(nearest_row.get("loss_ce", "nan")),
            "selected": int(abs(float(alpha) - selected_alpha) < 1e-12),
            "alpha_selection_source": "val_bicubic_and_val_bilinear",
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
            "nearest_used_for_checkpoint_selection": 0,
            "diagnostic_nearest_full_alpha_sweep": int(args.eval_all_test_alphas),
            "source_run_dir": str(run_dir.relative_to(project)) if str(run_dir).startswith(str(project)) else str(run_dir),
        })
    write_csv(out_dir / "dora_full_alpha_response.csv", response_rows)

    def get_acc(alpha: float, interp: str) -> float:
        for r in test_rows:
            if abs(float(r["alpha"]) - float(alpha)) < 1e-12 and r["interpolation"] == interp:
                return float(r["acc"])
        return float("nan")

    a1_iid = get_acc(1.0, "bicubic")
    a1_bilinear = get_acc(1.0, "bilinear")
    a1_nearest = get_acc(1.0, "nearest")

    sel_iid = get_acc(selected_alpha, "bicubic")
    sel_bilinear = get_acc(selected_alpha, "bilinear")
    sel_nearest = get_acc(selected_alpha, "nearest")

    comparison = [{
        "source_run_dir": str(run_dir.relative_to(project)) if str(run_dir).startswith(str(project)) else str(run_dir),
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "selected_alpha": selected_alpha,
        "selection_rule": selection_rule,
        "alpha1_iid": a1_iid,
        "alpha1_bilinear": a1_bilinear,
        "alpha1_nearest": a1_nearest,
        "selected_iid": sel_iid,
        "selected_bilinear": sel_bilinear,
        "selected_nearest": sel_nearest,
        "iid_gain": sel_iid - a1_iid,
        "bilinear_gain": sel_bilinear - a1_bilinear,
        "nearest_gain": sel_nearest - a1_nearest,
        "alpha1_nearest_drop": a1_iid - a1_nearest,
        "selected_nearest_drop": sel_iid - sel_nearest,
        "drop_reduction": (a1_iid - a1_nearest) - (sel_iid - sel_nearest),
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
        "max_eval_batches": args.max_eval_batches if args.max_eval_batches is not None else "",
    }]
    write_csv(out_dir / "same_checkpoint_comparison.csv", comparison)

    audit = {
        "dataset": cfg.get("dataset", {}).get("name", "CIFAR-100"),
        "peft_type": "dora",
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "checkpoint_selection_rule": "final_checkpoint_only",
        "alpha_candidates": alphas,
        "selected_alpha": selected_alpha,
        "selection_rule": selection_rule,
        "alpha_selection_source": "val_bicubic_and_val_bilinear",
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "nearest_used_for_checkpoint_selection": 0,
        "full_alpha_nearest_sweep_is_diagnostic_only": int(args.eval_all_test_alphas),
    }
    (out_dir / "protocol_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# CIFAR same-checkpoint DoRA alpha evaluation")
    md.append("")
    md.append(f"- source run: `{comparison[0]['source_run_dir']}`")
    md.append(f"- checkpoint: `{comparison[0]['checkpoint']}`")
    md.append(f"- seed: {seed}")
    md.append(f"- selected alpha: {selected_alpha}")
    md.append(f"- selection rule: `{selection_rule}`")
    md.append(f"- Nearest used for alpha selection: 0")
    md.append(f"- checkpoint selection: final checkpoint only")
    md.append("")
    md.append("| Comparison | IID | Bilinear | Nearest | Drop |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| alpha=1.0 | {100*a1_iid:.2f} | {100*a1_bilinear:.2f} | {100*a1_nearest:.2f} | {100*(a1_iid-a1_nearest):.2f} |")
    md.append(f"| selected alpha={selected_alpha:.2f} | {100*sel_iid:.2f} | {100*sel_bilinear:.2f} | {100*sel_nearest:.2f} | {100*(sel_iid-sel_nearest):.2f} |")
    md.append("")
    md.append(f"- Nearest gain: {100*(sel_nearest-a1_nearest):+.2f} pp")
    md.append(f"- Drop reduction: {100*((a1_iid-a1_nearest)-(sel_iid-sel_nearest)):+.2f} pp")
    md.append(f"- IID gain: {100*(sel_iid-a1_iid):+.2f} pp")
    (out_dir / "README_same_checkpoint_dora_alpha_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))


if __name__ == "__main__":
    main()
