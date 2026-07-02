# ADI on VPT/Adapter: Boundary Diagnostic Tables and Paper-Ready Analysis

This document summarizes the CIFAR-100 seed42 experiments where ADI was extended from LoRA/DoRA-style weight-delta PEFT to VPT and residual Adapter variants. These results should be used as **boundary diagnostics**, not as main positive results.

Core conclusion:

> Under the formal validation-selected ADI protocol, VPT-shallow, VPT-deep, and Adapter-all-b32 all select \(\alpha=1.0\), so ADI produces no main-protocol gain. However, diagnostic full-alpha test sweeps reveal that VPT-deep and Adapter contain shift-sensitive components that can be suppressed by smaller \(\alpha\), but the current Bicubic/Bilinear validation selection rule cannot choose them without target-shift information.

---

## 1. Experimental Scope

| Item | Setting |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | ViT-B/16 pretrained backbone, same project protocol |
| Seed | 42 only |
| Methods | VPT-shallow, VPT-deep, Adapter-all-b32 |
| Alpha grid | 0.2, 0.4, 0.6, 0.8, 1.0 |
| Formal selection signal | Validation Bicubic / Bilinear only |
| Forbidden for selection | Nearest test, CIFAR-C test, corruption labels |
| Checkpoint rule | Final checkpoint only |
| Use in paper | Boundary / supplementary diagnostic, not main method evidence |

Important limitation:

> These experiments are single-seed boundary diagnostics. They should not be reported as 3-seed main results and should not be used to claim that ADI works as a universal PEFT framework.

---

## 2. Formal Validation-Selected Protocol

This is the strict protocol result. Since all three methods select \(\alpha=1.0\), ADI degenerates to the original model and gives no formal gain.

### Table 1. Formal ADI results on VPT/Adapter under validation-selected alpha

| Method | Seed | Selected alpha | Bicubic base | Bicubic ADI | Delta Bicubic | Bilinear base | Bilinear ADI | Delta Bilinear | Nearest base | Nearest ADI | Delta Nearest | Drop base | Drop ADI | Drop reduction | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VPT-shallow | 42 | 1.0 | 89.87 | 89.87 | 0.00 | 88.98 | 88.98 | 0.00 | 49.43 | 49.43 | 0.00 | 40.44 | 40.44 | 0.00 | 89,188 |
| VPT-deep | 42 | 1.0 | 91.67 | 91.67 | 0.00 | 90.94 | 90.94 | 0.00 | 57.58 | 57.58 | 0.00 | 34.09 | 34.09 | 0.00 | 224,356 |
| Adapter-all-b32 | 42 | 1.0 | 92.04 | 92.04 | 0.00 | 91.63 | 91.63 | 0.00 | 48.61 | 48.61 | 0.00 | 43.43 | 43.43 | 0.00 | 694,756 |

Paper interpretation:

> Unlike LoRA/DoRA-style weight-delta PEFT, the current validation-selected ADI protocol does not produce formal gains for VPT or Adapter variants. In all three cases, the validation rule selects \(\alpha=1.0\), reducing ADI to the uncalibrated PEFT model.

Recommended paper placement:

- Main paper: short boundary paragraph or compact table.
- Supplementary: full alpha selection and diagnostic sweep.

---

## 3. Alpha Selection Behavior

The following table shows why the formal protocol selects \(\alpha=1.0\). The selection rule prioritizes validation Bilinear accuracy among candidates passing the Bicubic guard.

### Table 2. Validation alpha selection for VPT/Adapter

| Method | Alpha | Val Bicubic | Val Bilinear | Bicubic guard pass | Selection score | Selected |
|---|---:|---:|---:|---:|---:|---:|
| VPT-shallow | 0.2 | 79.34 | 77.82 | 0 | 77.82 | 0 |
| VPT-shallow | 0.4 | 88.62 | 87.18 | 0 | 87.18 | 0 |
| VPT-shallow | 0.6 | 90.06 | 88.88 | 1 | 88.88 | 0 |
| VPT-shallow | 0.8 | 90.14 | 89.10 | 1 | 89.10 | 0 |
| VPT-shallow | 1.0 | 90.16 | 89.32 | 1 | 89.32 | 1 |
| VPT-deep | 0.2 | 81.04 | 78.78 | 0 | 78.78 | 0 |
| VPT-deep | 0.4 | 90.38 | 89.58 | 0 | 89.58 | 0 |
| VPT-deep | 0.6 | 91.72 | 90.92 | 1 | 90.92 | 0 |
| VPT-deep | 0.8 | 91.92 | 90.96 | 1 | 90.96 | 0 |
| VPT-deep | 1.0 | 91.86 | 91.06 | 1 | 91.06 | 1 |
| Adapter-all-b32 | 0.2 | 82.66 | 82.66 | 0 | 82.66 | 0 |
| Adapter-all-b32 | 0.4 | 88.92 | 89.04 | 0 | 89.04 | 0 |
| Adapter-all-b32 | 0.6 | 91.16 | 90.66 | 0 | 90.66 | 0 |
| Adapter-all-b32 | 0.8 | 91.96 | 91.30 | 1 | 91.30 | 0 |
| Adapter-all-b32 | 1.0 | 92.18 | 91.62 | 1 | 91.62 | 1 |

Analysis:

- The selection rule is conservative and target-shift-free.
- It consistently favors \(\alpha=1.0\) because \(\alpha=1.0\) has the best validation Bilinear score.
- Therefore, the formal protocol cannot exploit any potential Nearest gain from smaller alpha values.
- This is not a leakage issue; it is a selection-signal limitation.

Paper-ready wording:

> The validation signal used in the main ADI protocol is insufficient to select smaller \(\alpha\) values for prompt- or residual-adapter-based PEFT. This explains why these variants do not enter the main result tables, despite showing useful diagnostic behavior under full-alpha sweeps.

---

## 4. Diagnostic Full-Alpha Test Sweep

The following table is **diagnostic-only**. It evaluates all alpha values on the test set to understand whether smaller deltas could improve Nearest robustness. These values must not be used to select alpha for the main method.

### Table 3. Diagnostic full-alpha sweep on CIFAR-100 test set

| Method | Alpha | Bicubic | Bilinear | Nearest | Nearest drop |
|---|---:|---:|---:|---:|---:|
| VPT-shallow | 0.2 | 78.94 | 77.20 | 26.35 | 52.59 |
| VPT-shallow | 0.4 | 88.14 | 86.55 | 46.62 | 41.52 |
| VPT-shallow | 0.6 | 89.57 | 88.47 | 49.81 | 39.76 |
| VPT-shallow | 0.8 | 89.87 | 88.83 | 49.75 | 40.12 |
| VPT-shallow | 1.0 | 89.87 | 88.98 | 49.43 | 40.44 |
| VPT-deep | 0.2 | 80.40 | 78.62 | 45.31 | 35.09 |
| VPT-deep | 0.4 | 90.42 | 89.37 | 59.55 | 30.87 |
| VPT-deep | 0.6 | 91.62 | 90.57 | 59.98 | 31.64 |
| VPT-deep | 0.8 | 91.75 | 90.93 | 58.71 | 33.04 |
| VPT-deep | 1.0 | 91.67 | 90.94 | 57.58 | 34.09 |
| Adapter-all-b32 | 0.2 | 82.03 | 82.29 | 44.45 | 37.58 |
| Adapter-all-b32 | 0.4 | 88.51 | 88.55 | 51.88 | 36.63 |
| Adapter-all-b32 | 0.6 | 91.21 | 90.84 | 54.84 | 36.37 |
| Adapter-all-b32 | 0.8 | 91.98 | 91.56 | 53.93 | 38.05 |
| Adapter-all-b32 | 1.0 | 92.04 | 91.63 | 48.61 | 43.43 |

Diagnostic interpretation:

- VPT-shallow has only a very small hidden gain. Its best Nearest improvement over \(\alpha=1.0\) is +0.38 pp at \(\alpha=0.6\), with a small Bilinear loss. This is too weak to support further expansion.
- VPT-deep shows meaningful hidden gain. \(\alpha=0.6\) improves Nearest by +2.40 pp while losing only -0.05 pp Bicubic and -0.37 pp Bilinear.
- Adapter-all-b32 shows the strongest hidden gain. \(\alpha=0.8\) improves Nearest by +5.32 pp with only -0.06 pp Bicubic and -0.07 pp Bilinear loss. \(\alpha=0.6\) improves Nearest by +6.23 pp but with larger clean/interpolation loss.

---

## 5. Hidden Gain Relative to Alpha = 1.0

This table directly compares each alpha to the uncalibrated model, \(\alpha=1.0\).

### Table 4. Diagnostic hidden gain relative to \(\alpha=1.0\)

| Method | Alpha | Delta Bicubic vs 1.0 | Delta Bilinear vs 1.0 | Delta Nearest vs 1.0 | Drop reduction vs 1.0 |
|---|---:|---:|---:|---:|---:|
| VPT-shallow | 0.2 | -10.93 | -11.78 | -23.08 | -12.15 |
| VPT-shallow | 0.4 | -1.73 | -2.43 | -2.81 | -1.08 |
| VPT-shallow | 0.6 | -0.30 | -0.51 | +0.38 | +0.68 |
| VPT-shallow | 0.8 | +0.00 | -0.15 | +0.32 | +0.32 |
| VPT-shallow | 1.0 | +0.00 | +0.00 | +0.00 | +0.00 |
| VPT-deep | 0.2 | -11.27 | -12.32 | -12.27 | -1.00 |
| VPT-deep | 0.4 | -1.25 | -1.57 | +1.97 | +3.22 |
| VPT-deep | 0.6 | -0.05 | -0.37 | +2.40 | +2.45 |
| VPT-deep | 0.8 | +0.08 | -0.01 | +1.13 | +1.05 |
| VPT-deep | 1.0 | +0.00 | +0.00 | +0.00 | +0.00 |
| Adapter-all-b32 | 0.2 | -10.01 | -9.34 | -4.16 | +5.85 |
| Adapter-all-b32 | 0.4 | -3.53 | -3.08 | +3.27 | +6.80 |
| Adapter-all-b32 | 0.6 | -0.83 | -0.79 | +6.23 | +7.06 |
| Adapter-all-b32 | 0.8 | -0.06 | -0.07 | +5.32 | +5.38 |
| Adapter-all-b32 | 1.0 | +0.00 | +0.00 | +0.00 | +0.00 |

Key reading:

- VPT-shallow: hidden gain is negligible.
- VPT-deep: hidden gain exists, but alpha selection does not find it.
- Adapter-all-b32: hidden gain is strong, but alpha selection still does not find it.

Recommended summary table:

### Table 5. Boundary diagnostic summary

| Method | Formal selected alpha | Formal ADI gain | Best diagnostic alpha | Best diagnostic Nearest gain | Near-clean diagnostic alpha | Near-clean Nearest gain | Paper conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| VPT-shallow | 1.0 | 0.00 | 0.6 | +0.38 | 0.8 | +0.32 | Weak / true boundary |
| VPT-deep | 1.0 | 0.00 | 0.6 | +2.40 | 0.6 | +2.40 | Hidden gain, selection bottleneck |
| Adapter-all-b32 | 1.0 | 0.00 | 0.6 | +6.23 | 0.8 | +5.32 | Strong hidden gain, selection bottleneck |

Definition of near-clean diagnostic alpha:

> Among alpha values whose Bicubic and Bilinear accuracy are within approximately 0.5 pp of \(\alpha=1.0\), choose the one with the best Nearest accuracy. This is for analysis only and must not be used as the formal ADI selection rule.

---

## 6. Paper-Ready Analysis

### 6.1 Main boundary result

The formal ADI protocol does not extend successfully to VPT and Adapter variants. Under the same target-shift-free selection rule used for LoRA/DoRA, all three methods select \(\alpha=1.0\). As a result, ADI degenerates to the uncalibrated PEFT model and produces no formal improvement in Bicubic, Bilinear, or Nearest accuracy.

This result is important because it prevents overclaiming. The current paper should not describe ADI as a universal PEFT method. The evidence supports ADI primarily for LoRA/DoRA-style weight-delta PEFT, where the pretrained weight \(W_0\) provides a clean anchor and the learned update can be directly interpolated.

### 6.2 VPT-shallow

VPT-shallow is a weak boundary case. The full-alpha diagnostic sweep shows only a minor Nearest improvement at smaller alpha values:

- Best diagnostic alpha: \(\alpha=0.6\)
- Nearest gain: +0.38 pp
- Bicubic change: -0.30 pp
- Bilinear change: -0.51 pp

This gain is too small to justify extending the main method or running additional seeds. VPT-shallow can be described as a negative or weak boundary result.

Paper wording:

> VPT-shallow shows negligible diagnostic gains under alpha scaling, suggesting that the current ADI formulation is not effective for shallow prompt tuning.

### 6.3 VPT-deep

VPT-deep is more nuanced. The formal protocol still selects \(\alpha=1.0\), so it is not a main positive result. However, the diagnostic sweep shows that smaller alpha values can improve Nearest robustness:

- \(\alpha=0.6\): Nearest +2.40 pp, Bicubic -0.05 pp, Bilinear -0.37 pp.
- \(\alpha=0.8\): Nearest +1.13 pp, Bicubic +0.08 pp, Bilinear -0.01 pp.

This suggests that VPT-deep contains shift-sensitive prompt updates, but the current validation selection rule cannot reliably select the robust alpha. Additionally, prompt tuning has a less clean anchor than LoRA/DoRA. For VPT, \(\alpha=0\) corresponds to the initial prompt or a prompt-related anchor, not necessarily the pretrained no-prompt model.

Paper wording:

> VPT-deep exhibits diagnostic hidden gains under smaller prompt-delta strengths, but these gains are not selected by the target-shift-free validation rule. We therefore treat VPT-deep as evidence of a selection bottleneck rather than as a main ADI result.

### 6.4 Adapter-all-b32

Adapter-all-b32 is the strongest diagnostic case. The formal protocol selects \(\alpha=1.0\), but full-alpha sweep reveals substantial hidden Nearest gains:

- \(\alpha=0.8\): Nearest +5.32 pp, Bicubic -0.06 pp, Bilinear -0.07 pp.
- \(\alpha=0.6\): Nearest +6.23 pp, Bicubic -0.83 pp, Bilinear -0.79 pp.

This indicates that residual adapter responses contain shift-sensitive components that can be suppressed by scaling the adapter residual. However, because the formal validation rule selects \(\alpha=1.0\), this cannot be counted as a main protocol gain.

Paper wording:

> Adapter residual scaling reveals strong diagnostic Nearest gains, indicating that residual adapters may also contain shift-sensitive response components. However, these gains are not accessible under the current validation-selected ADI protocol, so Adapter is reported only as a boundary diagnostic.

### 6.5 Overall interpretation

These results support a refined scope:

> ADI is currently validated as a method for LoRA/DoRA-style weight-delta PEFT, not as a universal calibration rule for all PEFT modules.

At the same time, the VPT-deep and Adapter diagnostics are valuable. They suggest that the broader idea of calibrating learned PEFT updates may extend beyond LoRA/DoRA, but doing so requires:

1. cleaner anchor definitions for prompt and residual modules;
2. target-shift-free selection signals that better predict Nearest robustness;
3. a new method version rather than retrofitting the current ADI-LoRA/DoRA protocol.

---

## 7. Recommended Paper Placement

### Main paper

Use one short paragraph in the scope/boundary section:

> We additionally examined VPT-shallow, VPT-deep, and residual Adapter variants. Under the same target-shift-free validation selection rule, all three variants selected \(\alpha=1.0\), yielding no formal ADI gain. We therefore restrict the main claims to LoRA/DoRA-style weight-delta PEFT.

Then add one compact table:

| Method | Selected alpha | Formal Nearest gain | Diagnostic best Nearest gain | Conclusion |
|---|---:|---:|---:|---|
| VPT-shallow | 1.0 | 0.00 | +0.38 | Weak boundary |
| VPT-deep | 1.0 | 0.00 | +2.40 | Hidden gain, selection bottleneck |
| Adapter-all-b32 | 1.0 | 0.00 | +6.23 | Strong hidden gain, selection bottleneck |

### Supplementary

Put the full alpha selection table and full diagnostic sweep table in the supplementary material.

---

## 8. LaTeX Table Snippets

### 8.1 Compact boundary table

```latex
\begin{table}[t]
\centering
\caption{Boundary diagnostics on CIFAR-100 for VPT and Adapter variants. Formal ADI uses validation-selected $\alpha$ without access to Nearest or corruption test sets. Diagnostic gains are reported from full-alpha test sweeps and are not used for alpha selection.}
\label{tab:boundary_vpt_adapter}
\begin{tabular}{lcccc}
\toprule
Method & Selected $\alpha$ & Formal $\Delta$Nearest & Best diag. $\Delta$Nearest & Conclusion \\
\midrule
VPT-shallow & 1.0 & 0.00 & +0.38 & Weak boundary \\
VPT-deep & 1.0 & 0.00 & +2.40 & Hidden gain; selection bottleneck \\
Adapter-all-b32 & 1.0 & 0.00 & +6.23 & Strong hidden gain; selection bottleneck \\
\bottomrule
\end{tabular}
\end{table}
```

### 8.2 Full diagnostic table

```latex
\begin{table*}[t]
\centering
\caption{Diagnostic full-alpha sweep on CIFAR-100. This table is diagnostic-only: Nearest test results are not used for alpha selection.}
\label{tab:diagnostic_full_alpha_vpt_adapter}
\begin{tabular}{llrrrr}
\toprule
Method & $\alpha$ & Bicubic & Bilinear & Nearest & Nearest drop \\
\midrule
VPT-shallow & 0.2 & 78.94 & 77.20 & 26.35 & 52.59 \\
VPT-shallow & 0.4 & 88.14 & 86.55 & 46.62 & 41.52 \\
VPT-shallow & 0.6 & 89.57 & 88.47 & 49.81 & 39.76 \\
VPT-shallow & 0.8 & 89.87 & 88.83 & 49.75 & 40.12 \\
VPT-shallow & 1.0 & 89.87 & 88.98 & 49.43 & 40.44 \\
VPT-deep & 0.2 & 80.40 & 78.62 & 45.31 & 35.09 \\
VPT-deep & 0.4 & 90.42 & 89.37 & 59.55 & 30.87 \\
VPT-deep & 0.6 & 91.62 & 90.57 & 59.98 & 31.64 \\
VPT-deep & 0.8 & 91.75 & 90.93 & 58.71 & 33.04 \\
VPT-deep & 1.0 & 91.67 & 90.94 & 57.58 & 34.09 \\
Adapter-all-b32 & 0.2 & 82.03 & 82.29 & 44.45 & 37.58 \\
Adapter-all-b32 & 0.4 & 88.51 & 88.55 & 51.88 & 36.63 \\
Adapter-all-b32 & 0.6 & 91.21 & 90.84 & 54.84 & 36.37 \\
Adapter-all-b32 & 0.8 & 91.98 & 91.56 & 53.93 & 38.05 \\
Adapter-all-b32 & 1.0 & 92.04 & 91.63 & 48.61 & 43.43 \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 9. Final Recommendation

Use these experiments to strengthen the paper's honesty and scope control:

1. Do not claim ADI works for all PEFT methods.
2. Main claims should remain LoRA/DoRA-style weight-delta PEFT.
3. Report VPT/Adapter as boundary diagnostics.
4. State that the current validation selection rule fails to select useful smaller alpha values for VPT-deep and Adapter.
5. Use Adapter and VPT-deep hidden gains to motivate future work on anchor-aware or tolerance-based ADI, but do not mix that future direction into the current main method.

Best one-sentence conclusion:

> VPT and Adapter experiments do not support extending the current validation-selected ADI protocol beyond LoRA/DoRA as main results, but their diagnostic full-alpha sweeps reveal that learned prompt/residual updates can also contain shift-sensitive components, motivating future anchor-aware delta calibration.

