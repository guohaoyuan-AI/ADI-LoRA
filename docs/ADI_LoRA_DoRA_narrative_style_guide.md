# ADI-LoRA/ADI-DoRA 论文叙事风格指导书

本文档用于指导 ADI-LoRA/ADI-DoRA 论文写作对话理解并模仿优秀视觉论文的叙事方式，尤其参考 ResNet 论文的写作骨架，但必须保持当前研究证据允许的克制表达。本文档不是实验交接材料，也不是提示词，而是论文写作风格、论证结构和叙事逻辑的指导书。

---

## 1. 总体写作定位

ADI 当前论文应收束为：

> A lightweight post-hoc delta calibration method for LoRA/DoRA-style parameter-efficient fine-tuning.

核心主张是：

> 完整 PEFT delta 并不总是最鲁棒的部署选择。对于 LoRA/DoRA-style weight-delta PEFT，从预训练权重 anchor 出发，通过 target-shift-free validation 选择合适的 delta interpolation strength，可以在保持 clean / interpolation validation 性能的同时提升 interpolation 和 corruption robustness，且不增加推理阶段模块。

必须避免把论文写成：

> A universal robust PEFT framework for all adapters.

当前论文主线只应覆盖 LoRA / DoRA。VPT / Adapter 只能作为 diagnostic boundary，不应作为主方法正结果。

---

## 2. 学 ResNet 的什么，不学什么

ResNet 论文的叙事优势不在于堆砌复杂模块，而在于它把一个反常问题清楚地重表述，并用极简方法和系统实验支撑该重表述。

### 2.1 可以学习的叙事骨架

| ResNet 叙事手法 | ADI-LoRA/DoRA 对应写法 |
|---|---|
| 从反常现象切入 | 完整 PEFT delta 在 IID 上有效，但在 Nearest / corruption 下不一定最鲁棒 |
| 提出关键问题 | 测试时是否一定要注入全部 learned delta? |
| 做问题重表述 | 从“训练更强 PEFT”转为“校准已有 delta 的部署强度” |
| 方法极简 | 只做 delta interpolation，不引入新训练模块 |
| 强调复杂度优势 | 推理阶段无额外模块，合并权重后无额外计算 |
| 系统实证 | LoRA/DoRA、3-seed、多数据集、多 shift、效率与机制指标 |
| 讨论边界 | VPT/Adapter 作为 diagnostic boundary，而不是强行纳入主方法 |

### 2.2 不应学习的表达方式

| 不应模仿 | 原因 |
|---|---|
| foundational / generic / universal | 当前证据仅支持 LoRA/DoRA-style PEFT |
| solve / guarantee / prove | 过度绝对，容易被审稿人抓住 |
| 竞赛冠军式叙事 | 与当前论文贡献类型不匹配 |
| 把所有 PEFT 都纳入主张 | VPT/Adapter 的结果属于边界诊断，不是主协议正结果 |
| 把 single-seed 结果写成定论 | 所有主结论必须基于 3-seed 或明确标注 diagnostic-only |

---

## 3. 论文的核心叙事线

全文应围绕一条清楚的线展开：

> 预训练 ViT 已经包含强视觉先验。LoRA/DoRA 在下游训练中学习到的 delta 能提高适配能力，但完整注入该 delta 可能同时引入 shift-sensitive components。ADI 不重新设计 PEFT 模块，而是在 target-shift-free validation 下校准 delta strength，使模型在保持 clean performance 的同时提升 interpolation / corruption robustness。

这条线要贯穿 Abstract、Introduction、Method、Experiments 和 Discussion。

ResNet 的核心重表述是：

> 直接学习原始映射不如学习 residual mapping。

ADI 的对应重表述应是：

> 直接部署完整 PEFT delta 不一定最鲁棒；从预训练 anchor 出发校准 delta strength 可以得到更稳的适配。

---

## 4. Abstract 写作结构

摘要应采用“问题 -> 观察 -> 方法 -> 约束 -> 证据 -> 代价”的顺序。

建议结构：

1. PEFT methods such as LoRA and DoRA efficiently adapt pretrained vision transformers with small trainable updates.
2. However, the fully learned update is not always the most robust deployed update under interpolation and corruption shifts.
3. We propose Adapter Delta Interpolation (ADI), a post-hoc calibration method that interpolates LoRA/DoRA weight deltas from the pretrained anchor.
4. The interpolation coefficient is selected using validation data without using nearest-neighbor or corruption test sets.
5. Experiments across LoRA/DoRA, multiple seeds, and multiple robustness benchmarks show consistent robustness improvements.
6. ADI introduces no additional inference-time modules after weight merging.

摘要中不要出现：

- universal PEFT framework
- solves robustness
- guarantees OOD robustness
- optimal alpha
- broadly applicable to all adapters

---

## 5. Introduction 叙事顺序

Introduction 应从“被忽视的问题”而不是“我们提出某方法”开始。

### 5.1 第一段：背景

说明 PEFT 在视觉 Transformer 下游适配中的价值：

- ViT 预训练模型具有强视觉先验；
- LoRA/DoRA 可以通过少量参数完成高效适配；
- 现有研究通常关注如何训练 PEFT delta，而较少讨论测试时是否应该完整部署该 delta。

### 5.2 第二段：反常现象

提出核心观察：

> The full PEFT update is not always the most robust deployed update.

解释原因时要克制：

- 下游训练可能使 delta 同时学习 task-relevant components 和 shift-sensitive components；
- 在 interpolation shift 或 corruption shift 下，完整 delta 可能放大非稳健成分；
- 适当缩放 delta 可以改善 robustness，同时几乎不损失 clean / bilinear accuracy。

不要写成因果已被完全证明。应使用：

- may contain
- can amplify
- suggests
- indicates
- empirical evidence

### 5.3 第三段：核心问题

可使用如下问题句作为引言中心：

> If a pretrained vision transformer already provides a strong visual prior, should a PEFT model always inject the entire learned delta at inference time?

或：

> Robust adaptation may require not only learning a useful delta, but also calibrating how much of the delta should be deployed.

这两句是 ADI 的“ResNet式问题重表述”。

### 5.4 第四段：方法

一句话介绍 ADI：

> ADI calibrates LoRA/DoRA-style weight deltas by interpolating between the pretrained anchor and the fine-tuned PEFT update.

紧接着强调：

- post-hoc；
- 不改训练；
- 不使用 target-shift test data 选 alpha；
- 合并后无推理额外模块。

### 5.5 第五段：贡献

贡献建议写为四点：

1. We identify a delta over-injection phenomenon in LoRA/DoRA-style PEFT under interpolation and corruption shifts.
2. We propose ADI, a simple post-hoc delta interpolation method with target-shift-free alpha selection.
3. We provide systematic empirical evidence across LoRA/DoRA, multiple random seeds, interpolation shifts, corruption shifts, and dataset extensions.
4. We analyze efficiency, feature/spectral behavior, and diagnostic boundary cases, clarifying both the benefit and the limitation of ADI.

---

## 6. Method 写作风格

Method 部分要保持极简清晰。不要把 ADI 包装成复杂新架构。越简单，越要把 anchor、delta、alpha selection 和 leakage boundary 写清楚。

### 6.1 LoRA 形式

\[
W_{\mathrm{eff}} = W_0 + \alpha \Delta W_{\mathrm{LoRA}}
\]

其中：

- \(W_0\) 是 frozen pretrained weight；
- \(\Delta W_{\mathrm{LoRA}}\) 是训练后的 LoRA update；
- \(\alpha=1\) 对应原始 LoRA；
- \(\alpha<1\) 表示部分注入 learned delta。

### 6.2 DoRA 形式

\[
W_{\mathrm{eff}} = W_0 + \alpha (W_{\mathrm{DoRA}} - W_0)
\]

其中：

- \(W_{\mathrm{DoRA}}\) 是 DoRA 训练后等效权重；
- \(W_{\mathrm{DoRA}} - W_0\) 表示从预训练 anchor 到 adapted weight 的 delta；
- ADI 只校准该 delta 的部署强度。

### 6.3 Alpha selection 必须写清楚

必须明确：

- alpha 只由 validation bicubic / bilinear 信号选择；
- 不使用 nearest test；
- 不使用 CIFAR-C test；
- 不使用目标 corruption label 或 severity 信息；
- diagnostic full-alpha sweep 不能作为主表选择依据。

推荐表述：

> The interpolation coefficient is selected without access to the target-shift test sets. Nearest-neighbor interpolation and CIFAR-C corruptions are used only for evaluation.

### 6.4 复杂度写法

应强调：

- 不增加训练参数；
- 不增加推理模块；
- 权重合并后推理计算图与原 LoRA/DoRA 等效；
- 额外成本主要来自 validation alpha sweep。

不要写：

> free method

应写：

> no additional inference-time module, with a small post-hoc validation overhead for alpha selection.

---

## 7. Experiments 证据阶梯

实验部分不要按时间线写，而要按证据逻辑写。

### 7.1 第一层：CIFAR-100 interpolation

回答问题：

> Does ADI improve interpolation robustness while preserving clean accuracy?

应报告：

- Bicubic；
- Bilinear；
- Nearest；
- Absolute Drop；
- RRR；
- seed42/43/44 mean±std；
- LoRA vs ADI-LoRA；
- DoRA vs ADI-DoRA。

### 7.2 第二层：CIFAR-100-C full19 severity=3

回答问题：

> Does the benefit extend beyond resize interpolation to corruption robustness?

应报告：

- 19 类 corruption mean；
- 每类 corruption 或类别分组；
- severity=3；
- LoRA/DoRA 3-seed mean±std；
- ADI 相对 baseline 的 mean gain；
- 多数 corruption 类别是否正向。

### 7.3 第三层：Tiny-ImageNet / ImageNet-100

回答问题：

> Does the effect transfer to larger or more diverse datasets?

如果只做 seed42，要明确写：

> dataset-level extension evidence

不要和 3-seed 主表混为同等强度。

如果也有 3-seed，则可以作为强主结果。

### 7.4 第四层：Alpha selection stability

回答问题：

> Is the selected alpha stable and selected without target-shift leakage?

建议表格包含：

- dataset；
- method；
- seed；
- selected alpha；
- validation bicubic；
- validation bilinear；
- test nearest；
- CIFAR-C mean；
- diagnostic_only flag。

### 7.5 第五层：Efficiency

回答问题：

> Does ADI improve robustness without inference overhead?

应报告：

- trainable params；
- total params；
- throughput；
- peak memory；
- training time per epoch；
- alpha sweep overhead；
- inference module overhead = 0。

### 7.6 第六层：Mechanism diagnostics

回答问题：

> Why might delta interpolation improve robustness?

可使用：

- LFER；
- CKA / FeatureSim；
- SpecDist；
- delta norm；
- alpha vs robustness trend。

表达要克制：

> These diagnostics suggest that ADI reduces shift-sensitive feature or spectral deviations.

不要写：

> These diagnostics prove the mechanism.

### 7.7 第七层：Boundary diagnostics

回答问题：

> Where does the current formulation apply, and where is it limited?

VPT / Adapter 写法：

- 不进入主表；
- 不作为主方法有效性；
- 可作为 diagnostic-only full-alpha sweep；
- 用于说明 current validation selection 的边界；
- 不改变主张范围。

推荐结论：

> ADI is most directly supported for LoRA/DoRA-style weight-delta PEFT. Prompt- and residual-adapter-based methods reveal related diagnostic behavior, but require more careful anchor definitions and are left as boundary analysis.

---

## 8. Results 写作原则

每个结果小节必须遵循：

1. 先说该实验回答什么问题；
2. 再给主结果；
3. 再解释趋势；
4. 最后说明限制或边界。

不要只写：

> Table 1 shows the results.

应写：

> Table 1 evaluates whether calibrating the PEFT delta improves robustness to interpolation shifts. Across both LoRA and DoRA, ADI improves Nearest accuracy while preserving Bicubic and Bilinear accuracy, indicating that the full PEFT delta is not always the most robust deployed update.

---

## 9. Discussion 写作重点

Discussion 不要重复实验数字，应回答三个问题。

### 9.1 ADI 为什么有意义

重点：

- 不是重新训练新 adapter；
- 不是增加数据增强；
- 不是使用 target OOD test 调参；
- 而是发现 PEFT delta 的部署强度本身需要校准。

### 9.2 ADI 为什么不是万能

必须承认：

- 当前主证据集中于 LoRA/DoRA；
- VPT/Adapter 需要不同 anchor 定义；
- alpha selection 仍可能不是最优；
- 更大规模 ImageNet-1K 不是当前证据范围。

### 9.3 后续工作

可写：

- target-shift-free robust alpha selection；
- anchor-aware ADI for prompt/residual modules；
- larger-scale benchmarks；
- theoretical understanding of delta calibration and robustness。

---

## 10. 用词边界

### 10.1 推荐用词

| 场景 | 推荐表达 |
|---|---|
| 机制解释 | suggest, indicate, provide evidence |
| 效果 | consistently improves, yields gains, reduces drop |
| 方法范围 | LoRA/DoRA-style weight-delta PEFT |
| 选择规则 | target-shift-free validation selection |
| 复杂度 | no additional inference-time module |
| 边界 | diagnostic boundary, limitation, scope |

### 10.2 避免用词

| 避免表达 | 替代表达 |
|---|---|
| prove | provide empirical evidence |
| solve | mitigate / improve |
| guarantee | indicate / suggest |
| universal | LoRA/DoRA-style |
| optimal alpha | selected alpha |
| no cost | no inference-time module; small validation overhead |
| all PEFT methods | weight-delta PEFT methods considered in this work |

---

## 11. 可直接采用的关键句

### 11.1 核心问题句

> If a pretrained vision transformer already provides a strong visual prior, should a PEFT model always inject the entire learned delta at inference time?

### 11.2 方法定位句

> ADI calibrates the deployment strength of LoRA/DoRA weight deltas by interpolating between the pretrained anchor and the adapted PEFT weights.

### 11.3 无泄漏声明

> The interpolation coefficient is selected without access to nearest-neighbor or corruption test sets; these shifts are used only for evaluation.

### 11.4 结果解释句

> The consistent gains suggest that robust PEFT adaptation depends not only on learning an effective delta, but also on calibrating how much of that delta should be deployed.

### 11.5 边界声明

> The present study focuses on LoRA/DoRA-style weight-delta PEFT. Prompt-based and residual-adapter-based methods are analyzed as diagnostic boundary cases rather than main claims.

### 11.6 克制贡献句

> Our results provide empirical evidence that post-hoc delta calibration can improve interpolation and corruption robustness without introducing additional inference-time modules.

---

## 12. 建议全文结构

1. Abstract
2. Introduction
   - PEFT 背景
   - 完整 delta 不一定最鲁棒
   - 核心问题
   - ADI 方法
   - 贡献
3. Related Work
   - PEFT for vision transformers
   - LoRA / DoRA and weight-delta adaptation
   - Robustness under interpolation and corruption shifts
   - Post-hoc model calibration / interpolation
4. Method
   - Problem formulation
   - ADI for LoRA
   - ADI for DoRA
   - Target-shift-free alpha selection
   - Complexity analysis
5. Experiments
   - Setup
   - CIFAR-100 interpolation robustness
   - CIFAR-100-C corruption robustness
   - Dataset extension
   - Alpha selection analysis
   - Efficiency
   - Mechanism diagnostics
   - Boundary diagnostics
6. Discussion
   - What ADI reveals about PEFT deltas
   - Why zero inference overhead matters
   - Scope and limitations
7. Conclusion

---

## 13. 一区风格的关键要求

如果论文要按 JCR 一区可能性组织，写作重点不是夸大方法，而是做到：

1. 问题足够清楚：完整 PEFT delta 不一定最鲁棒。
2. 方法足够简洁：从 pretrained anchor 插值 delta。
3. 协议足够严格：alpha selection 无 target-shift leakage。
4. 证据足够系统：LoRA/DoRA、3-seed、多数据集、多 shift。
5. 边界足够诚实：VPT/Adapter 不作为主方法。
6. 机制足够可信：LFER/CKA/SpecDist 至少支持趋势。
7. 工程价值明确：无额外推理模块，低部署成本。

最重要的是把论文从“一个涨点技巧”写成：

> A phenomenon-driven study showing that robust PEFT deployment may require calibrating the learned delta, not merely learning it.

---

## 14. 最终写作准则

全文应保持以下气质：

- 问题驱动，而不是模块驱动；
- 简洁明确，而不是堆叠复杂术语；
- 证据强，但语言克制；
- 主张收束，但意义上升；
- 主表稳健，边界透明；
- 不逃避负结果，但不让负结果稀释主线。

一句话总结：

> 学 ResNet 的“反常现象 -> 简洁重表述 -> 系统实证 -> 清晰边界”，但不要学它的历史级突破语气。ADI-LoRA/DoRA 的最佳写法，是把它塑造成一个严谨、低成本、证据充分的 PEFT delta calibration 原则。
