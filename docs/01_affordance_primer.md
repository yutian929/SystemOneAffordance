# Affordance 入门

Affordance 可译为“可供性”，关注环境对某种行动提供的可能性。在视觉定位任务中，它通常被具体化为：**给定图像与动作意图，找出支持该动作的位置或区域**。任务定义可参考 [Cross-view-AG](https://github.com/lhc1224/Cross-View-AG) 与 [AffordanceLLM](https://jasonqsy.github.io/AffordanceLLM/)。

## 从物体识别到动作相关定位

看见一个杯子，物体识别回答“这是什么”，物体分割标出杯子的范围。Affordance 则需要进一步考虑动作。例如，拿起杯子时可能关注把手或杯身；向杯中倒水时关注杯口；同一个物体会随动作产生不同的目标区域。

这些例子只是直观说明。具体数据集可能把“动作作用部位”和“手接触部位”标成不同区域，因此使用数据前必须阅读标注定义。例如“用刀切割”既可能涉及刀刃的功能，也可能涉及握住刀柄；没有统一动作语义就无法比较预测对错。

可供性也与执行者和场景有关：人能握住的部位不一定适合某个夹爪，视觉上合理的接触点也可能被障碍遮挡。图像标签只描述了任务所定义的部分条件。

## 2D 描述的是输出坐标系

2D Affordance 的输出位于图像平面，通常用像素坐标 `(u, v)` 或归一化坐标表示。这不限制机器人最终只能进行二维运动。

设图像为 `I`，动作类别或语言指令为 `q`，可以把任务写成 `A = f(I, q)`。常见的 `A` 有三种形式：

| 输出 | 含义 | 适合回答的问题 |
| --- | --- | --- |
| 稠密热力图 | 为每个像素赋一个动作相关得分 | 哪些区域更适合交互？ |
| 区域掩码 | 把一组像素标为可交互区域 | 哪一部分支持该动作？ |
| 一个或多个点 | 输出具体交互位置或空间目标 | 应该指向或接触哪里？ |

[AffordanceLLM](https://jasonqsy.github.io/AffordanceLLM/) 采用图像加动作、输出热力图的设置；[RoboPoint](https://github.com/wentaoyuan/RoboPoint) 使用语言条件下的图像关键点表示。区域预测与点预测可以相互转换，但转换会引入阈值、平滑核或区域选择规则，评测时应记录这些规则。

## 与相邻任务的关系

| 任务 | 主要输出 | 和本方向的关系 |
| --- | --- | --- |
| 目标检测 | 类别与边界框 | 可提供候选对象，通常不细分动作相关部位 |
| 语义或实例分割 | 物体或类别掩码 | 可提供候选区域，功能部位可能小于完整物体 |
| 指代表达定位 | 与语言指代匹配的目标 | 与 Affordance 共享视觉定位，但动作可行性要求可能不同 |
| 2D Affordance Grounding | 动作相关点、区域或热力图 | 本项目初步关注的任务 |
| 抓取姿态预测 | 夹爪位置、方向、开口等 | 包含更多几何与执行约束 |
| 机器人策略 | 随观测变化的动作序列 | 还需处理控制、时序、接触和失败恢复 |

因此，区域定位准确、抓取姿态可行和闭环任务成功应分别评价。

## 三种分数不要混用

1. **热力图得分**表示像素间的相对相关性。归一化为总和等于一后，也不自动具有物理成功概率的含义。
2. **候选选项概率**表示在给定候选集合中的选择分布。候选改变时，分布可能随之改变。
3. **操作成功概率**需要明确机器人、动作参数、初始状态和成功事件，并用实际执行结果验证。

一个区域得到 0.9 的模型输出，不能直接解释为“机器人在这里操作有 90% 的成功率”。如果研究校准，首先要定义它校准的是候选分类正确率、区域标签还是实际执行结果。

## 近期 SOTA 与代表模型

更新日期：**2026-10-15**。这里聚焦 2D Affordance，收录近期论文报告的领先方法，以及值得保留的强基线。**SOTA 要结合数据集、划分、监督方式和指标来看**：AGD20K 的热力图预测、ReasonAff 的指令驱动区域分割，以及新动作类别泛化，并不是同一张排行榜。以下结果来自作者报告，本项目尚未复现。

| 方法 / 时间 | 简单介绍 | 基准与定位 | 论文 / 项目 |
| --- | --- | --- | --- |
| **Affordance-R1** · AAAI 2026 | 用 GRPO 强化学习训练视觉语言模型，根据图像和隐含意图进行推理，输出框与点，再由分割模型生成交互区域。 | ReasonAff 上的推理型强基线，也测试 UMD、AGD20K 子集上的跨数据集泛化。已有后续方法报告更高成绩。 | [论文](https://arxiv.org/abs/2508.06206) · [代码](https://github.com/hq-King/Affordance-R1) · [权重](https://huggingface.co/hqking/affordance-r1) |
| **SpatialAfford** · 2026.08 预印本 | 基于 Qwen3-VL-4B，先监督注意力聚焦交互部位，再用空间奖励与 GRPO 改善坐标预测。 | 作者报告在 ReasonAff、PartAfford 上超过 Affordance-R1；ShareRobot-Bench 的部分指标领先，并非所有指标第一。 | [论文](https://arxiv.org/abs/2608.00502)；本轮未查到作者公开的项目或代码 |
| **A4-Agent** · ECCV 2026 | 无需任务微调，组合“想象交互场景、判断交互部件、检测与分割定位”三个阶段；是多模型协作框架。 | 作者报告在 ReasonAff、RAGNet、UMD 上取得领先结果，适合了解零样本推理路线。 | [论文](https://arxiv.org/abs/2512.14442) · [项目](https://zixinzhang02.github.io/A4-Agent-page/) · [代码](https://github.com/EnVision-Research/A4-Agent) |
| **TokAG** · ECCV 2026 | 从视觉语言模型生成的 token 中选择具有空间定位信息的注意力图，构建动作相关热力图，无需额外训练。 | 作者报告 AGD20K Unseen 与 HICO-IIF 的 NSS 超过此前弱监督方法；不代表超过所有全监督模型。 | [论文](https://arxiv.org/abs/2607.03595) · [项目](https://drfirstlee.github.io/tokAG_proj/) · [代码](https://github.com/DrFirstLee/tokAG) |
| **AffordanceSAM** · ACM MM 2026 | 将 SAM 适配到可供性定位，结合 C2F-Aff 数据与从粗到细的三阶段训练，预测功能区域。 | 作者报告在 AGD20K 上达到 SOTA；属于全监督路线，需要与弱监督、免训练方法区分。 | [论文](https://arxiv.org/abs/2504.15650)；本轮未核实到作者公开代码 |
| **WSAG-PLSP** · ICLR 2025 | 用部件语义和现成部件分割模型产生伪标签，再进行标签细化、特征对齐和轻量推理。 | AGD20K Seen / Unseen 的弱监督强基线，适合与热力图方法比较。 | [论文](https://arxiv.org/abs/2505.24103) · [代码与权重入口](https://github.com/woyut/WSAG-PLSP) |
| **AffordAnything / AffordAnything+** · 2026.08 预印本 | 利用物体部件的分割线索定位可供性；基础版无需训练，增强版学习组合这些线索。 | 面向训练中未出现过的动作类型；作者报告增强版在 AGD20K-NAT 的 IoU@0.4 超过 OOAL。NAT 划分不同于标准 AGD20K。 | [论文](https://arxiv.org/abs/2608.08929)；本轮未查到作者公开代码 |
| **AffordanceVLM / AffordanceNet（RAGNet 工作）** · ICCV 2025 | 用大规模指令与区域标注训练 AffordanceVLM，再将预测区域用于抓取。RAGNet 是这项工作的数据集与基准名称。 | 面向抓取的分割强基线，评测覆盖 HANDAL、GraspNet、3DOI，并包含真机实验。 | [论文](https://arxiv.org/abs/2507.23734) · [代码、权重与数据入口](https://github.com/wudongming97/AffordanceNet) |

### Affordance-R1 可以怎样理解？

可以把它理解为一个“读懂操作意图，再定位交互部位”的系统。例如，输入水龙头图片和“我想让水流出来”，模型需要推断应该操作哪个部位。其公开实现采用 Qwen2.5-VL-7B 与 SAM2，流程大致为：**图像 + 指令 → 推理与再检查 → 边界框和点 → 分割掩码**。[论文](https://arxiv.org/html/2508.06206v5)、[官方实现](https://github.com/hq-King/Affordance-R1)。

训练的关键是 GRPO：通过输出格式、空间定位和可供性语义奖励优化模型。作者构建了 ReasonAff 数据集，在该任务的后训练阶段直接进行 RL，不额外用人工推理链做监督微调；这并不意味着模型没有基础预训练或任务标注。[方法说明](https://arxiv.org/html/2508.06206v5#S4)。

它值得作为我们讨论 System One 的参照，因为它采用显式推理路线。**目前不能把 Affordance-R1 简称为整个领域的最强模型**：例如 SpatialAfford 的 ReasonAff 对比表报告 gIoU 为 74.92，而表中的 Affordance-R1 为 67.41。这是作者在其协议下的结果，实际比较仍需核对数据版本和评测实现。[SpatialAfford 结果](https://arxiv.org/html/2608.00502v1#S4)。

## 图像与指令输入、交互区域分割：榜单与数据集

更新日期：**2026-10-15**。当前希望跟进的是 Affordance-R1 这类 **instruction-conditioned affordance segmentation（指令条件下的可供性分割）**：输入 RGB 图像与自然语言指令，输出与指令对应的交互区域掩码。因此，选择数据集时应优先考虑是否提供指令和像素级区域标注，以及是否需要从任务意图推断交互部位。

### 有没有榜单？

**有公开的结果汇总，但本轮未找到覆盖上述任务、统一接受提交并持续维护的官方总榜。** 目前更实用的入口是：

| 入口 | 能看到什么 | 怎样使用 |
| --- | --- | --- |
| [SOTA2：Affordance Mask Segmentation](https://www.sota2.com/research/task/affordance-mask-segmentation) | 按 ReasonAff、InstructPart、HANDAL 等数据集聚合的论文结果 | 作为发现论文的第三方索引，分数与设置回到原论文核对 |
| [A4-Agent 官方结果页](https://zixinzhang02.github.io/A4-Agent-page/) | ReasonAff、RAGNet、UMD 上与 Affordance-R1、AffordanceVLM 等方法的对比表 | 很适合了解这条研究路线常用的测试组合；属于作者实验表 |
| [Affordance-R1 论文](https://arxiv.org/html/2508.06206v5) | ReasonAff 主实验，以及 UMD、AGD20K 的跨数据集测试 | 直接对标 Affordance-R1 时，先对齐这里的数据与指标 |

搜索时也会遇到 [SceneFun3D 的 EvalAI 官方榜单](https://eval.ai/web/challenges/challenge-page/2466/leaderboard/6121)，但它评测的是 3D 场景中的功能分割与定位，不是这里的单张图像掩码任务。

此外，值得补看 [AFUN 论文](https://arxiv.org/html/2606.02551v1)与[代码](https://github.com/EricWang12/AFUN)。作者报告 ReasonAff 分割 gIoU/cIoU 为 78.1/78.2，但训练使用了 ReasonAff、InstructPart、RAGNet、HOVA-500K 等混合数据，不能与零样本方法直接归为同一设置。它的完整系统还预测 3D 运动；对我们而言主要关注其分割分支。本轮仓库提供推理代码，训练代码仍列为待发布。

### 对标 Affordance-R1，哪些数据集优先做？

建议以 **ReasonAff 为主测试，RAGNet 推理子集和 UMD 为泛化测试，AGD20K 为补充**。这是针对当前目标的建议，不是领域统一规定的“必刷清单”。

| 优先级 / 数据集 | 数据与任务特点 | 为什么用、用哪个设置 | 资料与下载入口 |
| --- | --- | --- | --- |
| **首先做：ReasonAff** | 基于 InstructPart 的图像与部件掩码，重写为更隐含、依赖场景的任务指令；需要判断“该操作哪里”。 | Affordance-R1 的直接训练与主测试基准，最贴近当前目标。 | [论文](https://arxiv.org/abs/2508.06206) · [官方仓库](https://github.com/hq-King/Affordance-R1) · [数据](https://huggingface.co/datasets/hqking/affordance-R1) |
| **重点扩展：RAGNet** | 面向抓取的图像、指令与区域标注；完整数据包含约 27.3 万图像、180 类物体、2.6 万推理指令，这些不是测试集规模。 | 优先看 **HANDAL-easy、HANDAL-hard、3DOI-easy** 的推理分割设置；其中 3DOI 用于考察跨图像域泛化。不要与普通分割版、HANDAL 全集混用。 | [论文](https://arxiv.org/abs/2507.23734) · [划分与下载说明](https://github.com/wudongming97/AffordanceNet/blob/master/docs/dataset.md) · [数据](https://huggingface.co/datasets/Dongming97/RAGNet) |
| **重点扩展：UMD Part Affordance** | 厨房、园艺和车间工具的 RGB-D 数据，包含 105 件工具及 grasp、cut、contain 等 7 类部件可供性；原始标注是类别掩码，不是复杂自由指令。 | 用来检查跨数据集的功能部位识别。Affordance-R1 使用抽取的 **1,922 张图像**，应复用相同样本清单与提示词；本项目的 RGB 设置不额外输入深度。 | [原始数据与介绍](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index.html) · [R1 的测试协议](https://arxiv.org/html/2508.06206v5#S5) |
| **补充：AGD20K** | 经典 2D 可供性定位数据，含 36 类动作；通常以动作类别为条件，评价热力图，而非复杂语言推理。 | 便于与历史工作衔接。复现 R1 时用 **Seen/test 的 1,710 个物体–动作对**；标准 Seen/Unseen 热力图评测另行报告。 | [数据与原论文入口](https://github.com/lhc1224/Cross-View-AG) · [常用评测实现 LOCATE](https://github.com/Reagan1311/LOCATE) |
| **可选补充：InstructPart** | 2,400 张图像，1,800 训练 / 600 测试，提供部件掩码与多种指令；TRPS 要从任务指令推断部件，ORPS 直接给出部件名称。 | TRPS 很贴合我们的输入输出，也适合研究训练数据量的影响。但 ReasonAff 由它构建，不能把两者当成天然独立的数据来源。 | [项目与论文](https://zifuwan.github.io/InstructPart/) · [数据说明](https://github.com/zifuwan/InstructPart) |

**ReasonAff 的版本需要特别留意。** 官方 README 仍提示部分原始标注较粗，计划清理测试数据；Hugging Face 下载需要登录并接受访问条件。本轮未下载文件验证版本，因此不能预设当前公开包与论文中的测试清单完全一致。后续对比应固定文件版本和样本清单，并在同一套数据上重跑基线。[仓库说明](https://github.com/hq-King/Affordance-R1#dataset)、[数据访问页](https://huggingface.co/datasets/hqking/affordance-R1)。

如果用 InstructPart 或其他混合数据训练，再测试 ReasonAff，应按原始图像检查训练/测试重叠；只改写指令并不能使同一张图像成为新的独立测试样本。

### 看哪些指标，才算在比较同一个任务？

针对掩码输出，先采用 Affordance-R1 的四个指标。按其[官方评价代码](https://github.com/hq-King/Affordance-R1/blob/main/evaluation_scripts/calculate_iou.py)：

- **gIoU**：先计算每个图像–指令样本的掩码 IoU，再求平均。这里不是检测框的 Generalized IoU。
- **cIoU**：所有样本的交集像素数之和，除以并集像素数之和。
- **P@50**：掩码 IoU 大于 0.5 的样本比例。
- **P@50:95**：在 0.50 到 0.95、间隔 0.05 的阈值上，平均上述通过比例；不是检测任务的 mAP。

AGD20K 的 KLD、SIM、NSS 用于热力图比较，可以补充报告，但不能替代主要的掩码指标。框定位准确或接触点命中，也不等于完整交互区域分割准确。

对当前目标，先把 **ReasonAff 上同一版本的 Affordance-R1** 作为直接基线，再看 RAGNet 和 UMD 上能否保留提升。若同时关心 System One 的速度优势，还应比较从图像输入到最终掩码的完整耗时，包括后续分割模型。

## 初学者的阅读路径

先理解 [AGD20K 与 Cross-view-AG](https://github.com/lhc1224/Cross-View-AG) 的动作和区域对应关系，再看 [LOCATE](https://github.com/Reagan1311/LOCATE) 如何借助交互图像学习部件定位。随后阅读 AffordanceLLM 与 RoboPoint，比较热力图和点两种表示；再对照 Affordance-R1 的推理与分割路线、TokAG 的免训练热力图路线。前面的早期方法用于理解任务，不都属于当前 SOTA。
