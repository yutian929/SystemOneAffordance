# System One 用于 Affordance：已有工作与研究边界

调研日期：2026-10-15。本文围绕我们当前关注的任务：**输入 RGB 图像和自然语言指令，输出与指令对应的交互区域分割掩码**，以 Affordance-R1 为主要参照。基础概念与数据集见 [01](01_affordance_primer.md)，模型接口见 [02](02_visual_system_one_models.md)。

## 先回答“有没有人做过”

需要分三个层次回答：

| 问题 | 本轮调研结论 |
| --- | --- |
| 有没有直接用 Jev、OneJev、OmniJev、d1、decider 等决策模型，完成类似 Affordance-R1 的指令条件区域分割？ | **尚未找到可核实的公开论文或项目。** 目前未找到同时给出这类模型的实际接入、交互区域 mask 输出，以及对应分割实验的工作。 |
| 有没有 System One 与广义 Affordance 结合的工作？ | **有。** REIS 明确设计了 System One 内的 Affordance Router；OmniJev 已有机器人图像上的动作、状态与进度判断。这些与可交互性相关，但输出不是像素级交互区域。 |
| 有没有“不先生成文字推理，直接利用 VLM 特征预测 Affordance”的工作？ | **有。AFUN 尤其接近。** 因此，“快速视觉理解”“省去 CoT”“直接接分割头”本身不能当成尚无人探索的贡献。 |

这里的“未找到”是检索结论，不是不存在的证明，更不能据此写“首次”。上述相关工作的具体证据、输入输出和开放状态如下。

## 这里的 System One 指什么

沿用 02 的工作定义，本文重点调查的是**将视觉或文本状态转换成类型化判断、候选分数或概率的决策模型**，例如 Jev、OneJev、d1。它们通常避免逐 token 生成回答，但具体计算路径并不完全相同。

另有两种常见用法需要区分：

- **系统架构中的 System 1**：把快速感知、动作执行或状态检查称为 System 1，把复杂规划称为 System 2。REIS 属于这一层；这个名称并不意味着它调用了 Jev。
- **直接预测网络**：VLM 特征接 mask decoder，或者传统网络直接预测热力图。它们可以具有快速、非文本生成的特点，但不自动等于 02 中的决策模型。

同样，Affordance 可以指“某动作现在是否可执行”，也可以指“图像中哪里支持这个动作”。本项目目前关注后者。动作正确率、任务成功率、候选框命中率，都不能替代交互区域的分割指标。

## 最值得关注的已有工作

### 1. OmniJev：已有视觉机器人判断，也支持区域候选

**与我们的关系：最接近“拿现成 System One 模型做视觉交互判断”的开源起点，尚不是 Affordance 分割方法。**

注意命名：这里是 [`tinnel123666888/OmniJev`](https://github.com/tinnel123666888/OmniJev)，与 [`OmniJev/OneJev`](https://github.com/OmniJev/OneJev) 是两个不同仓库和模型项目。

OmniJev 基于 Qwen3.5，使用 LoRA 与决策头，提供 0.8B、2B、4B 版本及代码、权重。它接受图像与问题，返回 Choice、Noul、Score 等结果；候选项还可以带图像区域框。这说明“图像区域作为决策候选”已有实现。[官方说明与接口](https://github.com/tinnel123666888/OmniJev#what-it-is)

我进一步核对了 [`mso/templates.py`](https://github.com/tinnel123666888/OmniJev/blob/main/mso/templates.py)：机器人模板询问夹爪下一步移动方向、开合动作、是否持物、任务完成情况和进度。它们属于动作及状态判断；模板没有定义像素 mask 输出。将这些能力用于部件可供性评分，是合理的后续研究设想，尚不是已报告的分割结果。

其机器人演示是**记录轨迹上的离线回放**，仓库也注明部分演示输入可能与训练重合，不能按独立测试或模型闭环控制成功来理解。[演示说明](https://github.com/tinnel123666888/OmniJev/blob/main/docs/demos_v11.md)

### 2. REIS：明确存在 “System One + Affordance Router”

论文：**On-Device Robotic Planning: Eliminating Inference Redundancy for Efficient Decision-Making**，2026-05。资源：[论文](https://arxiv.org/abs/2605.31460)、[正文 §4.1、§5.3 与附录 A.1](https://arxiv.org/html/2605.31460v1)。本轮未核实到作者公开的配套代码仓库。

它的快速模块读取视觉观察与当前规划状态，检查子目标完成情况、后续转换是否有效、当前动作是否应继续，再决定是否调用慢速规划模块。Affordance Router 主要使用 **Qwen3-VL-4B-Instruct**，向 KV cache 注入离线构造的 steering 信号，直接读取 Yes/No 概率，不生成回答 token。评测涉及 ALFRED 与真实机器人任务，目标是减少重复规划和发现执行异常。

**它已证明有人研究“无文本生成的视觉 Affordance 判断”，但判断对象是规划状态和动作条件，输出并非交互区域 mask。** 整个 REIS 仍包含慢速推理模块，不能把局部 router 的零输出 token 扩大为整套系统完全不生成文字。

### 3. AFUN：对我们方法设计最直接的提醒

论文：**AFUN: Towards an Affordance Foundation Model for Functionality Understanding**，2026-06。资源：[论文](https://arxiv.org/abs/2606.02551)、[项目](https://www.zhaoningwang.com/AFUN/)、[代码](https://github.com/EricWang12/AFUN)。

AFUN 将可学习的 MetaQuery 放到冻结的 Qwen3-VL 输入中，提取对应隐藏状态，经 MLP 接到 SAM3 预测功能区域。它不依赖先生成文字推理或分割标记；还通过几何分支预测接触后的三维运动。完整模型使用 RGB-D 与任务描述，二维分割分支处理 RGB 和语义特征。[方法 §4](https://arxiv.org/html/2606.02551v1#S4)

它**没有使用 Jev、OneJev 或 d1**，但已经覆盖了“利用 VLM 内部表示直接预测 Affordance mask”的核心思路。对我们而言，AFUN 是比单纯生成式方法更需要比较的近邻。

公开仓库目前提供推理代码与 checkpoint 获取入口，训练代码仍列为 TODO。Demo 可以只提供 RGB，由额外模型估计深度；因此完整 demo 的成本还包括深度估计等步骤，不能把模型中的一次前向理解为整个应用只运行一个网络。[开放范围与运行说明](https://github.com/EricWang12/AFUN#-todo)

### 4. UniAfford：隐藏状态路由，但仍有自回归推理

论文：**UniAfford: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception**，2026-09。资源：[论文](https://arxiv.org/abs/2609.37264)、[项目](https://4dvlab.github.io/UniAfford/)、[代码](https://github.com/4DVLab/UniAfford)、[模型页](https://huggingface.co/yiqian7a/UniAfford-5B)。

它将 MLLM 的隐藏状态路由到二维与三维 Affordance 分支，支持仅图像输入的二维预测，值得参考如何把语义表示交给密集预测模块。

但有一个容易误读的地方：**不要求语言头生成固定任务标记，不等于取消自回归。** 论文 §4.2 明确构造自回归响应状态，附录 B.1 也明确描述自回归推理期间的路由。因此不能把它列为已经采用 Jev 式零输出 token 推理的工作。[方法与附录](https://arxiv.org/html/2609.37264v1)

仓库提供训练、验证和 demo 入口；README 仍有部分占位说明，权重链接已从 4B 名称重定向至 5B。这里确认的是公开资源入口，尚未进行下载和复现。[仓库说明](https://github.com/4DVLab/UniAfford/blob/master/README.md)

### 5. EgoLens：任务规划与区域分割的相邻路线

**EgoAfford / EgoLens** 面向第一视角图像和高层任务，同时预测剩余计划与下一步动作涉及的直接对象、工具、目标位置三类区域。EgoLens 基于 Qwen2.5-VL-3B、可学习查询与 SAM2，不使用上述 System One 决策模型。[论文](https://arxiv.org/abs/2608.04533)、[项目](https://egoafford.github.io/)

仓库区分完整任务和 **EgoLens-Seg**：后者接受已经给定的动作步骤来预测区域。比较时必须区分动作步骤来自标注还是模型预测，不能把获得正确下一步动作的分割成绩当成完整任务成绩。代码、训练与评测入口已公开。[代码与评测说明](https://github.com/Pandenotium/EgoLens)

它提示我们：**高层任务理解、下一步动作选择、区域分割可以拆开研究**。这给决策模型留下了潜在接口，但“用 System One 替换其中一个模块”还需要验证，不能当作已有事实。

## 对 02 中各模型的逐项核查

下表记录本轮在官方文档、模型卡、仓库和相关论文中实际找到的用途；“未找到分割工作”不表示模型不可能完成这个任务，也不表示已审计全部训练样本。

| 模型 | 已核实的公开能力或用途 | 与本项目的距离及主要来源 |
| --- | --- | --- |
| Jev | 1.13/1.14 有实验图像接口；1.14 支持 Box；若干机器人项目用于动作选择 | 图像输入和边界框接口不等于 Affordance mask。[Images](https://docs.typesafe.ai/images)、[Box](https://docs.typesafe.ai/primitives/box) |
| OneJev | 图像、视频与文本的类型化决策，公开通用决策与视觉问答评测 | 未找到指令条件交互区域分割实验。[官方仓库](https://github.com/OmniJev/OneJev) |
| OmniJev | 机器人图像问答、区域候选选择，支持本地推理 | 与视觉交互判断相邻；公开机器人示例为离线决策，详见上节。[官方仓库](https://github.com/tinnel123666888/OmniJev) |
| d1-3B / d1-omni-600M | 视觉类型化判断；官方另展示 d1-3B 在 Isaac Sim 中导航、Jetson 端推理 | 导航演示不能视为操作区域分割；发布材料未给出 Affordance mask 评测。[官方发布说明](https://www.liquid.ai/blog/d1-open) |
| decider-2b-vision | 视觉问答、像素输入的游戏动作选择；包含 Pong、Breakout 的 PPO 训练 | 是视觉到离散动作的实例，未找到机器人交互区域分割结果。[模型卡](https://huggingface.co/Mapika/decider-2b-vision) |
| Imajev | 图像与文本上的类型化判断、候选概率与 unknown 输出 | 可参考候选验证与拒答接口；未找到 Affordance 分割结果。[官方仓库](https://github.com/mohit67890/imajev) |
| PlayJev | 游戏帧到动作概率，公开训练与推理流程 | 可参考视觉决策训练，游戏动作表现不能代表部件分割能力。[官方仓库](https://github.com/OmniJev/PlayJev) |

**本轮没有找到这些模型在 ReasonAff、RAGNet 等指令条件 Affordance 分割设置上的可核实结果。** 02 中的模型推荐是可研究性判断，不是已验证的 Affordance 性能排名。

## 已经使用 Jev 的机器人工作，为什么还不算直接答案

| 工作 | 实际输入与输出 | 与我们任务的区别 |
| --- | --- | --- |
| [Jev × LIBERO](https://github.com/Dimweaker/jev-libero) | 模拟器状态、候选动作及局部物理预览，选择平移、转腕、夹爪动作 | 研究状态条件控制，没有从图像分割交互区域 |
| [RoboJEV](https://github.com/lykycy123/RoboJEV) | 结构化模拟器状态 → 意图 → XYZ 方向及夹爪指令 | README 明确说明 Jev 不接收相机图像；使用的是特权状态 |
| [SO101 + Jev](https://github.com/Control-derek/so101-jev) | 两个相机的照片先由另一个视觉模型转换成结构化偏移，再交给 Jev 选动作 | 系统有相机，不等于 Jev 自己进行了视觉区域理解；未输出可供性 mask |
| [FastJEV](https://arxiv.org/abs/2610.11379) | 加速 OmniJev 的候选评分，包含从 LIBERO-10 重构的离线图像问题 | 这里的 LIBERO 成绩是离线问答准确率，不是机器人闭环成功率，也不是分割成绩 |
| [Can Jev be Your Q or Policy in Reinforcement Learning?](https://arxiv.org/abs/2610.11692) | 在 MiniGrid、Atari 中将 Jev 用作参考策略、探索判断或 replay 评分组件 | 研究决策模型如何辅助 RL 学习，没有报告交互区域分割 |

前三项有公开仓库，但 Jev 推理依赖其服务，不能因应用代码公开就称 Jev 权重开源。FastJEV 的[代码入口](https://github.com/JieMaMagic/FastJEV)由论文提供；其使用的是 OmniJev，不能与 TypeSafe 的闭源 Jev 混写。

## 这些发现怎样影响我们的研究判断

下面是基于上述证据的分析，尚不是实验结论。

### 有研究空间，但需要把问题提得更具体

当前值得研究的问题是：**经过类型化决策训练的视觉模型，能否比同规模基础 VLM 更高效、可靠地判断“哪个区域支持这条指令”，并将判断转成准确的像素区域？**

这个问题比“把 System One 用于 Affordance”更可检验。已有 REIS 的快速可行性判断、OmniJev 的区域候选接口，以及 AFUN 的直接特征分割，因此不能仅靠更换模型名称、去掉 CoT 或接一个 SAM 来说明方法贡献。

### 两种可能接口，对应不同研究内容

| 接法 | System One 负责什么 | 需要解决的核心问题 |
| --- | --- | --- |
| 候选区域评分 | 给图像中的候选部件、点或 mask 打分，再选出或组合交互区域 | 候选能否覆盖正确部件？模型是否理解功能而非只识别物体？多个可用区域怎样表示？ |
| 特征接分割头 | 读取开放权重模型的视觉语言表示，以密集监督学习 mask | 决策训练究竟带来了什么？相对基础 VLM 与 AFUN 式特征接口是否有收益？ |

前一种更贴近现成决策 API；后一种更贴近分割模型研究，需要能访问模型内部表示。两者都不能仅凭“支持图像输入”推断分割效果。

有几项边界会直接影响实验是否成立：

- **区域选择概率不等于像素概率。** Choice 通常在候选之间竞争；同一动作可能对应多个有效区域。单选分数直接铺成热力图未必合理，独立区域判断也需要重新验证校准。
- **候选生成决定覆盖上限。** 如果候选里没有杯柄，后续评分再准确也无法分割杯柄。最终模型还必须在没有真值候选的测试输入上工作。
- **延迟需要计算完整链路。** 图像编码、候选生成、所有区域查询、分割与后处理都应计入；取消输出 token 不保证整套方法更快。
- **比较应分清能力来源。** Affordance-R1 是任务参照，AFUN 是直接特征预测参照，同骨干基础 VLM 是判断决策训练价值的参照。还需对齐训练数据、输入信息和测试协议，不能仅比较不同论文的最高数字。

如果下一步只精读少量材料，优先看 **OmniJev 的区域/机器人接口、REIS 的快速判断模块、AFUN 的分割接口**；它们分别回答“现成决策模型能做什么”“广义 Affordance 判断已经怎么做”“直接区域预测已经走到哪里”。

## 检索范围与结论边界

本轮使用公开网页搜索，回到作者论文全文、官方项目、GitHub 与 Hugging Face 页面核对。主要检索组包括：

```text
(Jev / OneJev / OmniJev / d1-3B / decider-2b-vision / Imajev)
    + affordance / affordance grounding / segmentation

(Jev / OneJev / d1-3B / Imajev)
    + ReasonAff / RAGNet / AGD20k

(System One / System-1 / System 1)
    + affordance / robot / segmentation

affordance + non-autoregressive / single forward pass
```

搜索命中不能直接作为归类依据。例如 [ACKnowledge（CHI 2025）](https://doi.org/10.1145/3706598.3713791)确实讨论 System 1 与 Affordance，但其实现围绕知识图谱、视觉语言模型和交互规划，不代表使用了本项目讨论的 Jev 类模型。类似地，UI affordance、机器人动作选择、论文摘要被 Jev 处理，都不能自动归入机器人区域分割。

本轮核对了关键论文的方法和推理描述，也查看了 OmniJev 的机器人问题模板；**没有运行模型或完整审计所有仓库代码与训练数据**。新发布、未被索引、私有或尚未公开的工作可能遗漏。开放状态按本次读取的页面记录，提供下载入口不等于已经完成可复现性验证。

后续如果发现同时满足“实际使用指定决策模型、图像与指令输入、交互区域 mask 输出”的项目，应更新本文的直接相关工作结论；若还有可复现的分割实验，才进一步把它列为可比较基线。
