# 视觉 System One 模型

核对日期：2026-10-15。

本项目把“视觉 System One 模型”作为工作术语：模型直接接收图像及问题，返回候选项分数、概率或类型化答案，用于快速决策。这个称呼不意味着各模型使用相同架构、训练目标或校准方法。选型时需要检查实际推理路径。

## Jev 已有实验性图像接口

TypeSafe 的 [Images 文档](https://docs.typesafe.ai/images) 明确说明，Jev 1.13 和 1.14 都接受图像，默认关闭，需在请求中设置：

```text
TypeSafe-Danger-Zone-Images: true
```

图像可出现在状态、问题说明与候选条件中。该功能被官方标为实验性质，接口与准确率仍可能变化。[Box](https://docs.typesafe.ai/primitives/box) 输出图像中的目标边界框，需要 Jev 1.14；边界框本身并不等价于可供性区域。

[模型页](https://docs.typesafe.ai/models) 将 2026-10-14 发布的 1.14 列为新版本，本轮查看时 `jev-preview` 指向 1.14、`jev-latest` 仍指向 1.13。模型汇总表仍写 Text，实验图像能力的具体说明应结合 Images 页面阅读。[Python SDK v0.7.5 更新记录](https://docs.typesafe.ai/sdk/python/changelog) 同日加入 Image 与 Box 支持。

因此，“早期机器人项目没有直接向 Jev 输入图像”和“现在 Jev 不支持图像”不能视为同一个结论。前者描述具体项目，后者已不符合当前官方说明。以上是文档确认，尚未在本项目调用 API 验证。

## 模型与资源对照

| 模型 | 图像与输出接口 | 开放范围 | 本项目可研究的用途 |
| --- | --- | --- | --- |
| [Jev](https://docs.typesafe.ai/images) | 图像与文本混合；Choice、Noul、Score；1.14 增加 Box | 托管 API | 作为闭源服务对照，测试原生视觉决策 |
| [OneJev](https://github.com/OmniJev/OneJev) | 图片、视频、文本；类型化问题与选项概率 | Apache-2.0；代码、权重和部分训练数据公开 | 通用视觉候选评分与后续微调 |
| [decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) | 图片加问题和选项；从答案位置读取选项 logits | Apache-2.0；[代码](https://github.com/Mapika/decider)和权重公开 | 小模型离散区域或动作选择 |
| [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 图像和文本；直接返回类型化答案 | 开放权重，LFM Open License | 本地视觉决策基线 |
| [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | 图像或音频与文本；选项分布 | 开放权重，LFM Open License | 小模型资源占用与精度对照 |
| [Imajev-4B](https://github.com/mohit67890/imajev) | 最多两张图；类型化结果与 unknown 概率 | Apache-2.0；基础模型另配，[LoRA 与决策头](https://huggingface.co/mohit67890/imajev-4b)公开 | 候选验证、图像比较与拒答研究 |
| [PlayJev-0.8B](https://github.com/OmniJev/PlayJev) | 单帧游戏画面与动作列表；动作概率 | Apache-2.0；训练与推理代码、权重公开 | 参考从像素到离散动作的训练流程 |

表中的研究用途是本项目的选型建议，不表示这些模型已在机器人 Affordance 上得到验证。LFM Open License 与 Apache-2.0 的授权条件不同，具体使用以各模型的许可文件为准。

## 各模型的关键边界

### OneJev

提供 0.8B、4B、9B、27B 等规模。[OneJev-4B 模型卡](https://huggingface.co/OmniJev/OneJev-4B) 包含权重与图像请求示例，接口与 TypeSafe System One 相近，但使用自己的媒体字段。

仓库公告称 2026-10-13 发布了 1.5。本轮读取到 1.0 的 4B 模型卡，但浏览工具未能打开 1.5 的权重页；后续选型须另行核对 1.5 的下载可用性和具体 revision。训练数据公开量也不等于原始全部训练数据。

### decider-2b-vision

[模型卡](https://huggingface.co/Mapika/decider-2b-vision) 描述的视觉版基于 Qwen3.5-2B，并继承 decider-2b v5 的文本权重。它的文本能力不能直接套用较新纯文本版本的成绩。现有视觉评测包括视觉问答与游戏，标注为主要支持英语；这些结果未证明机械臂定位能力。

### d1 系列

[d1-3B](https://huggingface.co/LiquidAI/d1-3B) 提供 `system_one(..., images=...)` 接口。[600M 版](https://huggingface.co/LiquidAI/d1-omni-600M) 较小，但带图像时会把状态和问题文本截到 896 tokens，且是早期研究版本。长候选列表需要特别检查截断，不能把接口执行成功等同于完整阅读了输入。

### Imajev 与 PlayJev

[Imajev-4B](https://huggingface.co/mohit67890/imajev-4b) 公开的是适配器和决策头，需要加载对应基础模型。其 unknown 输出可用来研究拒答，但在业务图片上的校准不能直接迁移到机器人图像。

[PlayJev](https://github.com/OmniJev/PlayJev) 的权重针对十种游戏训练。它可以作为训练与闭环接口的参考，直接作为通用 Affordance 模型仍需要新证据。

## 初步选型顺序

建议先以 **OneJev-4B 与 decider-2b-vision** 构建两个开放权重基线，并保留 d1-3B 作为另一架构的对照。选择依据是接口和可研究性，不是已经测得的性能排名。

复现阶段记录模型 revision、推理代码 commit、图像分辨率、候选数、问题数、精度、设备以及显存峰值。比较延迟时还应包含图像预处理、候选生成、全部模型调用与热力图重建；发布方的单次前向速度不能直接当作整个任务延迟。
