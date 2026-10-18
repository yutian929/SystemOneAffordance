# 调研文档导航

整理基础概念、可用模型与实验记录。当前代码只保留 Dense Partition：统一输入图像，输出全覆盖、互不重叠的 Mask；使用方式见 [仓库 README](../README.md)。文档 04 和 05 第 1～12 节中的完整 D 流程为历史研究记录；最新架构见文档 05 第 13 节。

| 文档 | 回答的问题 |
| --- | --- |
| [01 Affordance 入门](01_affordance_primer.md) | Affordance 是什么，2D 预测输出什么，和物体分割有什么关系？ |
| [02 视觉 System One 模型](02_visual_system_one_models.md) | 哪些模型接受图像，输出什么，是否能本地部署？ |
| [03 System One 与 Affordance](03_system_one_affordance_research.md) | 已有人使用这些模型做 Affordance 吗？哪些工作最接近我们的分割任务？ |
| [04 最小 Affordance 决策测试](04_minimal_affordance_test.md) | 区域独立判断与点级前后景判断，怎样预测指令对应的多个有效区域？ |
| [05 方法 D：初版流程、结果诊断与改进](05_method_d_analysis_and_improvements.md) | D 的输入输出和中间变量是什么？初版如何逐步执行，已有结果支持哪些改进方向？ |

返回 [仓库首页](../README.md)。
