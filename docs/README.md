# 调研文档导航

整理基础概念、可用模型与实验方案。当前代码提供 Dense Partition 及 UMD 分区评测：统一输入图像，输出全覆盖、互不重叠的 Mask，并评价其表达功能部位的能力；架构、使用方式与验证记录见 [仓库 README](../README.md)。文档 04 保留历史 A、C、D 实验方案，其中的完整 Affordance 决策流程已从当前代码移除；深度调研中的方案与建议用于后续研究。

当前选型：**MoSA whole-image R50**。720 张人工标注的五轮完整实验、最终参数和取舍见文档 07；调参已按用户要求停止。

| 文档 | 回答的问题 |
| --- | --- |
| [00 2D Affordance 深度调研](00_SystemOne_2DAffordance_深度调研.md) | 任务范式、相关工作与数据集有哪些？如何分析区域决策方案的创新性并设计实验？ |
| [01 Affordance 入门](01_affordance_primer.md) | Affordance 是什么，2D 预测输出什么，和物体分割有什么关系？ |
| [02 视觉 System One 模型](02_visual_system_one_models.md) | 哪些模型接受图像，输出什么，是否能本地部署？ |
| [03 System One 与 Affordance](03_system_one_affordance_research.md) | 已有人使用这些模型做 Affordance 吗？哪些工作最接近我们的分割任务？ |
| [04 最小 Affordance 决策测试](04_minimal_affordance_test.md) | 历史实验方案：区域独立判断与点级前后景判断，怎样预测指令对应的多个有效区域？ |
| [05 无类别部件分割方法接入调研](05_class_agnostic_part_segmentation_methods.md) | HASSOD、MoSA、DiffCut、CAST 等是否满足单图无类别输入，怎样接入和评估多物体部件划分？ |
| [06 UMD 分割评价方案](06_umd_affordance_partition_benchmark.md) | 六种分割方法怎样比较？评测流程、输入输出、文件格式和指标分别是什么？ |
| [07 UMD 720 张评测报告](07_umd720_benchmark_report.md) | 为什么最终采用 MoSA？五轮实测结果、参数影响、质量与区域数/耗时的取舍是什么？ |

返回 [仓库首页](../README.md)。
