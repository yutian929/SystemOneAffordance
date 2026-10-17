# SystemOneAffordance

探索支持视觉输入的 System One 模型在 **2D Affordance（二维可供性）** 任务中的应用。

目前已整理基础概念和可用模型，并完成方法 A、D 的首轮调试实验，具体研究方向随后续讨论逐步整理。

## 阅读入口

1. [Affordance 入门](docs/01_affordance_primer.md)：理解任务、输入输出和机器人执行之间的关系。
2. [视觉 System One 模型](docs/02_visual_system_one_models.md)：了解可选模型、图像接口和开放范围。
3. [System One 与 Affordance 的已有工作](docs/03_system_one_affordance_research.md)：核实直接应用、相邻研究，以及与交互区域分割的差别。
4. [最小 Affordance 决策测试](docs/04_minimal_affordance_test.md)：比较区域独立判断与随机点前后景判断，两种方式均支持多个有效区域。
5. [方法 D：初版流程、结果诊断与改进](docs/05_method_d_analysis_and_improvements.md)：详细记录 D 的输入输出、中间变量、流程图、初版结果及待验证的改进方案。

资料来源随正文附链接，首轮核对日期为 2026-10-15。方法 A、D 已完成少量样本的调试运行；结果与后续验证计划见文档 05，尚未完成独立检查集评估。
