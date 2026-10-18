# 无类别部件分割方法接入调研

初次核查日期：2026-10-18。调研时 `DensePartition` 已有 SAM2.1、UnSAMv2+、Semantic-SAM、PartGLEE 和 SLIC。

**HASSOD 和 MoSA 整图分支都适合作为优先实验。** 两者都有无类别的整图候选 Mask 路径和公开的成品权重。若希望增加原理不同的基线，接入 DiffCut 或 CAST；HQ-SAM2 适合补充边界质量实验。以下保留接入前的技术分析，候选排序不代表本仓库图像上的性能排名。

后续进展：根据用户选择，已先接入 MoSA，方法名为 `mosa`，完成完整权重 SHA-256 校验及五张调试图的 GPU 推理。当前用法、过滤参数和验证记录见 [仓库 README](../README.md)；[五图结果](../outputs/mosa_debug/index.html) 可直接查看。其余候选仍处于调研阶段。

## 1. 我们需要的输入输出

### 1.1 无类别的含义

外部输入只有一张 RGB 图像，不提供物体名称、部件词表、任务文本、人工点框、物体 Mask、深度或其他视角。输出是匿名区域，不需要知道某一区域叫“刀柄”还是“杯把”。

算法内部自动生成网格点、超像素或物体候选，可以符合这个接口；需要串联两个模型时，应明确标记为组合方法。无类别并不要求训练过程完全无监督。关键是推理过程不能依靠预设物体类别选择对应的部件类别。

因此，“开放词表”不等于“无类别”，“输出时删掉类别名”也不等于消除了模型对类别的依赖。检测器输出名为 `instances` 的容器则不构成障碍：只要里面是无类别的 Mask，我们可以直接取用几何区域。

### 1.2 当前代码的实际契约

入口是 `DensePartition.create(method).segment(image)`，支持图像路径、PIL 图像和 `uint8[H,W,3]` RGB 数组。子类返回 `(labels, owners, metadata, raw)`，最终得到原图大小的 `uint32[H,W]` 标签图：ID 从 1 连续编号，每个区域四连通，每个像素恰好归属一个区域。[入口与子类](../dense_partition.py)、[结果校验](../common/result.py)

现有模型路线先输出 `bool[N,H,W]` 候选 Mask 和原生分数。统一后处理让面积较小的候选优先占有重叠像素，面积相同再比较质量分数；模型没有覆盖的像素保持 `owners=0`，随后也被转成连通区域。最终统一合并面积小于配置阈值的区域，当前默认阈值为 16 像素。[几何处理](../common/partition.py)、[配置](../config.py)

**最终 100% 覆盖是框架保证，不是模型已经找到了所有部件。** 一个模型即使没有找到杯把，后处理仍会产生完整的图像划分。因此需要同时查看原始候选和最终区域。

## 2. 候选与接入顺序

这里的“可接入”表示存在可用推理路径与公开资源，仍需要写适配器和验证依赖。

| 方法 | 无类别、单图自动推理 | 原始输出与多物体能力 | 资源及主要限制 | 建议 |
| --- | --- | --- | --- | --- |
| [HASSOD](https://github.com/Shengcao-Cao/HASSOD) | 是 | 整图 Mask 候选；预测 whole / part / subpart 层级 | 官方成品权重可访问；需适配 Detectron2、层级筛选和候选预算 | 优先，直接补充部件层级方法 |
| [MoSA 整图分支](https://github.com/360CVGroup/MoSA) | 是 | 单类别 Mask2Former 候选；整图覆盖多个物体及部件 | 成品权重可访问；2000 queries，需评估内存、重复候选和 CUDA 算子 | 优先，较新的独立方法 |
| [DiffCut](https://github.com/PaulCouairon/DiffCut) | 是，使用空文本提示 | 扩散特征递归划分，输出区域 ID 图 | SSD-1B 公开；细部受特征分辨率限制，需处理依赖和设备硬编码 | 第二批，增加稠密划分路线 |
| [CAST](https://github.com/twke18/CAST) | 是，超像素自动生成 | 多层区域 ID 图；对整张图分组 | 自监督权重可访问；原 Demo 中心裁剪，组数预算固定，DGL 依赖较旧 | 第二批，增加层次聚类路线 |
| [HQ-SAM2](https://github.com/SysCV/sam-hq/tree/main/sam-hq2) | 是，使用自动点网格 | SAM 风格多尺度候选 Mask | 权重可访问；改善边界不等于发现更多语义部件 | 补充边界对照 |
| [DiffSeg](https://github.com/google/diffseg) | 核心分割路径可以 | 注意力聚类生成整图区域 | 官方实现为 TensorFlow；可跳过附加的文本命名模块 | 备选，优先级低于 DiffCut |
| [OPS / OpenPartSeg](https://github.com/tydpan/OpenPartSeg) | 原 Demo 不满足 | 给定物体区域后，分割该物体的无类别部件 | 需要 RGB + 物体 Mask；未核实公开的训练完成权重 | 暂缓，可考虑组合方法 |
| [PartDistillation](https://github.com/facebookresearch/PartDistillation) | 第一阶段有无类别候选思路 | 最终完整流程还含物体类别内的部件组织 | 权重说明列的是训练初始化模型，未核实成品部件模型 | 暂缓 |
| [Real2Code 的 2D SAM 模块](https://github.com/MandiZhao/real2code/tree/main/part_segmentation) | 可改为自动点提示 | 针对运动学部件微调的 SAM | 公开说明要求自行微调；评测点由多视图几何产生 | 训练方向，暂非直接接入候选 |
| [SAM3 几何提示分支](https://github.com/facebookresearch/sam3) | 加自动点生成后可以 | 交互式 Mask；需自行包装整图候选生成 | 新环境与权重访问要求；未确认其部件收益 | 后续探索 |
| [SOHES](https://arxiv.org/abs/2404.12386)、[Falcon](https://openreview.net/forum?id=PvWHzAf9qp) | 方法方向符合 | 前者是层级实体分割，后者是无监督图划分 | 本次未找到可核实的完整公开代码与成品推理资源组合 | 跟踪资源发布 |

## 3. HASSOD 的接入价值

### 3.1 为什么符合要求

HASSOD 从自监督视觉特征生成层级伪标签，再训练检测和分割模型。部署时可以只使用训练好的检测器；无需重新执行伪标签生成和训练流程。它的层级是“整体、部件、子部件”，并不要求输入“刀、杯子、椅子”等类别。[论文](https://openreview.net/pdf?id=sqkGJjIRfG)、[复现说明](https://github.com/Shengcao-Cao/HASSOD/blob/main/reproduction.md)

源码的 `ROI_HEADS.NUM_CLASSES` 是 1，边框回归也设置为类别无关。官方 Demo 输入一张图像，支持 `whole / part / subpart` 层级，权重下载地址指向训练完成的模型。[模型配置](https://github.com/Shengcao-Cao/HASSOD/blob/main/detector_training/configs/cascade_mask_rcnn_R_50_FPN.yaml)、[Demo 与权重](https://github.com/Shengcao-Cao/HASSOD/blob/main/demo.md)

### 3.2 接入形式

建议方法名 `hassod`，直接使用现有候选 Mask 路线。取出 `instances.pred_masks`、`scores` 和 `pred_levels`，把层级作为元数据保存；`score` 保留原意，不改名为 `predicted_iou`。RGB/BGR 转换按上游 Predictor 的实际契约处理。

首轮比较两种设置：只保留 part + subpart，以及保留 all 层级。前者贴近部件目标，后者允许 whole 候选提供残余区域。两种都能交给当前的重叠消解流程，不必输出对象与部件的父子树。

一个容易漏掉的问题是：上游 `VisualizationDemo.run_on_image` 对层级的筛选仅作用于局部可视化变量，返回的 `predictions` 没有同步筛选。适配器必须自己按 `pred_levels` 过滤原始候选。[Demo Predictor 源码](https://github.com/Shengcao-Cao/HASSOD/blob/main/demo/predictor.py)

### 3.3 多物体与风险

它本身就是整图检测器，不需要先裁出一个物体。默认 `DETECTIONS_PER_IMAGE=100` 限制的是全部候选数量，包含整体与各层部件；多物体场景应检查是否频繁触顶，不能理解为“支持 100 个物体，每个还能输出无限部件”。

源码按检测类别执行 NMS，而检测类别只有一个。因此整体与部件候选之间可能发生抑制。先保留官方行为建立基线，再单独评估阈值或层级处理变体，并记录变更。[推理与 NMS](https://github.com/Shengcao-Cao/HASSOD/blob/main/detector_training/modeling/roi_heads/fast_rcnn.py)

适配的主要工作在 Detectron2 和 ROI 算子兼容性。层级预测直接契合任务，但不能据此保证细长刀柄、杯把或金属反光区域优于现有 PartGLEE。

## 4. MoSA 整图分支的接入价值

### 4.1 训练用视频，推理用单张图

MoSA 从视频运动产生多粒度伪标签，学习视觉分组，再把能力迁移到静态图像的整图和点提示模型。官方仓库标注 ECCV 2026，并公开了 PGM、整图模型和点提示模型。我们的直接候选是训练完成的 **whole-image segmentation** 分支。[论文](https://arxiv.org/abs/2609.39785)、[官方实现](https://github.com/360CVGroup/MoSA)

整图配置具有以下关键属性：`NUM_CLASSES=1`、`NUM_OBJECT_QUERIES=2000`、`INSTANCE_ON=True`、`SEMANTIC_ON=False`、`PANOPTIC_ON=False`。这里 `INSTANCE_ON` 选择的是输出独立候选 Mask 的代码路径，不代表需要输入实例或具体类别。[整图配置](https://github.com/360CVGroup/MoSA/blob/main/Adaption_to_Segment_Anything/whole_image_segmentation/configs/maskformer2_R50_bs16_8ep.yaml)

官方 Hugging Face 仓库包含 `whole_image_segmentation.pth`，文件端点可访问，无须为了得到这个分支重新训练运动模型。[模型文件](https://huggingface.co/qihoo360/MoSA/tree/main)

### 4.2 接入形式与多物体限制

建议方法名 `mosa`，以独立 worker 加载整图模型，输入原图并传递原始高宽，取 `instances.pred_masks` 和 `scores`。输出分数由候选分类置信度和 Mask 内平均概率结合得到，应保存为原生 `score`。[推理源码](https://github.com/360CVGroup/MoSA/blob/main/Adaption_to_Segment_Anything/whole_image_segmentation/mask2former/maskformer_model.py)

默认推理保留最多 1000 个候选，输入短边 1024、长边上限 2048。这个预算适合调查多物体与多部件召回，但大量低质量或重复小 Mask 可能让当前“小区域优先”的重叠规则产生碎片。首轮应报告质量过滤前后数量、空 Mask、重复率、峰值内存和最终碎片数量。[基础配置](https://github.com/360CVGroup/MoSA/blob/main/Adaption_to_Segment_Anything/whole_image_segmentation/configs/Base-COCO-InstanceSegmentation.yaml)

环境主要涉及 Detectron2 和 Mask2Former 的多尺度可变形注意力算子。现有框架已经处理过同类依赖，但仍需确认版本与权重能一起运行。

### 4.3 如何解读公开结果

官方报告的整图 Average Recall 在 PartImageNet 为 52.7、PACO 为 28.4，对照 UnSAM 为 51.6 和 27.5。这支持把它列为部件候选生成器进行比较；该表比较的是 **UnSAM，不是本仓库的 UnSAMv2+**，也不是本地工具图像的效果。[官方结果表](https://github.com/360CVGroup/MoSA#main-results)

同仓库的 PGM 也提供单图滑窗推理，会执行质量筛选、NMS 和 DenseCRF；可以另设 `mosa-pgm` 实验。它与最终整图模型是不同路径，不能混用名称或把两者结果放进同一配置。[PGM 推理](https://github.com/360CVGroup/MoSA/blob/main/PGM/inference.py)

## 5. DiffCut 与 CAST 提供不同的区域划分路线

### 5.1 DiffCut

DiffCut 使用扩散模型视觉特征和递归 normalized cut 生成区域。官方特征提取路径可使用空字符串提示，不需要物体或部件名称；示例使用 SSD-1B，已有公开的模型组件。[官方仓库](https://github.com/PaulCouairon/DiffCut)、[特征提取代码](https://github.com/PaulCouairon/DiffCut/blob/main/tools/ldm.py)、[SSD-1B](https://huggingface.co/segmind/SSD-1B)

`generate_masks` 返回整数区域图，没有每个部件的置信度。默认 1024 输入的聚类特征网格是 32×32，经过递归划分，再利用上采样特征分配高分辨率标签。区域数由切分停止条件控制，不要求事先指定物体或部件数。[核心算法](https://github.com/PaulCouairon/DiffCut/blob/main/diffcut/recursive_normalized_cut.py)

建议新增 `diffcut` 子类，直接走整数标签图路线。上游的零起始 ID 可以无损加 1，并记录 ID 偏移，然后调用 `connected_labels` 和公共小区域合并。不能用全零伪分数把它硬包装成有质量排序的检测器。

适配时要处理三件事：保留整张图并还原原始分辨率；把硬编码的 `cuda` 改为当前设备；为常量特征、极小图像和特征矩阵分解失败增加有效的整图回退。评测脚本里的真实标签匹配不能进入推理路径，应从示例 Notebook 的图像特征到区域生成步骤提取最小实现。[示例](https://github.com/PaulCouairon/DiffCut/blob/main/diffcut.ipynb)、[评测脚本](https://github.com/PaulCouairon/DiffCut/blob/main/eval_diffcut.py)

多物体不会造成接口限制，但全局聚类可能把相似材质的不同物体区域分到同一组，之后需要按连通性拆分。粗特征网格也可能忽略杯把或刀刃细节。它适合作为不同原理的基线，不能预先假定会自然分出人类定义的功能部件。

### 5.2 CAST

CAST 在图像表征学习中学习超像素的层次分组。可使用自监督权重，输入标准化 RGB 和自动生成的 SEEDS 超像素；不需要提供语义类别。虽然仓库也有结合开放词表模型的评测，这不属于核心分组的必要输入。[输入输出说明](https://github.com/twke18/CAST/blob/main/docs/OVERVIEW.md)、[PartImageNet 评测说明](https://github.com/twke18/CAST/blob/main/docs/GETTING_STARTED_PartImageNet.md)

普通 `forward` 返回图像表示。分割适配器应使用 `forward_features` 中的分组 logits，逐层把粗组索引映射回原始超像素，得到四层 ID 图；默认分组预算为 64、32、16、8。[模型源码](https://github.com/twke18/CAST/blob/main/cast_models/cast.py)、[层级可视化示例](https://github.com/twke18/CAST/blob/main/notebooks/visualize_hierarchical_segmentation.ipynb)

建议方法名 `cast`，首轮选最细层输出最终 partition，同时把粗层留作可选实验。它可以接收多物体整图，但分组预算作用于全图；不是每个物体分别拥有 64 个区域。连接分量拆分之后区域数还可能增加。

官方示例使用 `Resize(224)` 加 `CenterCrop(224)`，会裁掉非正方形图像边缘。接入必须明确选择保留完整视野的缩放或填充，移除填充并还原原图，记录相对官方 Demo 的预处理差异。初始超像素如果跨过真实部件边界，后续只做分组无法将它拆开。

官方 `environment.yaml` 使用 Python 3.9、PyTorch 1.13.1；另有较旧的 DGL 安装说明。当前环境不能直接照搬整套依赖，宜提取独立推理所需的模型、分组和超像素模块。[环境文件](https://github.com/twke18/CAST/blob/main/environment.yaml)、[独立使用说明与权重](https://github.com/twke18/CAST)

## 6. HQ-SAM2 的定位

HQ-SAM2 仓库提供 `SAM2AutomaticMaskGenerator`，可以自动产生网格点并返回 SAM 风格候选，与当前 proposal 适配接口接近。它还保留常见的 IoU 预测和稳定性过滤字段。[自动 Mask 生成器](https://github.com/SysCV/sam-hq/blob/main/sam-hq2/sam2/automatic_mask_generator.py)

HQ 解码分支结合额外特征改善 Mask 边界。对于细长手柄、孔洞和边缘的收益值得测试，但边界更准确与“把杯把从杯身分出来”是两种能力，不能互相替代。[HQ 解码器](https://github.com/SysCV/sam-hq/blob/main/sam-hq2/sam2/modeling/sam/mask_hq_decoder.py)

官方 HQ-SAM2 权重为 SAM2.1 Hiera-Large，当前主基线是 SAM2.1-Small。因此应增加普通 SAM2.1-Large、相同点网格及裁剪参数的对照，才能判断改善来自 HQ 分支还是更大的骨干。上游包名同样叫 `sam2`，需要 worker 内隔离导入，避免覆盖现有后端。[HQ-SAM2 说明与权重](https://github.com/SysCV/sam-hq/tree/main/sam-hq2)

## 7. 暂缓与不满足当前输入要求的方法

### 7.1 OPS 需要物体区域，仍可能作为组合方法

OPS 的部件目标是类别无关的，但官方 `predict_part.py` 读入 RGBA：RGB 是图像，alpha 通道是目标物体区域。它不是现成的任意多物体 RGB 整图自动分割器。[预测入口](https://github.com/tydpan/OpenPartSeg/blob/main/predict_part.py)

可研究“自动物体候选 → 对每个物体构造 alpha → OPS 预测部件 → 映射回整图”的组合。漏掉的物体仍会漏分；错误物体区域会传递给部件模型。把整张图的 alpha 全设为 1 不等于已经验证其适合多物体输入。

本次未在官方说明和发布页核实训练完成的 OPS 权重，仓库中也有询问 CKPT 的公开问题。找到并验证成品权重，是把它提升为直接接入候选的前提。[仓库](https://github.com/tydpan/OpenPartSeg)、[权重问题](https://github.com/tydpan/OpenPartSeg/issues/1)

### 7.2 PartDistillation 与 Real2Code

PartDistillation 的第一阶段有无类别部件候选模型，但完整流程还会组织物体类别内的部件。官方权重说明给出的是 COCO Mask2Former 和 Detic 等训练初始化权重，不能把它们当作训练完成的部件发现模型。本次未确认可直接下载的目标阶段成品权重。[方法说明](https://github.com/facebookresearch/PartDistillation)、[权重说明](https://github.com/facebookresearch/PartDistillation/blob/main/weights/README.md)

Real2Code 虽然整体是三维关节物体重建，其 2D 模块确实是从 RGB 预测部件的 SAM 微调模型，值得单独考虑。问题是公开说明要求用其数据自行微调，官方评测点又来自多视图几何流程；要成为当前框架的方法，还需取得成品权重并设计单图自动点生成。其运动学部件定义也不一定把刚性工具的刀柄和刀刃分开。[2D 模块说明](https://github.com/MandiZhao/real2code/blob/main/part_segmentation/README.md)

### 7.3 SAM3 可以研究几何分支，但不能直接照搬文本 Demo

SAM3 常见的 `set_text_prompt` 使用方式依赖概念文本。它也提供不依赖类别词的几何交互路径：`enable_inst_interactivity=True` 和 `predict_inst`。因此，可以自动生成点网格、逐点或批量预测，再增加质量筛选、裁剪与去重，构建满足单图无类别接口的候选生成器。[官方交互示例](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_for_sam1_task_example.ipynb)

该路径不能直接等同于官方完整的自动部件分割器。README 要求 Python 3.12+、PyTorch 2.7+ 和 CUDA 12.6+，权重有访问申请流程，与当前环境存在明显差异。SAM3.1 的视频改进也不能直接作为单图部件能力提升的证据。[官方说明](https://github.com/facebookresearch/sam3)

### 7.4 其他工作

| 工作 | 当前判断与依据 |
| --- | --- |
| [DiffSeg](https://github.com/google/diffseg) | 核心注意力合并不依赖语义类别，附加的 caption 命名步骤可以跳过。官方使用 TensorFlow，当前以 PyTorch 为主，先接 DiffCut 更便于共用环境；无须为纯分割照搬 BLIP 命名流程。 |
| [SOHES](https://arxiv.org/abs/2404.12386) | 方法目标非常合适，涵盖实体及组成部件。但论文链接的项目页在核查时不可访问，未找到可验证的公开成品代码和权重组合；不能列为已具备复现条件。 |
| [Falcon](https://openreview.net/forum?id=PvWHzAf9qp) | 无监督图划分方向有潜力。论文关联的 Falcon-Seg 仓库在核查时返回 404，先跟踪代码资源；这不等于断言作者永远没有发布。 |
| [TAPPS](https://github.com/tue-mps/tapps) | 部件推理依赖固定语义类别及物体到部件映射。把结果中的类别 ID 丢掉，不能改成我们的无类别部件发现。 |
| [LangHOPS](https://github.com/y9miao/LangHOPS-release) | 输入候选物体、部件概念及语言层级，属于开放词表设置。已有更强的语义分割结果也不代表满足无类别接口。 |
| [PartCATSeg](https://github.com/kaist-cvml/part-catseg) | 开放词表图文对应路线，不能直接当作不需要候选词表的部件发现器。 |
| [SAMPart3D](https://github.com/Pointcept/SAMPart3D)、[PartField](https://research.nvidia.com/labs/toronto-ai/partfield-release/) | 原生任务以三维形状、点云或由三维对象生成的视图为基础，不能直接接收我们只有一张 RGB 的输入。 |

## 8. 框架需要怎样适配

### 8.1 候选 Mask 后端

HASSOD、MoSA 和 HQ-SAM2 可以沿用现有路线：

```text
RGB → 模型 worker → bool[N,H,W] + 原生分数
    → resolve_masks → 连通区域 → 小区域合并 → PartitionResult
```

新增方法注册、配置校验、来源及权重校验、worker 分发和模型适配模块即可。保留原始候选以及层级、裁剪、质量阈值等必要元数据，便于区分模型的漏分与后处理影响。[模型 worker](../common/models.py)、[来源配置](../common/settings.py)

### 8.2 稠密标签后端

DiffCut 和 CAST 更适合：

```text
RGB → 特征提取和分组 → integer[H,W]
    → 无损正整数重编号 → 连通区域 → 小区域合并 → PartitionResult
```

当前 `PartitionResult` 已接受二维整数原始分割，但要求 ID 为正数；`export_raw_segmentation` 则把所有二维输出写成 `kind='slic_labels'`，错误信息也特指 SLIC。接入新的稠密标签方法，应把这里的命名泛化，并检查展示页和相关校验，不能让 DiffCut 或 CAST 的原始结果被标成 SLIC。零起始 ID 的偏移要写入元数据，保留可逆关系。[结果类型](../common/result.py)、[原始结果导出](../common/io.py)

### 8.3 子进程不等于独立 Python 环境

当前 worker 用 `sys.executable` 启动，仍与主程序使用同一解释器。若旧依赖无法兼容，需要增加显式的 worker 解释器配置及相应验证，或将最小推理路径移植到现有环境；目前不能宣称已经支持任意 Conda 环境。[进程启动代码](../common/models.py)

本机当前主环境为 Python 3.10、PyTorch 2.6 + CUDA 12.4。版本声明相容也不能替代实际加载、算子调用和图像推理测试。

## 9. 多物体场景的比较方案

先使用现有 knife、mug、spoon、hammer、clutter 五张调试图，验证接入、导出和可视化；它们只适合冒烟检查。再准备覆盖遮挡、接触物体、重复同类物体、细长结构、弱纹理和反光表面的测试图，并将调参图与最终评测图分开。

评价部件时允许人工标注匿名部件 Mask，不需要建立类别词表；这些标注只用于离线评估，不能作为推理输入。

| 检查层次 | 应记录的结果 | 目的 |
| --- | --- | --- |
| 输入输出契约 | 原图尺寸、正连续 ID、四连通、每像素唯一归属、导出回读一致 | 保证新方法能在现有框架使用 |
| 原始模型输出 | 候选数、覆盖率、重叠率、空候选、分数分布、是否达到 top-k 上限 | 判断模型是否真正发现部件 |
| 部件召回 | 每个匿名真实部件的最佳候选 IoU；IoU 0.5 / 0.75 召回率 | 检查杯把、刀柄等是否存在于候选中 |
| 最终区域 | 边界准确度、跨部件错误合并、一个部件被切成多少碎片 | 评估重叠消解和小区域合并的影响 |
| 下游可用性 | 在固定可选区域数量下，用区域并集表示目标部件的最佳 IoU | 区分允许组合区域时是否仍可恢复目标 |
| 资源开销 | 加载时间、纯推理时间、worker 总时间、显存、CPU 内存、导出大小 | 避免只比较一次 forward 的速度 |

候选召回应比较多个相近候选预算，最终 partition 应展示不同区域数下的质量，避免靠无限碎分提高分数。区域并集的最佳 IoU 是使用真实标注计算的离线能力上限，不是实际自动选择效果。

所有方法使用一致的公共小区域处理；另外保留关闭该步骤的诊断结果，检查 16 像素阈值是否吞掉细小部件。HQ-SAM2 加同骨干对照；CAST 的层级、DiffCut 的停止阈值、HASSOD 的层级筛选、MoSA 的质量和候选数量设置分别记录，不在测试集上逐张手工挑最优值。

## 10. 资源核查记录

下表仅表示资源可获取性。HTTP Range 检查读取了文件前 32 字节与总长度，没有完整下载、计算权重 SHA-256 或加载模型；文件大小也不是推理显存需求。

| 资源 | 核查结果 |
| --- | --- |
| [HASSOD 官方成品权重](https://drive.google.com/file/d/1pZ1yP3cs-Ezrw3OVJxi4arUFkxQ8W5fo/view) | HTTP 206；总长 287,504,069 字节 |
| [MoSA whole-image 权重](https://huggingface.co/qihoo360/MoSA/blob/main/whole_image_segmentation.pth) | HTTP 206；总长 180,338,134 字节；HF 仓库非 gated |
| [HQ-SAM2 Hiera-Large 权重](https://huggingface.co/lkeab/hq-sam/blob/main/sam2.1_hq_hiera_large.pt) | HTTP 206；总长 898,844,313 字节 |
| [CAST-S 自监督权重](https://huggingface.co/twke/CAST/blob/main/snapshots/moco/imagenet1k/cast_small/checkpoint_0099.pth.tar) | HTTP 206；总长 767,156,042 字节 |
| [DiffCut 使用的 SSD-1B](https://huggingface.co/segmind/SSD-1B/tree/main) | HF API 可访问，非 gated，列有 UNet、VAE 和文本编码器等组件 |

代码核查版本如下；实际接入时应固定版本并另行记录完整权重校验值。

| 仓库 | 核查的 Git revision |
| --- | --- |
| HASSOD | `215d90a4acbd72876e20c4abfcde9f578e7e6a4c` |
| MoSA | `7909eed2516607b40653fa5cd88ba23147131850` |
| DiffCut | `aa16df45619515abe1d7a728ce7927ad1019f950` |
| CAST | `ec85adff6fae5802684e21423112cab55fc732ae` |
| sam-hq | `e696978d60352dc9a26b12631cd91781502c6546` |
| OpenPartSeg | `671e269390d921f9e77c60d82f56e942b2f708b0` |
| sam3 | `7f473cfd983d5eb43040ff608f66bff0d7bdcb53` |
| DiffSeg | `77ea59657da1be832c89e48085e04b97952e2a36` |
| PartDistillation | `a0bb45947115443aec4fe09f0bc3c032a008743d` |

返回 [调研文档导航](README.md)。
