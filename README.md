# SystemOneAffordance

当前实现 **Dense Partition / Exhaustive Segmentation（全覆盖分割）** 及其 **UMD Affordance 分区评测**，作为后续 Affordance 研究的基础。输入一张 RGB 图片，输出一组 Mask，严格满足：**每个像素恰好属于一个 Mask**。目前支持 SAM 2.1 small、UnSAMv2+、Semantic-SAM、PartGLEE RN50、MoSA R50 和 SLIC，统一通过 `DensePartition` 接口调用。评测器使用 UMD 标签评价这些区域的表达能力，尚未接入任务决策模型。

最终采用 **MoSA whole-image R50**，Python API 默认使用第 5 轮选定的配置：`max_masks=250`、`score_thresh=0.25`、`mask_nms_thresh=0.95`。720 张人工标注的调参已停止，共完成五轮完整比较；结果与选择理由见 [07：UMD 720 张评测报告](docs/07_umd720_benchmark_report.md)。命令行比较入口不指定 `--method` 时仍运行全部六种方法；只运行最终方案可用 `python dense_partition.py --method mosa`。

## 代码结构

```text
dense_partition.py     # 抽象基类、六种实现、统一工厂及命令行入口
affordance_benchmark.py # 评测基类、UMD 实现、指标、CSV/页面导出及命令行入口
config.py              # 日常调节项、设备、debug 图片
common/
  partition.py         # 解重叠、补齐区域、连通域、小区域合并、验证
  settings.py          # 模型资源、官方 Demo 内部设置与参数校验
  result.py            # PartitionResult：标签图、Mask 访问与保存
  io.py                # 图像读取、哈希、JSON、无损 Mask 导出与读取
  models.py            # 各模型的资源检查、推理、尺寸还原与运行兼容
  partglee.py          # PartGLEE 资源检查、多物体/部件候选、过滤和 worker 适配
  mosa.py              # MoSA 整图无类别候选、分批尺寸还原、过滤和 worker 适配
  visualization.py     # 边界图、总览、交互查看页面
scripts/
  download_sam2.py      # 仅下载 SAM 2.1
  download_unsamv2.py   # 仅下载 UnSAMv2+ 权重及固定版本源码
  download_semantic_sam.py # Semantic-SAM 权重、固定源码及 Detectron2 编译
  download_partglee.py # PartGLEE RN50、CLIP 权重、固定源码及 Detectron2 编译
  download_mosa.py     # MoSA 整图 R50 权重、固定源码及 Detectron2 0.6 编译
tests/test_dense_partition.py
tests/test_semantic_sam.py
tests/test_small_regions.py
tests/test_partglee.py
tests/test_mosa.py
tests/test_affordance_benchmark.py
docs/
environment.yml
requirements.txt
requirements-sam.txt
requirements-semantic-sam.txt
requirements-partglee.txt
requirements-mosa.txt
```

全覆盖分割功能的命令行逻辑直接写在 `dense_partition.py` 的 `if __name__ == '__main__':` 中，负责参数解析、单图或批量运行及结果导出。

所有学习模型由 `common/models.py` 统一通过当前 Conda Python 启动独立子进程。每个模型只加载一次，在同一个进程中连续处理所有图片，随后退出释放显存，再启动下一个模型。PartGLEE 和 MoSA 的适配实现分别位于 `common/partglee.py`、`common/mosa.py`；Semantic-SAM 的尺寸还原和兼容处理在 `common/models.py` 内完成。本项目使用 `common` 包名，与上游包可以同时导入。

旧 `configs/` 目录、完整 D 流程、Valen 判断、任务指令和旧评估代码已经移除。当前 benchmark 单独实现分区评测及代表点命中统计。历史实验保留在 `docs/` 和本地 `outputs/`；旧实现可从 Git 提交 `3d62972` 查看。数据、模型、输出不纳入 Git。

## Miniconda 环境

本机环境已经建立，日常使用：

```bash
cd /data0/yutian/SystemOneAffordance
source /home/hri/miniconda3/etc/profile.d/conda.sh
conda activate systemone-affordance
unset PYTHONPATH
```

新机器调整 Miniconda 路径后创建环境：

```bash
conda env create -f environment.yml
conda activate systemone-affordance
unset PYTHONPATH
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements-sam.txt
python -m pip check
```

环境使用 Python 3.10、PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、scikit-image `0.25.2`。SAM 源码版本固定在 `requirements-sam.txt`，在 PyTorch 安装后单独安装；关闭可选 CUDA 扩展，当前默认参数不依赖该扩展。GPU 推理需要兼容 CUDA 12.4 的 NVIDIA 驱动。

`environment.yml` 设置 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`，避免其他项目的用户包干扰和仓库内字节码缓存。Conda 环境、Hugging Face 全局缓存及 UnSAMv2 源码放在仓库外。基础依赖不包含 Transformers 和视频解码依赖；Semantic-SAM 的增量依赖单独列出。运行 SLIC 不导入 PyTorch，也不需要模型文件。

Semantic-SAM 也使用同一个 `systemone-affordance` Conda 环境。激活后安装增量依赖并准备模型：

```bash
python -m pip install -r requirements-semantic-sam.txt
python scripts/download_semantic_sam.py
python -m pip check
```

保留当前 PyTorch `2.6.0+cu124`、torchvision、NumPy 和 Pillow，不执行上游面向旧环境的整份依赖安装。准备脚本需要 `git` 和 C++ 编译器，固定并验证 Semantic-SAM / Detectron2 源码，校验 Swin-T 权重 SHA-256，在 `models/Semantic-SAM/` 中构建 Detectron2 的 CPU 扩展以满足导入要求。推理使用 CUDA，deformable attention 使用上游提供的 PyTorch 参考实现，适配当前 PyTorch；这些兼容处理只在独立进程内生效。运行时不下载额外权重。

PartGLEE 使用同一环境，安装增量依赖后运行准备脚本：

```bash
python -m pip install -r requirements-partglee.txt
python scripts/download_partglee.py
python -m pip check
```

源码固定为官方提交 `8c6b83e0be076e8fe640943096059ee5c180de0e`，保存在 `models/PartGLEE/source/`；官方 RN50 权重保存在 `models/PartGLEE/RN50/`。脚本校验 RN50 和 CLIP 权重的 SHA-256，CLIP 来自官方安装文档指定的 GLEE Demo，固定其资源版本。单独编译这份源码的 Detectron2 CPU 扩展，推理和 PyTorch 参考 attention 在 CUDA 上执行。worker 只加载 RN50 推理所需模块和上游类别常量，避免训练/评估初始化、重复注册和无关骨干依赖；Transformers 5 的 tokenizer 接口兼容也仅在 worker 中生效。推理权重严格匹配，忽略的仅为官方训练权重中未参与推理的文本蒸馏教师。

MoSA 使用同一环境，准备整图模型：

```bash
python -m pip install -r requirements-mosa.txt
python scripts/download_mosa.py
python -m pip check
```

源码固定为官方提交 `7909eed2516607b40653fa5cd88ba23147131850`；整图 R50 权重来自 [官方 Hugging Face 仓库](https://huggingface.co/qihoo360/MoSA)，固定资源版本 `c915f4f7020c88b8a1e09095ab89396216c96bf5` 并校验 SHA-256。资源保存在 `models/MoSA/`。下载脚本单独准备官方 Detectron2 0.6 的 CPU 导入扩展，MoSA 推理使用 CUDA 和上游 PyTorch 参考 attention；不覆盖其他方法的 Detectron2。依赖版本与现有环境共用，推理不需要 PGM、视频、类别词表或训练数据，也不自动下载骨干权重。

## 模型与参数

只用 SLIC 可以跳过下载；运行学习模型前执行：

```bash
python scripts/download_sam2.py
python scripts/download_unsamv2.py
```

权重分别保存在 `models/SAM2.1-small/` 和 `models/UnSAMv2-plus/`。下载脚本校验固定版本与权重 SHA-256。UnSAMv2 源码默认放在 `~/.cache/systemone-affordance/UnSAMv2-f4ec1136/`，也可设置仓库外的 `UNSAMV2_CODE_DIR`。推理读取本地资源，不自动下载。

日常只修改 [config.py](config.py) 中的少量参数：

| 配置 | 用途（当前值以 `config.py` 为准） |
| --- | --- |
| `DEFAULT_METHOD` | Python API 默认 `mosa`；命令行未指定 `--method` 时运行全部方法 |
| `DEVICE` | 当前 `cuda:1`，单卡机器改为 `cuda:0`。SLIC 始终使用 CPU |
| `MIN_MASK_AREA` | 所有方法统一的后处理阈值，默认 `16`：面积严格小于 16 个原图像素的区域并入邻居；`0` 关闭 |
| `UNSAMV2_PLUS.granularity` | 范围 `[0.1, 1.0]`，默认 `0.2`；越小倾向越细，但过小可能漏掉物体/部件，并非越小越好；HF Demo 初始值为 `0.15` |
| `UNSAMV2_PLUS.pred_iou_thresh` | 对应 Predicted IoU threshold；降低保留更多候选，HF Demo 默认 `0.77` |
| `UNSAMV2_PLUS.stability_score_thresh` | 对应 Stability threshold；降低筛选更宽松，HF Demo 默认 `0.9` |
| `SEMANTIC_SAM.level` | 唯一日常控制项：`[1]` / `[2]` 语义级，`[3]` 实例级，`[4]` / `[5]` / `[6]` 不同部件粒度；默认 `[1, 2, 3, 4, 5, 6]` 保留全部层级 |
| `PARTGLEE.max_object_masks` / `max_part_masks` | 全图候选输出上限，默认 `50` / `500`；不改变训练时的 query 数，不代表真实物体或部件数量 |
| `PARTGLEE.score_thresh` | 默认 `0.25`；过滤类别概率 × Mask 内平均概率低于阈值的候选 |
| `PARTGLEE.mask_nms_thresh` | 默认 `0.95`；同层级 Mask 的 IoU 严格超过阈值时保留高分候选，物体与部件不互相抑制 |
| `MOSA.max_masks` | 默认 `250`，按前景分类概率选取整图 top-k；范围 `[1, 2000]`，不改变网络训练时的 2000 个 query |
| `MOSA.score_thresh` | 默认 `0.25`，过滤前景 softmax 概率 × Mask 内平均概率低于阈值的候选；`0` 关闭分数过滤 |
| `MOSA.mask_nms_thresh` | 默认 `0.95`，按 Mask IoU 去除高度重复候选；`1` 关闭，显著更小的嵌套部件保留 |
| `SLIC` | `n_segments` 越大越细；`compactness` 控制形状规则性，`sigma` 控制平滑；目标数不等于最终 Mask 数 |
| `DEBUG_IMAGES` | 5 张不同的 debug 图片，只有图片路径，不包含任务指令或标注 |
| `OUTPUT_DIR` | 两个脚本共用的输出根目录，默认 `outputs/`；每次自动创建“脚本名_时间戳”子目录 |

SAM 2.1 直接使用官方自动生成器默认值。UnSAMv2+ 独立使用上述三个控制项，其余采样设置参考 [官方 HF Demo 源码](https://huggingface.co/spaces/yujunwei04/UnSAMv2/blob/675a2bceabf284b69d5d537a3aa798b53c50a8ff/app.py)，不再与 SAM 共用一组生成参数。模型资源与内部设置在 `common/settings.py`；`model_config='configs/…yaml'` 是第三方包内部资源路径。

Semantic-SAM 使用官方 Swin-T 权重和 `SemanticSamAutomaticMaskGenerator`，只传入 `level`，其他生成参数采用官方默认值。按官方 `prepare_image` 将短边缩放到 640，生成候选后使用最近邻还原至原图分辨率，再进入统一分区流程。`level` 的含义见 [官方用法](https://github.com/UX-Decoder/Semantic-SAM#-a-few-lines-to-get-generated-results)；它是离散层级列表，可填写 `[3]`、`[6]` 或组合，不能直接等同于目标 Mask 数。公开权重只在 SA-1B 上训练，这里输出多粒度候选，不输出 affordance 或语义类别标签。

PartGLEE 使用 `sa1b_joint` 分支和固定的 `['object', 'part']` 词表，调用时只需图片。整图缩放至最长边 1024，网络保留 300 个物体 query，从前 50 个派生 50×10 个部件 query；适配层将官方可视化用的 1 个物体、全图 10 个部件截断改为可配置的 50/500 上限。每个候选只取有效类别一次，绕过上游按 query×类别展平的最终 top-k，避免重复候选和零分填充。Mask logits 按上游方式双线性还原至原图，再以 0 为阈值二值化；过滤空 Mask、低分和同层级高度重复候选后进入统一分区。`raw_proposals` 保留 `kind`、`score`、`class_score`、`mask_score`、原图 `area` / `bbox` 和筛选前 `source_index`；不伪造 SAM 的 IoU / 稳定性分数，也不声称这些索引提供物体—部件归属树。`backend.filtering` 记录筛选前后统计。

MoSA 使用 [官方 whole-image R50 分支](https://github.com/360CVGroup/MoSA/tree/7909eed2516607b40653fa5cd88ba23147131850/Adaption_to_Segment_Anything/whole_image_segmentation)，输入只有 RGB。网络有一个前景类别和一个 no-object 输出，所有 2000 个 query 联合处理整张图片，不要求物体名称或物体 Mask。保持官方短边 1024、长边上限 2048 的缩放，按前景 softmax 概率选取 top-k，再分批将 logits 还原至填充尺寸、去除填充、双线性还原到原图，以 0 为阈值二值化。分批只降低尺寸还原阶段的显存占用，不改变网络结构、候选选择和分数定义。权重以 `strict=True` 加载，避免静默漏载。

MoSA 的 `score_thresh=0.25` 和 `mask_nms_thresh=0.95` 是本框架增加的候选过滤，官方 instance 路径没有这两步。需要比较官方候选时设为 `0` / `1`，仅移除空 Mask。`raw_proposals` 保存原生 `score`、`class_score`、`mask_score`、`query_index`、筛选前 `source_index`、原图 `area` / `bbox`，不输出语义类别或虚构物体—部件归属。`backend.filtering` 记录空候选、低分和重复候选的数量；其他元数据包括实际源码、输入/填充尺寸、加载与推理时间、过滤时间和峰值 CUDA 张量内存。默认整图推理对显存需求较高，若有必要可在配置副本中降低 `preprocess.short_edge` / `max_size`，并保留参数快照进行比较。

最终分区先消除重叠、补齐未覆盖区域并拆分连通域，再按 `MIN_MASK_AREA` 合并小区域。合并在 `common/partition.py` 的 `merge_small_regions()` 中实现，由 `DensePartition.segment()` 对六种方法统一调用：从最小区域开始，并入共享边界最长的上下左右相邻区域；共享边界相同时优先面积更大的邻居，再按区域 ID 决定。每次合并后更新面积和邻接关系，直到所有区域达到阈值或整图仅剩一个区域；若整张图片面积就小于阈值，保留这一个区域。阈值 `16` 用于清除少量像素的碎片，面积恰好 16 的区域保留；设置为 `0` 恢复不合并的行为。

Raw 候选与原始覆盖率不受此合并影响。HF Demo 最多展示前 1000 个原始候选，本地保留全部候选后建立严格分区，因此最终 Mask 数及展示方式不等同于 Demo。

## Python 使用

```python
from dense_partition import DensePartition

# 也可选 "sam2"、"unsamv2-plus"、"semantic-sam"、"partglee"、"mosa"
with DensePartition.create("slic") as segmenter:
    result = segmenter.segment("data/umd/tools/knife_01/knife_01_00000122_rgb.jpg")
    # 可以继续调用 segmenter.segment(下一张图片)，模型不会重新加载。
# 离开 with 后模型显存已释放，result 仍可使用。

labels = result.labels          # uint32[H, W]，ID 为 1…K，保留原图分辨率
mask = result.mask(1)            # bool[H, W]
print(result.region_count)      # K
for region_id, mask in result.iter_masks():
    pass
result.save("outputs/my_partition")  # 必须使用新目录
```

输入也支持 PIL 图像或 `uint8[H,W,3]` RGB 数组。`result.masks` 可一次生成 `bool[K,H,W]`，大图建议逐个读取以节约内存。`result.rgb` 和 `result.labels` 默认只读；`result.metadata` 记录配置、原模型覆盖率与耗时。`result.owners` 为原始模型候选索引加一，0 表示原模型未覆盖；SLIC 的该字段为 `None`。`owners` 始终记录合并前的像素来源，最终标签会拆分连通域、合并小区域并重新编号，因此一个最终 Mask 可以包含多个来源。

`start()` 显式初始化并返回 `startup_seconds`（资源检查、进程启动、导入、模型加载的总时间）与 `model_load_seconds`（其中模型构建与权重加载的时间）。不调用 `start()` 时，第一次 `segment()` 会自动初始化；后续调用复用模型。用 `with` 或 `try/finally` 中的 `close()` 释放资源。`metadata.segment_seconds` 不含初始化；初始化信息单独保存在 `metadata.initialization`，是同一个模型会话的记录，不应按图片累加。接口按顺序处理图片，不支持同一实例的并发调用。

需要在调用时改变参数，可先取得独立配置；修改不会影响其他实例：

```python
from config import get_config
from dense_partition import DensePartition

settings = get_config("slic")
settings["algorithm"]["n_segments"] = 600
result = DensePartition.create("slic", config=settings).segment("your_image.jpg")
```

`DensePartition` 是抽象基类；`SAM21Partition`、`UnSAMv2PlusPartition`、`SemanticSAMPartition`、`PartGLEEPartition`、`MoSAPartition`、`SLICPartition` 均位于 `dense_partition.py`。新增方法实现 `_segment_rgb()`，由基类 `segment()` 统一读取图片、记录配置并验证输出契约。

## 命令行与可视化

```bash
# 默认使用 config.DEBUG_IMAGES，对比全部六种方法
python dense_partition.py

# 指定单图，默认对比全部六种方法
python dense_partition.py --image your_image.jpg

# 只运行 Semantic-SAM，粒度在 config.py 的 SEMANTIC_SAM.level 中调节
python dense_partition.py --method semantic-sam

# PartGLEE：整图多物体和部件候选，无需提供类别或指令
python dense_partition.py --method partglee

# MoSA：单张 RGB 自动产生整图无类别候选，支持多物体
python dense_partition.py --method mosa

# 只运行 SLIC；也支持 --image 后面指定多张图片
python dense_partition.py --method slic
```

不指定 `--image` 时默认读取 `config.DEBUG_IMAGES`；不指定 `--method` 时运行全部方法（SAM 2.1、UnSAMv2+、Semantic-SAM、PartGLEE、MoSA、SLIC）。`--debug` 仍可显式使用，与 `--image` 互斥。

不传 `--output` 时，自动创建 `outputs/dense_partition_年月日_时分秒_微秒/`，连续运行会产生不同目录。`--output` 仅在需要自定义路径时使用，指定目录必须尚不存在。打开输出目录中的 `index.html`，可切换图片、查看原尺寸，点击像素查看各方法对应的最终 Mask，并下载二值 PNG。

页面每个方法占一行，固定按 **Original → Raw colors → Boundaries → Colors → Overlay** 展示：原图、原始候选纯色图、最终分区边界图、最终分区纯色图、最终分区叠加图。每行只保留 Raw / Final Mask 数量和选中区域信息；详细统计在 `summary.json` 中。`overview.jpg` 按图片、方法逐行排列，每行使用相同的五列顺序；单个方法的 `comparison.jpg` 也采用此顺序。

Raw 对 SAM 系列指自动生成器 `generate()` 返回的全部候选；对 PartGLEE 指适配层完成分数过滤和同层级去重后的候选；对 MoSA 指适配层过滤空 Mask、低分和高度重复候选后的结果。它们均尚未经过分区后处理；SLIC 则展示连通域重编号前的直接输出。Raw colors 中黑色表示没有候选覆盖，重叠处优先展示小 Mask。额外的覆盖情况图仍保存在 `raw_coverage.png`（粉色为未覆盖、橙色为重叠）和 `coverage.png`（粉色为补齐区域来源）中。

原始模型 Mask 另外无损保存在 `raw_masks.npz`，保留候选顺序、重叠和空白；`pixel_owners.npy` 仅记录消除重叠后的归属，不能代替原始候选。历史输出没有保留这些候选时，需要重新运行才能生成前后对照。

已生成全部 5 张 debug 图片的三方法 [前后对照页面](outputs/partition_before_after/index.html) / [总览图](outputs/partition_before_after/overview.jpg)，本次参数见 [summary.json](outputs/partition_before_after/summary.json)。

`summary.json` 开头的 `config` 按方法保存本次运行的参数快照。完成模型推理后，`generator` 会包含官方默认值和本次覆盖值；UnSAMv2+ 的粒度位于 `model.granularity`，两个过滤阈值位于 `generator`。以后修改 `config.py` 不会改变已有结果记录的参数。

小区域合并阈值保存在 `config.postprocess.min_mask_area`；每个分区的 `metadata.json` 中，`postprocess` 记录阈值、合并前后区域数及合并次数。历史输出不会随阈值修改而更新。

Semantic-SAM 的层级记录在 `generator.level`；`backend` 保存源码版本、实际 Python、输入缩放尺寸及 attention 实现。其原始候选以原图尺寸保存，`raw_proposals` 中的 `area` / `bbox` / `point_coords` / `crop_box` 也转换至原图坐标；`inference_geometry` 另外保留上游推理尺寸的记录，其中提示点是上游的归一化坐标。

```text
summary.json                         # 参数、统计和 completed 状态
index.html / overview.jpg
partitions/<图像内容哈希>/<方法>/
  labels.npy                         # uint32[H,W]，最终标签
  masks.npz                          # 无损位压缩 bool Mask 集合
  raw_masks.npz                      # 仅模型：原始候选，格式同 masks.npz，允许重叠/遗漏
  raw_labels.npy                     # 仅 SLIC：算法直接输出的标签
  metadata.json                      # 区域面积、来源、配置和导出验证结果
  pixel_owners.npy                    # 仅模型方法：原始候选的像素归属
  original.png / ids.png
  boundaries.png / overlay.png / colors.png / coverage.png
  raw_boundaries.png / raw_overlay.png / raw_colors.png / raw_coverage.png
  comparison.jpg                     # Original / Raw colors / Boundaries / Colors / Overlay
```

读取已有导出：

```python
import numpy as np
from common.io import iter_masks

labels = np.load("<分区目录>/labels.npy", allow_pickle=False)
for region_id, mask in iter_masks("<分区目录>/masks.npz"):
    assert np.array_equal(mask, labels == region_id)

# 模型原始候选也可用 iter_masks 读取；ID 对应 metadata.json 的 raw_proposals[id - 1]。
for proposal_id, mask in iter_masks("<模型分区目录>/raw_masks.npz"):
    print(proposal_id, mask.sum())
```

所有模型的候选先按小区域优先消除重叠，面积相同则比较后端原生质量分数（SAM 系列为预测 IoU、稳定性；PartGLEE 和 MoSA 为原生候选分数），再将所有区域（包括未覆盖像素）拆成连通域。两个脚本都按“模型 → 全部图片”运行，模型分别使用当前 Conda 的 Python 在独立进程中驻留；同一时间只保留当前模型，正常完成、异常和中断都会清理进程。SLIC 直接生成分区，随后与模型方法一样执行小区域合并。基类检查正整数连续编号、原分辨率和 4 连通性；导出时再读取二值 Mask，逐像素检查 `sum(masks[:, y, x]) == 1`。

全覆盖是输出结构保证，不能作为分割准确率；模型漏覆盖区域补齐后也不代表模型识别出了这些区域。当前不进行模型与 SLIC 求交细分，六种方法各自独立输出。

## UMD Affordance 分区评测

激活同一个 Conda 环境后运行：

```bash
# 默认五张 debug 图、全部六种方法；自动创建带时间戳的输出目录
python affordance_benchmark.py

# 指定 UMD 图片，自动配对同目录的 *_label.mat，默认全部方法
python affordance_benchmark.py --image data/umd/tools/knife_01/knife_01_00000122_rgb.jpg

# 只用 SLIC 快速检查，仍自动创建输出目录
python affordance_benchmark.py --method slic

# 按固定清单运行正式测试；清单需要人工标注身份和核实依据
python affordance_benchmark.py --manifest benchmarks/umd/manifest.json --split test
```

`AffordanceBenchmark` 是抽象基类，`UMDBenchmark` 读取 UMD `gt_label`。评测逻辑集中在 `affordance_benchmark.py`，六种后端继续复用 `DensePartition`。Python 可调用 `benchmark = AffordanceBenchmark.create('umd')`，然后调用 `benchmark.evaluate(labels, gt)` 评估已有分区，或 `benchmark.run(manifest, output, methods=None)` 执行完整流程；`methods=None` 默认全部方法。

五项指标为最优区域组合的 Oracle IoU、IoU ≥ 0.8 的比例、最终区域数 K、代表点命中率和每图分割耗时。多个碎片可以合并成一个功能部位；分区和代表点生成都不读取 GT。`-1` 像素按本项目约定忽略，背景参与误选惩罚，不存在的功能类别跳过。Tools 和 Clutter 分开统计，按帧、实例/场景、有效功能类别分层平均。

结果自动保存在 `outputs/affordance_benchmark_年月日_时分秒_微秒/`，无需传入 `--output`：`index.html` 可切换图片和功能、查看原尺寸与代表点；`summary.csv`、`per_class.csv`、`per_sequence.csv`、`scores.csv`、`image_metrics.csv` 分别保存汇总与明细；`run.json` 保存实际配置、版本、耗时口径和失败记录；`manifest.json` 保存输入路径、哈希和标签信息。普通模式逐图保存原始候选、最终分区和各功能的 GT/最优组合/误差图。

大批量评测使用 `--streaming`：每张图的最终标签和完整评分逐项提交到 `results.sqlite`，只为 `--export-limit` 指定数量的样本导出完整图片（默认 12 张）。`--config 参数.json` 接受“方法名 → 嵌套配置覆盖”，同时启用流式模式。`--resume --output 已有目录` 跳过已经成功的任务，重试失败或缺失任务；恢复会核对清单、文件哈希、参数、方法、示例选择、实现版本和环境。同一输出目录只允许一个流式运行进程。

每个模型的一次性启动时间记录在 `run.json.initialization`（分割脚本为 `summary.json.initialization`）；评测页面分别展示首次启动时间和单张分割时间。`wall_seconds` / `mean_wall_seconds` 不再包含模型加载，但包含进程间数据传递、推理及分区后处理；不包含结果保存、评测和绘图。没有额外预热，首张推理也计入平均值。耗时协议版本为 2，不能直接与旧版每图重新加载的时间混用。

默认 debug 和 `--image` 都是开发评测，标注身份记为 `unknown`。正式清单需将样本标记为 `test`、`manual`，并填写非空 `annotation_evidence`。流式模式对 `manual` 样本额外检查 MAT 内的 `gt_type`。本地已核实人工标注共 9,922 张（Tools 9,632、Clutter 290）；当前从中固定抽取 720 张做多轮调参，不能作为独立测试成绩。混合 dev/test 清单必须用 `--split` 选择，Tools 同一实例不能跨两组。运行中的个别方法失败后继续其他任务，记录失败并返回非零退出码；未完成的方法/子集汇总分数留空，逐图成功结果仍保留。格式与指标解释见 [06：UMD 分割评价](docs/06_umd_affordance_partition_benchmark.md)。

本次验证通过 55 项单元/集成测试，包括最优组合与穷举结果一致、忽略像素、分组平均、失败记录、两个入口无参数运行、模型按顺序复用、进程异常和中断清理。五张真实 debug 图、六种方法共 30 次分割全部成功；每种学习模型的五张图片使用同一个进程，完成后均已退出。30 份最终标签与旧流程逐像素一致。查看 [常驻模型开发集评测页面](outputs/affordance_benchmark_20261018_205942_939471/index.html) / [汇总表](outputs/affordance_benchmark_20261018_205942_939471/summary.csv)。该小样本仅验证实现，未覆盖全部七类，不能作为正式性能排名。

上述常驻模型验证在 `cuda:1` 上运行。SAM2、PartGLEE、MoSA 的单张分割平均约为 0.87、0.85、0.92 秒，首次准备分别约为 4.31、6.02、3.24 秒；实际耗时取决于图片与设备负载。[初版开发集结果](outputs/umd_benchmark/debug_all_v1/index.html) 包含每张图重新加载的开销，仅供历史对照。

## 检查与研究记录

```bash
python -m unittest discover -s tests -v
```

测试覆盖统一接口、配置隔离、重叠和漏覆盖、空候选、连通性、非整字节宽度的无损导出，以及不导入模型依赖的 SLIC 命令行运行。PartGLEE 还覆盖多物体及超过 10 个部件的候选、分层去重、原生分数排序和损坏权重检测。MoSA 还覆盖按前景概率选择 top-k、原生 softmax 分数、先裁除填充再恢复非正方形尺寸、多物体及嵌套 Mask 去重、可关闭过滤和损坏权重检测。

2026-10-18 首次重构验证（历史参数，提交 `0fbe7d0`）：11 项测试与 `pip check` 通过；15 份分区与重构前对应参数逐像素一致。查看 [历史三方法结果](outputs/dense_partition_refactored/index.html)。当前生成参数及后处理已调整，历史结果不代表当前默认输出。

加入 Semantic-SAM 前的配置通过 11 项测试和单图三方法真实运行；三份导出均通过全覆盖、无重叠及连通性检查。[此前结果](outputs/demo_controls_smoke/index.html) 的 [summary.json](outputs/demo_controls_smoke/summary.json) 记录实际参数。

2026-10-18 Semantic-SAM 接入验证：同一 Conda 环境中 16 项测试及 `pip check` 通过；默认全部层级完成 5 张 debug 图片的 GPU 推理，所有导出通过原分辨率、全覆盖、无重叠及连通性验证。查看 [Semantic-SAM 对照页面](outputs/semantic_sam_debug/index.html) / [总览图](outputs/semantic_sam_debug/overview.jpg)。最终区域数包含解重叠与连通域拆分，不能当作模型原始候选数或准确率。

2026-10-18 PartGLEE 接入验证：同一 Conda 环境中 30 项测试及 `pip check` 通过；完成 5 张 debug 图片的 GPU 推理及同一杂乱场景的五方法运行，所有导出通过原分辨率、全覆盖、无重叠、连通性和保存后回读检查。查看 [PartGLEE 五图结果](outputs/partglee_debug/index.html) / [总览图](outputs/partglee_debug/overview.jpg)，以及 [五方法对照](outputs/partglee_comparison/index.html)。杂乱场景筛选后保留 10 个物体候选、13 个部件候选，统一后处理后得到 65 个分区；候选数不等于真实物体数，这些检查也不代表分割准确率评估。

2026-10-18 MoSA 接入验证：同一 Conda 环境中 36 项测试及 `pip check` 通过；完成 5 张 debug 图片及同一杂乱场景的六方法运行，所有导出通过原分辨率、全覆盖、无重叠、四连通和保存后回读检查。在非正方形、带填充的合成 logits 上，以批量 1 / 4 / 16 对照上游 `MaskFormer.instance_inference`，二值 Mask 完全一致，原生分数在浮点容差内一致。查看 [MoSA 五图结果](outputs/mosa_debug/index.html) / [总览图](outputs/mosa_debug/overview.jpg)，以及 [六方法对照](outputs/mosa_comparison/index.html)。

| 调试图 | MoSA 筛选后候选 | 最终连通区域 |
| --- | ---: | ---: |
| knife | 197 | 123 |
| mug | 217 | 141 |
| spoon | 212 | 131 |
| hammer | 201 | 130 |
| clutter | 202 | 145 |

本次五图在 RTX 4090 D、`cuda:1` 上运行，默认短边 1024、FP32；模型推理及候选尺寸还原约 1.35–2.33 秒/图，worker 总耗时约 5.59–9.18 秒/图，峰值 CUDA 张量分配约 10.78 GiB。该内存统计不包含其他进程或 CUDA 保留内存，实际耗时会随图像和设备负载变化。可视化中工具的部分局部区域被分开，背景细分也较多；尚未进行带匿名部件标注的准确率评测，候选数或覆盖率不能证明优于其他方法。

研究资料见 [文档导航](docs/README.md)。历史 A、C、D 实验方案见 [最小 Affordance 决策测试](docs/04_minimal_affordance_test.md)；任务范式、数据集与后续研究方向见 [2D Affordance 深度调研](docs/00_SystemOne_2DAffordance_深度调研.md)；六种分割方法的评测流程与指标见 [UMD 分割评价方案](docs/06_umd_affordance_partition_benchmark.md)。当前实现的架构、用法与验证记录以本 README 为准。
