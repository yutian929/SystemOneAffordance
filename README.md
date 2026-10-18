# SystemOneAffordance

当前只实现 **Dense Partition / Exhaustive Segmentation（全覆盖分割）**，作为后续 Affordance 研究的基础。输入一张 RGB 图片，输出一组 Mask，严格满足：**每个像素恰好属于一个 Mask**。目前支持 SAM 2.1 small、UnSAMv2+、Semantic-SAM 和 SLIC，统一通过 `DensePartition` 接口调用。

## 代码结构

```text
dense_partition.py     # 抽象基类、四种实现、统一工厂及命令行入口
config.py              # 日常调节项、设备、debug 图片
common/
  partition.py         # 解重叠、补齐区域、连通域、小区域合并、验证
  settings.py          # 模型资源、官方 Demo 内部设置与参数校验
  result.py            # PartitionResult：标签图、Mask 访问与保存
  io.py                # 图像读取、哈希、JSON、无损 Mask 导出与读取
  models.py            # 各模型的资源检查、推理、尺寸还原与运行兼容
  visualization.py     # 边界图、总览、交互查看页面
scripts/
  download_sam2.py      # 仅下载 SAM 2.1
  download_unsamv2.py   # 仅下载 UnSAMv2+ 权重及固定版本源码
  download_semantic_sam.py # Semantic-SAM 权重、固定源码及 Detectron2 编译
tests/test_dense_partition.py
tests/test_semantic_sam.py
tests/test_small_regions.py
docs/
environment.yml
requirements.txt
requirements-sam.txt
requirements-semantic-sam.txt
```

全覆盖分割功能的命令行逻辑直接写在 `dense_partition.py` 的 `if __name__ == '__main__':` 中，负责参数解析、单图或批量运行及结果导出。

SAM 2.1、UnSAMv2+、Semantic-SAM 的推理都归入 `common/models.py`，统一通过当前 Conda Python 启动短生命周期子进程。Semantic-SAM 的尺寸还原和兼容处理也在该模块内完成；本项目使用 `common` 包名，与其官方 `utils` 包可以同时导入。

旧 `configs/` 目录、完整 D 流程、Valen 判断、取点、任务指令和 GT 评估代码已经移除。历史实验保留在 `docs/` 和本地 `outputs/`；旧实现可从 Git 提交 `3d62972` 查看。数据、模型、输出不纳入 Git。

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
| `DEFAULT_METHOD` | Python API 默认 `slic`；命令行未指定 `--method` 时运行全部方法 |
| `DEVICE` | 当前 `cuda:1`，单卡机器改为 `cuda:0`。SLIC 始终使用 CPU |
| `MIN_MASK_AREA` | 所有方法统一的后处理阈值，默认 `16`：面积严格小于 16 个原图像素的区域并入邻居；`0` 关闭 |
| `UNSAMV2_PLUS.granularity` | 范围 `[0.1, 1.0]`，默认 `0.2`；越小倾向越细，但过小可能漏掉物体/部件，并非越小越好；HF Demo 初始值为 `0.15` |
| `UNSAMV2_PLUS.pred_iou_thresh` | 对应 Predicted IoU threshold；降低保留更多候选，HF Demo 默认 `0.77` |
| `UNSAMV2_PLUS.stability_score_thresh` | 对应 Stability threshold；降低筛选更宽松，HF Demo 默认 `0.9` |
| `SEMANTIC_SAM.level` | 唯一日常控制项：`[1]` / `[2]` 语义级，`[3]` 实例级，`[4]` / `[5]` / `[6]` 不同部件粒度；默认 `[1, 2, 3, 4, 5, 6]` 保留全部层级 |
| `SLIC` | `n_segments` 越大越细；`compactness` 控制形状规则性，`sigma` 控制平滑；目标数不等于最终 Mask 数 |
| `DEBUG_IMAGES` | 5 张不同的 debug 图片，只有图片路径，不包含任务指令或标注 |
| `OUTPUT_DIR` | 默认 `outputs/dense_partition` |

SAM 2.1 直接使用官方自动生成器默认值。UnSAMv2+ 独立使用上述三个控制项，其余采样设置参考 [官方 HF Demo 源码](https://huggingface.co/spaces/yujunwei04/UnSAMv2/blob/675a2bceabf284b69d5d537a3aa798b53c50a8ff/app.py)，不再与 SAM 共用一组生成参数。模型资源与内部设置在 `common/settings.py`；`model_config='configs/…yaml'` 是第三方包内部资源路径。

Semantic-SAM 使用官方 Swin-T 权重和 `SemanticSamAutomaticMaskGenerator`，只传入 `level`，其他生成参数采用官方默认值。按官方 `prepare_image` 将短边缩放到 640，生成候选后使用最近邻还原至原图分辨率，再进入统一分区流程。`level` 的含义见 [官方用法](https://github.com/UX-Decoder/Semantic-SAM#-a-few-lines-to-get-generated-results)；它是离散层级列表，可填写 `[3]`、`[6]` 或组合，不能直接等同于目标 Mask 数。公开权重只在 SA-1B 上训练，这里输出多粒度候选，不输出 affordance 或语义类别标签。

最终分区先消除重叠、补齐未覆盖区域并拆分连通域，再按 `MIN_MASK_AREA` 合并小区域。合并在 `common/partition.py` 的 `merge_small_regions()` 中实现，由 `DensePartition.segment()` 对四种方法统一调用：从最小区域开始，并入共享边界最长的上下左右相邻区域；共享边界相同时优先面积更大的邻居，再按区域 ID 决定。每次合并后更新面积和邻接关系，直到所有区域达到阈值或整图仅剩一个区域；若整张图片面积就小于阈值，保留这一个区域。阈值 `16` 用于清除少量像素的碎片，面积恰好 16 的区域保留；设置为 `0` 恢复不合并的行为。

Raw 候选与原始覆盖率不受此合并影响。HF Demo 最多展示前 1000 个原始候选，本地保留全部候选后建立严格分区，因此最终 Mask 数及展示方式不等同于 Demo。

## Python 使用

```python
from dense_partition import DensePartition

segmenter = DensePartition.create("slic")  # 也可选 "sam2"、"unsamv2-plus"、"semantic-sam"
result = segmenter.segment("data/umd/tools/knife_01/knife_01_00000122_rgb.jpg")

labels = result.labels          # uint32[H, W]，ID 为 1…K，保留原图分辨率
mask = result.mask(1)            # bool[H, W]
print(result.region_count)      # K
for region_id, mask in result.iter_masks():
    pass
result.save("outputs/my_partition")  # 必须使用新目录
```

输入也支持 PIL 图像或 `uint8[H,W,3]` RGB 数组。`result.masks` 可一次生成 `bool[K,H,W]`，大图建议逐个读取以节约内存。`result.rgb` 和 `result.labels` 默认只读；`result.metadata` 记录配置、原模型覆盖率与耗时。`result.owners` 为原始模型候选索引加一，0 表示原模型未覆盖；SLIC 的该字段为 `None`。`owners` 始终记录合并前的像素来源，最终标签会拆分连通域、合并小区域并重新编号，因此一个最终 Mask 可以包含多个来源。

需要在调用时改变参数，可先取得独立配置；修改不会影响其他实例：

```python
from config import get_config
from dense_partition import DensePartition

settings = get_config("slic")
settings["algorithm"]["n_segments"] = 600
result = DensePartition.create("slic", config=settings).segment("your_image.jpg")
```

`DensePartition` 是抽象基类；`SAM21Partition`、`UnSAMv2PlusPartition`、`SemanticSAMPartition`、`SLICPartition` 均位于 `dense_partition.py`。新增方法实现 `_segment_rgb()`，由基类 `segment()` 统一读取图片、记录配置并验证输出契约。

## 命令行与可视化

```bash
# 默认使用 config.DEBUG_IMAGES，对比全部四种方法
python dense_partition.py --output outputs/partition_comparison

# 指定单图，默认对比全部四种方法
python dense_partition.py --image your_image.jpg --output outputs/one_image

# 只运行 Semantic-SAM，粒度在 config.py 的 SEMANTIC_SAM.level 中调节
python dense_partition.py --method semantic-sam --output outputs/semantic_sam_run

# 只运行 SLIC；也支持 --image 后面指定多张图片
python dense_partition.py --method slic --output outputs/slic_debug
```

不指定 `--image` 时默认读取 `config.DEBUG_IMAGES`；不指定 `--method` 时运行全部方法（SAM 2.1、UnSAMv2+、Semantic-SAM、SLIC）。`--debug` 仍可显式使用，与 `--image` 互斥。

输出目录必须尚不存在，防止覆盖旧结果。打开输出目录中的 `index.html`，可切换图片、查看原尺寸，点击像素查看各方法对应的最终 Mask，并下载二值 PNG。

页面每个方法占一行，固定按 **Original → Raw colors → Boundaries → Colors → Overlay** 展示：原图、原始候选纯色图、最终分区边界图、最终分区纯色图、最终分区叠加图。每行只保留 Raw / Final Mask 数量和选中区域信息；详细统计在 `summary.json` 中。`overview.jpg` 按图片、方法逐行排列，每行使用相同的五列顺序；单个方法的 `comparison.jpg` 也采用此顺序。

Raw 指自动生成器 `generate()` 返回的全部候选，已包含其内部的质量过滤和去重，尚未经过本项目的分区后处理；SLIC 则展示连通域重编号前的直接输出。Raw colors 中黑色表示没有候选覆盖，重叠处优先展示小 Mask。额外的覆盖情况图仍保存在 `raw_coverage.png`（粉色为未覆盖、橙色为重叠）和 `coverage.png`（粉色为补齐区域来源）中。

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

SAM / UnSAM / Semantic-SAM 的原始候选先按小区域优先消除重叠，再将所有区域（包括未覆盖像素）拆成连通域。模型分别使用当前 Conda 的 Python 在短生命周期子进程中运行，以隔离两套 `sam2` 包并释放显存；每张图都需要重新加载模型。SLIC 直接生成分区，随后与模型方法一样执行小区域合并。基类检查正整数连续编号、原分辨率和 4 连通性；导出时再读取二值 Mask，逐像素检查 `sum(masks[:, y, x]) == 1`。

全覆盖是输出结构保证，不能作为分割准确率；模型漏覆盖区域补齐后也不代表模型识别出了这些区域。当前不进行模型与 SLIC 求交细分，四种方法各自独立输出。

## 检查与研究记录

```bash
python -m unittest discover -s tests -v
```

测试覆盖统一接口、配置隔离、重叠和漏覆盖、空候选、连通性、非整字节宽度的无损导出，以及不导入模型依赖的 SLIC 命令行运行。

2026-10-18 首次重构验证（历史参数，提交 `0fbe7d0`）：11 项测试与 `pip check` 通过；15 份分区与重构前对应参数逐像素一致。查看 [历史三方法结果](outputs/dense_partition_refactored/index.html)。当前生成参数及后处理已调整，历史结果不代表当前默认输出。

加入 Semantic-SAM 前的配置通过 11 项测试和单图三方法真实运行；三份导出均通过全覆盖、无重叠及连通性检查。[此前结果](outputs/demo_controls_smoke/index.html) 的 [summary.json](outputs/demo_controls_smoke/summary.json) 记录实际参数。

2026-10-18 Semantic-SAM 接入验证：同一 Conda 环境中 16 项测试及 `pip check` 通过；默认全部层级完成 5 张 debug 图片的 GPU 推理，所有导出通过原分辨率、全覆盖、无重叠及连通性验证。查看 [Semantic-SAM 对照页面](outputs/semantic_sam_debug/index.html) / [总览图](outputs/semantic_sam_debug/overview.jpg)。最终区域数包含解重叠与连通域拆分，不能当作模型原始候选数或准确率。

研究资料见 [文档导航](docs/README.md)；历史方法 D、三方法分割对照和本次重构架构见 [文档 05](docs/05_method_d_analysis_and_improvements.md)。
