# SystemOneAffordance

当前只实现 **Dense Partition / Exhaustive Segmentation（全覆盖分割）**，作为后续 Affordance 研究的基础。输入一张 RGB 图片，输出一组 Mask，严格满足：**每个像素恰好属于一个 Mask**。目前支持 SAM 2.1 small、UnSAMv2+ 和 SLIC，统一通过 `DensePartition` 接口调用。

## 代码结构

```text
dense_partition.py     # 抽象基类、三种实现、统一工厂及命令行入口
config.py              # 所有算法参数、模型路径、运行设置、debug 图片
utils/
  partition.py         # 解重叠、补齐区域、连通域、微小区域合并、验证
  result.py            # PartitionResult：标签图、Mask 访问与保存
  io.py                # 图像读取、哈希、JSON、无损 Mask 导出与读取
  models.py            # 模型资源检查与独立推理进程
  visualization.py     # 边界图、总览、交互查看页面
  cli.py               # 单图或批量运行
scripts/
  download_sam2.py      # 仅下载 SAM 2.1
  download_unsamv2.py   # 仅下载 UnSAMv2+ 权重及固定版本源码
tests/test_dense_partition.py
docs/
environment.yml
requirements.txt
requirements-sam.txt
```

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

`environment.yml` 设置 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`，避免其他项目的用户包干扰和仓库内字节码缓存。Conda 环境、Hugging Face 全局缓存及 UnSAMv2 源码放在仓库外。依赖清单已移除 Transformers 和视频解码依赖；运行 SLIC 不导入 PyTorch，也不需要模型文件。

## 模型与参数

只用 SLIC 可以跳过下载；运行学习模型前执行：

```bash
python scripts/download_sam2.py
python scripts/download_unsamv2.py
```

权重分别保存在 `models/SAM2.1-small/` 和 `models/UnSAMv2-plus/`。下载脚本校验固定版本与权重 SHA-256。UnSAMv2 源码默认放在 `~/.cache/systemone-affordance/UnSAMv2-f4ec1136/`，也可设置仓库外的 `UNSAMV2_CODE_DIR`。推理读取本地资源，不自动下载。

所有参数集中修改 [config.py](config.py)：

| 配置 | 默认值与用途 |
| --- | --- |
| `DEFAULT_METHOD` | `slic` |
| `RUNTIME` | `device='cuda:1'`、种子 0、GPU 使用 bfloat16；单卡机器改为 `cuda:0`。SLIC 始终使用 CPU |
| `SAM2` / `UNSAMV2_PLUS` | 权重目录、模型配置；UnSAMv2+ `granularity=0.15`，范围 `[0.1, 1.0]`，较小值请求较细候选 |
| `AUTOMATIC_MASK_GENERATOR` | 两模型共用默认参数：点网格 96×96、batch 64、IoU 阈值 0.5、稳定性阈值 0.8；各实例持有独立副本 |
| `PARTITION` | 模型分区中面积小于 16 像素的区域合并到相邻且平均 Lab 颜色最接近的区域；设为 1 可关闭面积合并 |
| `SLIC` | `n_segments=1200`、`compactness=5`、`sigma=0.5`，并启用连通性约束；目标区域数不等于最终 Mask 数 |
| `DEBUG_IMAGES` | 5 张不同的 debug 图片，只有图片路径，不包含任务指令或标注 |
| `OUTPUT_DIR` | 默认 `outputs/dense_partition` |

配置中的 `model_config='configs/…yaml'` 是**第三方 SAM 包内部**的资源路径，与已经删除的本仓库 `configs/` 无关。

## Python 使用

```python
from dense_partition import DensePartition

segmenter = DensePartition.create("slic")  # 也可选 "sam2"、"unsamv2-plus"
result = segmenter.segment("data/umd/tools/knife_01/knife_01_00000122_rgb.jpg")

labels = result.labels          # uint32[H, W]，ID 为 1…K，保留原图分辨率
mask = result.mask(1)            # bool[H, W]
print(result.region_count)      # K
for region_id, mask in result.iter_masks():
    pass
result.save("outputs/my_partition")  # 必须使用新目录
```

输入也支持 PIL 图像或 `uint8[H,W,3]` RGB 数组。`result.masks` 可一次生成 `bool[K,H,W]`，大图建议逐个读取以节约内存。`result.rgb` 和 `result.labels` 默认只读；`result.metadata` 记录配置、原模型覆盖率与耗时。`result.owners` 为原始模型候选索引加一，0 表示原模型未覆盖；SLIC 的该字段为 `None`。`owners` 是合并前的来源记录，不能当作最终标签图。

需要在调用时改变参数，可先取得独立配置；修改不会影响其他实例：

```python
from config import get_config
from dense_partition import DensePartition

settings = get_config("slic")
settings["algorithm"]["n_segments"] = 600
result = DensePartition.create("slic", config=settings).segment("your_image.jpg")
```

`DensePartition` 是抽象基类；`SAM21Partition`、`UnSAMv2PlusPartition`、`SLICPartition` 均位于 `dense_partition.py`。新增方法实现 `_segment_rgb()`，由基类 `segment()` 统一读取图片、记录配置并验证输出契约。

## 命令行与可视化

```bash
# 单图，默认 SLIC
python dense_partition.py --image your_image.jpg --output outputs/one_image

# debug 中每张图对比三种方法
python dense_partition.py --debug --method sam2 unsamv2-plus slic \
  --output outputs/partition_comparison

# 只运行 SLIC；也支持 --image 后面指定多张图片
python dense_partition.py --debug --method slic --output outputs/slic_debug
```

输出目录必须尚不存在，防止覆盖旧结果。打开输出目录中的 `index.html`，可切换图片、显示方式，点击像素查看各方法对应的 Mask，并下载二值 PNG；`overview.jpg` 是并排总览。

```text
summary.json                         # 参数、统计和 completed 状态
index.html / overview.jpg
partitions/<图像内容哈希>/<方法>/
  labels.npy                         # uint32[H,W]，最终标签
  masks.npz                          # 无损位压缩 bool Mask 集合
  metadata.json                      # 区域面积、来源、配置和导出验证结果
  pixel_owners.npy                    # 仅模型方法：原始候选的像素归属
  original.png / ids.png
  boundaries.png / overlay.png / colors.png / coverage.png
```

读取已有导出：

```python
import numpy as np
from utils.io import iter_masks

labels = np.load("<分区目录>/labels.npy", allow_pickle=False)
for region_id, mask in iter_masks("<分区目录>/masks.npz"):
    assert np.array_equal(mask, labels == region_id)
```

SAM / UnSAM 的原始候选先按小区域优先消除重叠，再将未覆盖像素按连通域补齐、合并微小区域。两模型分别在短生命周期子进程中运行，以隔离同名 `sam2` 包并释放显存；每张图都需要重新加载模型。SLIC 直接生成分区。基类检查正整数连续编号、原分辨率和 4 连通性；导出时再读取二值 Mask，逐像素检查 `sum(masks[:, y, x]) == 1`。

全覆盖是输出结构保证，不能作为分割准确率；模型漏覆盖区域补齐后也不代表模型识别出了这些区域。当前不进行模型与 SLIC 求交细分，三种方法各自独立输出。

## 检查与研究记录

```bash
python -m unittest discover -s tests -v
```

测试覆盖统一接口、配置隔离、重叠和漏覆盖、空候选、连通性、非整字节宽度的无损导出，以及不导入模型依赖的 SLIC 命令行运行。

2026-10-18 验证：11 项测试与 `pip check` 通过；5 张 debug 图片 × 3 方法的 15 份分区均通过导出检查，标签图与重构前对应参数逐像素一致；交互页面及下载 Mask 已在浏览器中验证。查看 [本次三方法结果](outputs/dense_partition_refactored/index.html)。

研究资料见 [文档导航](docs/README.md)；历史方法 D、三方法分割对照和本次重构架构见 [文档 05](docs/05_method_d_analysis_and_improvements.md)。
