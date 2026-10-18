# UMD：面向 2D Affordance 的分割评价方案

评测已实现在根目录 [affordance_benchmark.py](../affordance_benchmark.py)，包含抽象基类 `AffordanceBenchmark`、UMD 实现 `UMDBenchmark`、指标计算、CSV 导出和可视化。默认运行五张 debug 图及全部六种方法；正式人工标注帧清单仍需核实和提供，默认结果属于开发评测。

## 1. 我们到底要比较什么？

**比较哪种方法能用较少的区域，把刀柄、刀刃、杯柄等功能部位分清楚，并且运行得快。**

例如，UMD 标注了一把刀的刀柄是 `grasp`，刀刃是 `cut`：

- 方法 A 把刀柄和刀刃分开，后续容易选出刀柄。
- 方法 B 把刀柄分成三块，三块合起来也能得到刀柄，但需要判断更多区域。
- 方法 C 把整把刀分成一块，选它会带上刀刃，不选它会漏掉刀柄。

这次先评价上述分区能力。用真实标注替代后续判断，看看各方法的区域能组合得多准确；得到的是**理想选择下的分割能力**，不能当作模型已经理解了 `grasp`。

## 2. 整个流程是什么？

```text
固定同一批 UMD 图片及对应标注
                  │
                  ▼
只把 RGB 图片分别交给六种分割方法
SAM2 / UnSAMv2+ / Semantic-SAM / PartGLEE / MoSA / SLIC
                  │
                  ▼
每种方法输出一张区域编号图 labels.npy
                  │
                  ▼
评测器读取 UMD 标注，例如取出全部 grasp 像素
                  │
                  ▼
从分区中选出最接近这个真实区域的一组区域，计算分数
再检查每区域取一个点时，是否命中了真实功能区域
                  │
                  ▼
输出：逐图分数、各方法汇总表、图片对照页面
```

一张图只需要每种方法分割一次。图中有 `grasp` 和 `cut` 时，使用同一份分区分别评测两个功能。

**分割方法只接收 RGB。真实标注只交给评测器，不参与分割、调节测试图参数或选点。** 第一版无需自然语言指令、深度图或决策模型。

## 3. 输入是什么？文件长什么样？

### 3.1 一张 RGB 图片 + 一张真实标签图

| 输入 | 文件格式 | 读取后的内容 |
| --- | --- | --- |
| 图片 | `*_rgb.jpg` | `uint8[480,640,3]`，RGB 三通道 |
| 真实标签，简称 GT | `*_label.mat` | 读取其中的 `gt_label`，形状为 `[480,640]`，每个像素是一个功能类别编号 |

类别编号为：`0=背景`、`1=grasp（抓握）`、`2=cut（切割）`、`3=scoop（舀取）`、`4=contain（容纳）`、`5=pound（敲击）`、`6=support（支撑）`、`7=wrap-grasp（包握）`。

例如，`gt_label == 1` 就得到 grasp 的二值 Mask。第一版使用这种“主要功能”标签，不使用 `*_label_rank.mat` 中的多功能排序标签。[UMD 官方格式说明](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt)

### 3.2 用一个 JSON 文件固定评测图片

自定义清单可以保存为 `benchmarks/umd/manifest.json`，每条记录是一张图片；仓库没有预置正式测试清单。下面是格式示例；这张现有 debug 图的人工标注身份尚未核实，所以写成 `unknown`，不能直接列入正式测试。

```json
{
  "version": 1,
  "samples": [
    {
      "sample_id": "knife_01_00000122",
      "subset": "tools",
      "sequence_id": "knife_01",
      "split": "dev",
      "annotation_status": "unknown",
      "rgb": "data/umd/tools/knife_01/knife_01_00000122_rgb.jpg",
      "label": "data/umd/tools/knife_01/knife_01_00000122_label.mat"
    }
  ]
}
```

路径相对仓库根目录；`subset` 是 `tools` 或 `clutter`；`sequence_id` 是工具实例或场景；`split` 是开发用的 `dev` 或正式测试用的 `test`；`annotation_status` 是 `manual`、`automatic` 或 `unknown`。

正式测试要求 `split="test"`、`annotation_status="manual"`，并在样本中或 JSON 顶层提供非空字符串 `annotation_evidence`，说明核实人工标注身份的依据。程序校验并保存这些字段，不自动验证依据的真实性，也不猜测人工帧号。图片清单和每种方法的配置在运行前固定，所有方法使用同一批图片。

## 4. 分割结果是什么？

继续使用当前接口：

```python
with DensePartition.create("mosa") as segmenter:
    result = segmenter.segment(rgb_path)  # 同一个 segmenter 可连续分割多张图
labels = result.labels
```

`labels` 是 `uint32[H,W]` 的区域编号图。它与原图同尺寸，值为 `1…K`；相同编号的像素属于同一个区域。

```text
分区编号示意：           UMD 功能标签示意：
1 1 2 2                0 0 0 0
1 3 3 2                0 1 1 0
1 4 4 2                0 2 2 0
```

**左边的 1、2、3、4 只是区域 ID，不代表 grasp、cut 等类别；右边的编号才有功能含义。** 评测器要比较区域的形状，而不是直接比较这两张图的编号是否相等。

第一版评价的是完成解重叠、补齐、连通域拆分和小区域合并后的最终分区。六种方法统一读取 `config.MIN_MASK_AREA`，默认值为 16；运行时保存完整配置快照。

## 5. 用哪些指标？每个是什么意思？

| 指标 | 直白解释 | 方向 |
| --- | --- | --- |
| **最优组合 IoU（Oracle IoU）** | 如果区域都选对了，组合出的形状能有多接近真实功能部位？ | 越高越好 |
| **达到 0.8 的比例** | 有多少评测样本能组合到至少 0.8 的 IoU？ | 越高越好 |
| **区域数 K** | 一张图被切成多少块？逐区域判断需要处理多少块？ | 质量相近时越少越好 |
| **代表点命中率** | 每块取一个点，能覆盖多少个真实功能连通区域？ | 越高越好 |
| **分割耗时** | 从调用分割到拿到最终分区需要几秒？ | 越低越好 |

### 5.1 最优组合 IoU：主要质量指标

IoU 就是“重合面积 ÷ 合并后的总面积”：

```text
IoU = 正确选中的像素数 /（正确选中的像素数 + 多选的像素数 + 漏选的像素数）
```

例如真实刀柄有 100 个像素，组合结果正确选中 80 个、多选 10 个、漏选 20 个，那么 IoU 为 `80 / (80 + 10 + 20) ≈ 0.727`。完全一致是 1，完全不重合是 0。

“最优组合”是指评测器利用 GT，在现有区域中找出 IoU 最大的一组。例如选中区域 `3、5、8`，把它们合起来，再与 grasp 的 GT 比较。允许选多个区域，也允许结果包含多个互不相连的部位。

对当前互不重叠的分区，计算方法很简单：

1. 计算每个区域中，目标功能像素占有效像素的比例。
2. 按这个比例从高到低排列区域。
3. 依次计算“选前 1 个、前 2 个、……、全部区域”的 IoU。
4. 取最高分，并保存对应区域 ID；同分时取区域数较少的前缀。

这能得到当前互斥分区的精确最优 IoU，不是固定用 50% 阈值选区域。没有有效像素的区域不参与计算。这个算法不直接适用于互相重叠的原始候选 Mask。

这个分数是“选择并合并已有区域”的能力上限。以后若让 SAM 根据点重新生成边界，需要另测最终结果。

### 5.2 达到 0.8 的比例：有多少样本已经分得比较好？

一条质量评测样本是“图片 + 一个出现的功能类别”。例如一张刀图有 grasp 和 cut，就是两条。

若 100 条样本里有 75 条的最优组合 IoU **大于等于 0.8**，达标比例就是 75%。0.8 是本项目固定的观察阈值，不是 UMD 官方规定的成功标准。

### 5.3 区域数 K：判断成本有多大？

统计最终分区的全部区域数，包括背景区域。假设后续每区域判断一次，`K=50` 就对应 50 次判断，`K=500` 就对应 500 次。

它不是模型原始候选数量，也不是最优组合中选中的区域数量。只把图切得更碎，IoU 可能升高，所以必须一起看 **IoU 和 K**，不按 IoU 单独宣布赢家。

### 5.4 代表点命中率：会不会漏掉需要判断的部位？

对每个最终区域，计算像素坐标的平均位置，再取区域内部离它最近的像素。距离相同时按行、列顺序决定。点由分区产生，不看 GT；保存坐标时统一用 `[x,y]`。

对某个功能 GT 按上下左右相邻拆分连通区域，检查每个连通区域内有没有至少一个代表点：

```text
代表点命中率 = 被至少一个点命中的 GT 连通区域数 / GT 连通区域总数
```

例如画面有三个分离的 grasp 区域，点只落入其中两个，命中率就是 `2/3`。第一版不删除细小 GT 连通区域；落在忽略像素上的点不算命中，也不根据 GT 重新选点。

这里的连通区域不等于物体实例。命中一个点也不代表整个部位已经分准；这个指标用于检查后续取点流程的漏采问题。

### 5.5 分割耗时：真实使用要等多久？

现在的运行顺序是：加载模型 A → 处理全部图片 → 释放模型 A → 加载模型 B → 处理全部图片。每种学习模型只有一个独立进程、一份驻留权重，不再每张图重新加载；同一时间只保留当前模型，避免多个大模型同时占用显存。SLIC 使用 CPU，没有权重加载。

耗时分开记录：

| 字段 | 含义 | 保存位置 |
|---|---|---|
| `startup_seconds` | 每种模型只发生一次的准备时间，包括权重校验、启动进程、导入依赖、构建和加载模型 | `run.json.initialization.<方法名>` |
| `model_load_seconds` | 上述准备时间中，构建模型、加载权重并移到设备所花的时间；它是子项，不要再与准备时间相加 | 同上 |
| `wall_seconds` | 模型准备好后，一张图从调用 `segment(rgb)` 到拿到最终分区的时间，包括数据传递、推理、后处理和验证 | `image_metrics.csv` |

图片文件读取、结果保存、GT 评测和绘图都不计入单张分割耗时。`summary.csv.mean_wall_seconds` 按实例/场景汇总这些单张时间。分割结果中的 `metadata.segment_seconds` 从读取 RGB、初始化之后开始计时，范围略窄；benchmark 用外层调用统一计时。模型内部生成候选的时间另存于 `metadata.backend.generate_seconds`，其范围由对应后端定义。

所有方法在相同机器条件下依次运行，并保存设备和配置。没有额外预热，首张图的推理时间也计入结果；它可能比后续图片慢。耗时协议版本为 2，旧版包含每张图重新加载的耗时不能直接混入本版比较。页面分别显示一次性准备时间和分割秒/图。模型初始化失败时一次记录该方法全部图片失败，并继续下一个方法；单图失败后继续其余图片，退出或中断时释放进程。

## 6. 最后输出哪些文件？

统一输出到一个新目录，默认目录名带时间戳。下列文件已实现，数字示例仅用于说明格式，不代表实测结果。

```text
outputs/affordance_benchmark_<年月日_时分秒_微秒>/
  manifest.json              # 本次实际使用的图片清单
  run.json                   # 参数快照、版本、设备、一次性初始化耗时、标注依据、失败记录
  image_metrics.csv          # 每张图、每种方法的 K 和耗时
  scores.csv                 # 每张图、每种方法、每个功能的质量分数
  summary.csv                # 按方法、Tools/Clutter 汇总的结果表
  per_class.csv              # grasp、cut 等七类分别的结果
  per_sequence.csv           # 按工具实例/Clutter 场景、功能分别统计
  index.html                 # 评测对照页面
  partitions/<sample_id>/<method>/
    labels.npy               # 沿用现有分区导出
    metadata.json            # 沿用现有配置、来源和统计
    ...                      # 现有原图、Mask、可视化等文件
  evaluations/<sample_id>/<method>/
    points.json              # 区域 ID 与 [x,y] 代表点坐标
    grasp/
      gt.png                 # 真实区域，0/255 二值图
      oracle.png             # 最优组合区域，0/255 二值图
      valid.png              # 参与评价的像素为 255，忽略像素为 0
      errors.png             # 绿=正确覆盖，红=多选，蓝=漏选，灰=忽略
      selected_regions.json  # 例如 {"region_ids": [3, 5, 8]}
```

`image_metrics.csv` 每张图、每种方法只有一行，避免一张图出现多个功能时重复统计耗时：

```csv
sample_id,subset,sequence_id,method,region_count,wall_seconds,valid_pixels,ignored_pixels,ignored_fraction
example_001,tools,example_tool,mosa,100,8.2,307200,0,0.0
```

`scores.csv` 每行是一张图、一个方法、一个功能的结果：

```csv
sample_id,subset,sequence_id,method,affordance_id,affordance,oracle_iou,success_080,gt_components,hit_components,point_hit_rate,gt_pixels,true_positive_pixels,false_positive_pixels,false_negative_pixels,selected_region_count
example_001,tools,example_tool,mosa,1,grasp,0.88,1,3,2,0.666667,100,88,0,12,3
```

页面按 **原图 → 功能 GT → 方法分区 → 最优组合 → 多选/漏选像素** 展示，同时显示代表点和分数。这样能看到“是功能边界混在一起，还是区域太碎，还是点没有落进去”。

## 7. 数据怎样选，分数怎样汇总？

输入与计分采用以下规则；正式集的规模是建议，实际样本由清单决定，脚本不自动抽样：

| 项目 | 规则 |
| --- | --- |
| 开发集 | 先用现有 5 张 debug 图检查流程，结果只用于调试 |
| Tools 正式集 | 参考官方 `category_split.txt` 的测试实例，从已核实人工标注的帧中按实例均匀抽样，第一轮约 200～500 张；反复调参用过的实例不作为独立测试证据，排除项需记录 |
| Clutter 正式集 | 三个场景分别从已核实人工标注的帧中均匀抽样，每场景最多 20 张；用过的 debug 帧单独标记，不计正式分数 |
| 人工标注帧 | 官方说明每三帧中一帧是人工标注，但具体帧号规则仍需核实；核实前可以做流程测试，不发布正式成绩 |
| 背景 0 | 参与评价；把背景误选进功能区域会扣 IoU |
| 标签 -1 | 本项目第一版约定为忽略像素，所有质量指标都排除它，并记录比例；官方含义尚未明确，不把它解释成背景 |
| 图中不存在的功能 | 不计该图该类的 Oracle IoU、达标比例和点命中率，避免“空对空”抬高成绩 |
| 失败样本 | 记录原因及失败率；同一批样本未完成时标记结果不完整，不通过静默删除失败行给出排名 |

人工/自动标注的区别来自 [UMD 官方说明](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt)。本地图片数量、原始划分位置和 `-1` 检查记录见 [UMD 本地数据说明](../data/umd/README.md)。上述抽样及忽略规则是本项目的评测约定。

Tools 与 Clutter 在汇总表中分开记录。每种方法、每个功能先在同一实例内对有效帧取平均，再对实例取平均，最后对有有效样本的功能类别等权平均；Clutter 将“实例”换成“场景”，单场景成绩保存在 `per_sequence.csv`。点命中率和达标比例采用相同汇总方式。某类没有有效样本时记为缺失，不补 0 或 1，`class_count` 记录实际覆盖了七类中的几类。

K 和耗时按图片统计，先在实例/场景内平均，再对实例/场景等权平均，不按功能类别重复计算。

最终表的格式是：

| 方法 | 平均最优组合 IoU ↑ | 0.8 达标比例 ↑ | 平均 K | 代表点命中率 ↑ | 平均秒/图 ↓ |
| --- | --- | --- | --- | --- | --- |
| SAM2 | 待测 | 待测 | 待测 | 待测 | 待测 |
| UnSAMv2+ | 待测 | 待测 | 待测 | 待测 | 待测 |
| Semantic-SAM | 待测 | 待测 | 待测 | 待测 | 待测 |
| PartGLEE | 待测 | 待测 | 待测 | 待测 | 待测 |
| MoSA | 待测 | 待测 | 待测 | 待测 | 待测 |
| SLIC | 待测 | 待测 | 待测 | 待测 | 待测 |

## 8. 怎样运行？

激活仓库 README 中的 `systemone-affordance` 环境后，在仓库根目录执行：

```bash
# 默认五张 debug 图，全部六种方法
python affordance_benchmark.py

# 只运行 SLIC，快速检查完整流程
python affordance_benchmark.py --method slic

# 指定 UMD 图片，自动寻找同目录标签；支持多张图片，仍默认为全部方法
python affordance_benchmark.py --image data/umd/tools/knife_01/knife_01_00000122_rgb.jpg

# 自定义清单；混合 dev/test 清单必须指定其中一组
python affordance_benchmark.py --manifest benchmarks/umd/manifest.json --split test

# 自定义输出目录，目录必须尚不存在
python affordance_benchmark.py --method mosa slic --output outputs/umd_mosa_slic
```

`--image`、`--debug` 和 `--manifest` 互斥。没有指定输入时用 `config.DEBUG_IMAGES`；没有指定 `--method` 时跑六种方法。默认和 `--image` 均标记为开发评测。两个脚本直接执行即可跑完全部方法：`python dense_partition.py` 自动输出到 `outputs/dense_partition_时间戳/`，`python affordance_benchmark.py` 自动输出到 `outputs/affordance_benchmark_时间戳/`。输出根目录由 `config.OUTPUT_DIR` 控制，时间戳精确到微秒，重复运行生成新目录；`--output` 只是可选覆盖项。

Python 接口：

```python
from affordance_benchmark import AffordanceBenchmark

benchmark = AffordanceBenchmark.create("umd")
manifest = benchmark.development_manifest()
run = benchmark.run(manifest, "outputs/my_umd_benchmark")  # 全部方法

# 已有分区和同尺寸 GT 时，可只计算质量，不重新推理
evaluation = benchmark.evaluate(labels, gt)
```

本版完整运行每次重新分割，不自动复用旧目录；输出目录不能覆盖。个别任务失败时会记录异常并继续其他方法，`run.json` 标记为 `incomplete`，命令退出码为 1；未完成的方法/子集不计算汇总质量与成本，保留成功的逐图明细。Ctrl+C 会保存已有分数和页面，并标记为 `interrupted`。

后续扩展是核实并固定正式测试清单，以及在开发集上选择粗、中、细几组参数，画“区域数 K—IoU”曲线。这两项不影响当前 debug 评测入口使用。
