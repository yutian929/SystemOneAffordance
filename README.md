# SystemOneAffordance

探索支持视觉输入的 System One 模型在 **2D Affordance（二维可供性）** 任务中的应用。

后续以 **方法 D** 为主线：SAM 自动候选 → 连通域内部中心点 → Valen 点判断 → SAM 点提示分割 → mask 并集。初版 A、D 的结果保留用于对照。

## 环境配置与使用

使用独立的 Miniconda 环境。环境定义在 [environment.yml](environment.yml)，Python 依赖固定在 [requirements.txt](requirements.txt)，SAM 源码提交固定在 [requirements-sam.txt](requirements-sam.txt)。

| 项目 | 当前配置 |
| --- | --- |
| 环境名称 | `systemone-affordance` |
| 本机 Miniconda | `/home/hri/miniconda3` |
| 本机环境目录 | `/home/hri/miniconda3/envs/systemone-affordance` |
| Python | 3.10 |
| PyTorch / torchvision | `2.6.0+cu124` / `0.21.0+cu124` |
| Transformers | `5.4.0` |
| 模型 | SAM 2.1 small、Valen-4B；可选 UnSAMv2+ 全图候选 |

### 日常使用：激活已有环境

**本机环境已经配置完成。** 每次打开终端，先执行：

```bash
cd /data0/yutian/SystemOneAffordance
source /home/hri/miniconda3/etc/profile.d/conda.sh
conda activate systemone-affordance
unset PYTHONPATH
```

随后可执行下文的模型下载、D 推理或测试命令。`unset PYTHONPATH` 用于清除其他项目遗留的导入路径；隔离和缓存设置会随 Conda 环境自动生效。若刚更新过环境变量，重新激活一次环境。结束工作时可执行 `conda deactivate`。

### 首次安装：创建环境并安装依赖

以下步骤用于环境尚未创建的机器。本机日常运行可跳过。其他机器先按 [Miniconda 官方安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/) 安装，并调整下面的仓库和 Miniconda 路径；安装 SAM 源码依赖还需要系统中有 Git。

```bash
cd /data0/yutian/SystemOneAffordance
source /home/hri/miniconda3/etc/profile.d/conda.sh
conda create --override-channels -c conda-forge -n systemone-affordance \
  python=3.10 pip setuptools=79.0.1 wheel -y
conda env config vars set -n systemone-affordance \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
conda activate systemone-affordance
unset PYTHONPATH
python -m pip install -r requirements.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements-sam.txt
python -m pip check
```

这些步骤与 `environment.yml` 的依赖定义一致，并显式指定 conda-forge，避免全局渠道配置干扰。SAM 需在 PyTorch 安装后单独安装；`environment.yml` 中的 pip 步骤只安装 `requirements.txt`。`SAM2_BUILD_CUDA=0` 关闭 SAM 的可选 CUDA 扩展编译，当前 D 使用的配置无需该扩展。GPU 运行使用 CUDA 12.4 的 PyTorch wheel，需要可用且兼容的 NVIDIA 驱动。

下载遇到本机 CA 证书问题时，可给该次命令设置 `PIP_CERT=/etc/ssl/certs/ca-certificates.crt` 或 `CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt`，保持证书验证开启。

### 检查环境

激活环境后执行：

```bash
python - <<'PY'
import site
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("User site enabled:", site.ENABLE_USER_SITE)
print("Skip bytecode:", sys.dont_write_bytecode)
PY
python -m pip check
```

本机 Python 路径应为 `/home/hri/miniconda3/envs/systemone-affordance/bin/python`；GPU 运行时 `CUDA available` 应为 `True`，`User site enabled` 应为 `False`，`Skip bytecode` 应为 `True`。`pip check` 应输出 `No broken requirements found.`。

### 缓存与仓库整洁

Python 环境和 Hugging Face 全局缓存放在仓库外：

| 配置或目录 | 用途 |
| --- | --- |
| `~/.cache/huggingface` | 默认 Hugging Face 全局缓存，可用 `HF_HOME` 调整 |
| `~/.cache/huggingface/modules` | 默认 Valen 自定义模型代码缓存，可用 `HF_MODULES_CACHE` 单独调整 |
| `PYTHONNOUSERSITE=1` | 避免导入用户级 `site-packages` 中的包 |
| `PYTHONDONTWRITEBYTECODE=1` | 避免运行时生成 `__pycache__/` |

自定义缓存位置时选择仓库外的路径。模型代码缓存可从本地模型文件自动重建；迁移前的仓库内 `.venv/`、根目录 `.cache/` 和已有 `__pycache__/` 已清理。模型下载工具也会在 `models/` 对应模型目录内维护下载元数据。

## 下载模型与运行

激活环境后下载并校验模型，已有验证完成的模型目录可以直接使用：

```bash
python scripts/download_models.py
```

模型保存在 `models/SAM2.1-small/` 和 `models/Valen-4B/`。下载需要联网；D 推理使用本地模型并启用离线加载。`configs/umd_debug.json` 中的图片及可选标注也需要事先准备到对应路径，模型下载脚本只下载模型。

D 直接读取样本清单并生成候选：

```bash
python run_method_d.py \
  --manifest configs/umd_debug.json \
  --output outputs/method_d_run \
  --device cuda:1
```

`cuda:1` 表示第二张可见 GPU；单卡机器可改为 `--device cuda:0`。可用 `nvidia-smi` 查看设备和显存占用。

`--output` 必须是尚不存在的目录，重复实验时换一个名称。阈值可用 `--threshold` 指定，默认 0.5。已有兼容候选目录时，可增加 `--candidate-cache <旧运行目录>` 复用旧 A/D 或新 D 批量运行的候选；程序会核对图像、SAM 版本和候选配置。

`configs/umd_debug.json` 决定图片和指令。`label`、`target_label` 仅用于预测完成后的评估，可同时省略。

运行结果写入指定输出目录：`summary.json` / `summary.csv` 为汇总，`overview.jpg` 为总览，`samples/<任务ID>/prediction.png` 为最终 mask，`run.json` 记录环境及参数。数据、模型和实验输出均由 `.gitignore` 排除。

### UnSAMv2+：可调粒度的全图候选

激活同一 Conda 环境后，安装新增依赖并下载官方权重及固定版本的推理代码：

```bash
python -m pip install opencv-python-headless==4.11.0.86
python scripts/download_unsamv2.py
```

权重保存到 `models/UnSAMv2-plus/`；官方推理源码保存到仓库外的 `~/.cache/systemone-affordance/UnSAMv2-f4ec1136/`，可通过 `UNSAMV2_CODE_DIR` 调整。官方 fork 也使用 `sam2` 包名，因此候选生成在独立子进程中执行，原 SAM 2 安装继续供 D 的最终点提示分割使用。

只查看全分割候选和中心点：

```bash
python run_method_d.py \
  --stage prepare \
  --candidate-model unsamv2-plus \
  --granularity 0.15 \
  --max-candidates 0 \
  --manifest configs/umd_debug.json \
  --output outputs/unsamv2_prepare \
  --device cuda:1
```

`--granularity` 范围为 `[0.1, 1.0]`，越小越细；`--max-candidates 0` 保留去重后的全部候选。默认上限仍为 24，细粒度探索时建议显式设置为 0。`--stage prepare` 生成候选和点图，跳过 Valen 及任务分割。该阶段的 `run.json` 中 `prepared=true`、`completed=false`，表示候选准备完成，完整任务预测尚未执行。

改为 `--stage run` 并选择新输出目录，即可运行完整 D；若要复用上面的候选，增加 `--candidate-cache outputs/unsamv2_prepare`，候选模型、粒度和数量上限需与缓存一致。当前 UnSAMv2+ 接入的是**候选生成阶段**，最终点提示分割仍使用 SAM 2.1 small。

比较已有运行的全图候选、取点和离线覆盖指标：

```bash
python compare_candidates.py \
  --runs sam2=outputs/method_d_run unsam015=outputs/unsamv2_prepare \
  --output outputs/candidate_comparison
```

打开比较目录中的 `index.html` 可切换方案、显示采样点；`overview.jpg` 是总览，`comparison.csv` / `comparison.json` 记录配置和指标。两组输入目录都必须已经完成同一份清单的候选准备。实验设计与结果见 [文档 05](docs/05_method_d_analysis_and_improvements.md)。

### Dense Partition：每个像素恰好属于一个 Mask

这一入口只比较候选分割，不运行 Valen 或后续点提示分割，也不读取 GT。对 debug 的 5 张不同图像分别生成 SAM 2.1、UnSAMv2+ 和传统 SLIC 超像素结果：

```bash
python -m pip install scikit-image==0.25.2
python run_dense_partition.py --device cuda:1 \
  --granularity 0.15 --segments 600 1200 \
  --output outputs/dense_partition_new
```

两个模型的原始候选会经过统一的重叠归属、连通空白区补齐和微小区域合并，得到互斥分区；另外与同一份 SLIC 分区取交集，进一步细分大区域。这两档细分明确属于 **模型 + SLIC** 组合，页面默认展示没有 SLIC 细分的模型分区。SLIC 本身直接生成全覆盖分区。

打开输出目录的 `index.html`，可以切换每张图、细分档位和显示方式，点击像素查看三个方法对应的 Mask，并下载选中的二值 Mask。`overview_native.jpg`、`overview_s600.jpg`、`overview_s1200.jpg` 为全部图像的并排总览。粉色视图展示原模型未覆盖的像素。

`partitions/<图像哈希>/<方法>/<档位>/labels.npy` 是完整像素标签图，`labels == region_id` 即一个 Mask；`masks.npz` 保存无损位压缩的二值 Mask 集合，可用 `dense_partition.iter_masks(path)` 逐个读取。全部导出会重新读入验证：每个像素的 Mask 数严格为 1，且每个 Mask 为 4 连通。具体参数、选型依据和结果见 [文档 05 第 12 节](docs/05_method_d_analysis_and_improvements.md#12-dense-partition全覆盖互斥的细粒度候选)。

调整 SLIC 密度时，可加 `--proposal-cache outputs/dense_partition_new` 复用同参数模型候选，输出仍需选择新目录。

## Python 主类

主类 [MethodD](method_d.py) 位于仓库根目录，可用于单图预测或批量实验：

```python
from method_d import MethodD

with MethodD(device="cuda:1", output_dir="outputs/my_prediction") as pipeline:
    result = pipeline.predict(
        "data/umd/tools/knife_01/knife_01_00000122_rgb.jpg",
        "Find all parts used to cut other objects.",
        sample_id="knife_cut",
    )
    mask = result.mask  # bool[H, W]
    points = result.prepared.points
    point_records = result.records
    individual_masks = result.segmentations
```

`prepare_image(image)` 生成并缓存候选和点；`prepare_manifest(path)` 只准备整份清单的候选；`predict(image, instruction)` 返回 `PredictionResult`，不接收 GT；`run_manifest(path)` 执行批量预测及可选评估。同一实例复用 SAM 2、Valen 和同图候选，不同指令重新判断点；UnSAMv2+ 按图启动子进程，加载及推理耗时单独记录。再次使用同一个 `sample_id` 会报错，避免覆盖。

主类同样支持 `candidate_model="unsamv2-plus"`、`granularity=0.15` 和 `max_candidates=0`。默认 `candidate_model="sam2"` 对应初版基线；`generator_config` 可覆盖候选生成器参数，用于保持生成参数一致的对照。

省略 `output_dir` 时使用临时工作目录，退出上下文后自动清理；返回的 mask 等数组仍可使用。需要保留中间文件时指定新输出目录。实例用于串行调用。

## 代码结构与检查

```text
method_d.py             # MethodD、PreparedImage、PredictionResult
run_method_d.py         # 命令行入口
method_d_utils.py       # 取点、标记、点提示及点评估
mask_utils.py           # 候选过滤、mask 合并及像素评估
experiment_utils.py     # 清单、文件、模型元数据和可视化工具
unsamv2_backend.py      # UnSAMv2+ 官方实现的独立进程适配
compare_candidates.py  # 已生成候选的可视化与离线比较
dense_partition.py     # 互斥全覆盖分区、SLIC、Mask 导出与验证
run_dense_partition.py # 三方法分区对照与交互查看
scripts/
  download_models.py    # 下载 SAM 2、Valen
  download_unsamv2.py    # 下载 UnSAMv2+ 权重和仓库外的推理代码
configs/umd_debug.json
tests/
environment.yml
requirements.txt
requirements-sam.txt
```

```bash
python -m unittest discover -s tests -v
```

测试包括连通域与并集语义，以及主类的同图多指令、空预测、GT 读取时机和候选缓存校验。首版代码可从 Git 提交 `3d57ce5` 查看，历史实验结论见文档 05。

2026-10-17 已在上述 Miniconda 环境中通过 18 项测试和 `pip check`，并完成 6 条任务的两次真实模型回归：导入旧候选、D 自行生成候选。两次运行的候选、点、111 个 Noul 分数、15 次 SAM 分割及最终 mask 均与初版完全一致，平均 IoU 为 0.1241727471。缓存清理后也通过了 18 项测试和真实模型单图推理检查。实验输出是未纳入 Git 的本地产物，清理后需重新运行生成。

新增 UnSAMv2+ 后，21 项测试与 `pip check` 通过；已实际运行 4 档粒度候选、同生成参数 SAM 2 对照，以及两组新的完整 D。`g=0.15` 的平均 IoU 为 0.125702，改善了部分部件采样，也存在漏检；详细结果与可视化见 [文档 05 第 11 节](docs/05_method_d_analysis_and_improvements.md#11-unsamv2细粒度全图候选实验)。

Dense Partition 实验新增后，27 项测试与 `pip check` 通过。5 张图、3 个方法、3 个展示档位的 45 份导出全部通过逐像素互斥全覆盖检查；本地浏览器验证了方案切换、Mask 高亮和二值 Mask 下载。实际效果与限制见文档 05 第 12 节。

## 阅读入口

1. [Affordance 入门](docs/01_affordance_primer.md)：理解任务、输入输出和机器人执行之间的关系。
2. [视觉 System One 模型](docs/02_visual_system_one_models.md)：了解可选模型、图像接口和开放范围。
3. [System One 与 Affordance 的已有工作](docs/03_system_one_affordance_research.md)：核实直接应用、相邻研究，以及与交互区域分割的差别。
4. [最小 Affordance 决策测试](docs/04_minimal_affordance_test.md)：比较区域独立判断与随机点前后景判断，两种方式均支持多个有效区域。
5. [方法 D：初版流程、结果诊断与改进](docs/05_method_d_analysis_and_improvements.md)：详细记录 D 的输入输出、中间变量、流程图、初版结果及待验证的改进方案。

资料来源随正文附链接，首轮核对日期为 2026-10-15。方法 A、D 已完成少量样本的调试运行；结果与后续验证计划见文档 05，尚未完成独立检查集评估。
