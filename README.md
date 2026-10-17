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
| 模型 | SAM 2.1 small、Valen-4B |

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

`prepare_image(image)` 生成并缓存候选和点；`predict(image, instruction)` 复用这些数据并返回 `PredictionResult`，不接收 GT；`run_manifest(path)` 执行批量预测及可选评估。同一实例复用模型和同图候选，但不同指令重新判断点。再次使用同一个 `sample_id` 会报错，避免覆盖。

省略 `output_dir` 时使用临时工作目录，退出上下文后自动清理；返回的 mask 等数组仍可使用。需要保留中间文件时指定新输出目录。实例用于串行调用。

## 代码结构与检查

```text
method_d.py             # MethodD、PreparedImage、PredictionResult
run_method_d.py         # 命令行入口
method_d_utils.py       # 取点、标记、点提示及点评估
mask_utils.py           # 候选过滤、mask 合并及像素评估
experiment_utils.py     # 清单、文件、模型元数据和可视化工具
scripts/
  download_models.py    # scripts 中唯一的脚本
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

## 阅读入口

1. [Affordance 入门](docs/01_affordance_primer.md)：理解任务、输入输出和机器人执行之间的关系。
2. [视觉 System One 模型](docs/02_visual_system_one_models.md)：了解可选模型、图像接口和开放范围。
3. [System One 与 Affordance 的已有工作](docs/03_system_one_affordance_research.md)：核实直接应用、相邻研究，以及与交互区域分割的差别。
4. [最小 Affordance 决策测试](docs/04_minimal_affordance_test.md)：比较区域独立判断与随机点前后景判断，两种方式均支持多个有效区域。
5. [方法 D：初版流程、结果诊断与改进](docs/05_method_d_analysis_and_improvements.md)：详细记录 D 的输入输出、中间变量、流程图、初版结果及待验证的改进方案。

资料来源随正文附链接，首轮核对日期为 2026-10-15。方法 A、D 已完成少量样本的调试运行；结果与后续验证计划见文档 05，尚未完成独立检查集评估。
