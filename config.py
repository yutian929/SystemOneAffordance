"""日常配置；SAM 使用官方默认值，UnSAMv2+ 暴露 HF Demo 的三个控制项。"""
from utils.settings import ROOT, METHOD_NAMES, make_config, validate_config

DEFAULT_METHOD = 'slic'
OUTPUT_DIR = ROOT / 'outputs/dense_partition'
DEVICE = 'cuda:1'  # 单卡改为 cuda:0；SLIC 使用 CPU

UNSAMV2_PLUS = {
    'granularity': 0.2,  # [0.1, 1.0]，越小倾向越细
    'pred_iou_thresh': 0.77,  # Predicted IoU threshold：降低保留更多候选
    'stability_score_thresh': 0.9,  # Stability threshold：降低筛选更宽松
}

SLIC = {
    'n_segments': 50,  # 目标区域数，越大越细
    'compactness': 5.0,  # 小值更贴颜色边界，大值形状更规则
    'sigma': 0.5,  # 平滑强度，减小保留更多纹理
}

DEBUG_IMAGES = (
    ROOT / 'data/umd/tools/knife_01/knife_01_00000122_rgb.jpg',
    ROOT / 'data/umd/tools/mug_01/mug_01_00000151_rgb.jpg',
    ROOT / 'data/umd/tools/spoon_01/spoon_01_00000172_rgb.jpg',
    ROOT / 'data/umd/tools/hammer_01/hammer_01_00000137_rgb.jpg',
    ROOT / 'data/umd/clutter/scene_01/scene_01_00000166_rgb.jpg',
)


def get_config(method=DEFAULT_METHOD):
    """合并内部设置，返回本次运行的完整参数副本。"""
    return make_config(method, device=DEVICE, unsam=UNSAMV2_PLUS, slic=SLIC)
