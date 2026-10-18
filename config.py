"""日常配置；各方法的主要调节项集中在这里。"""
from common.settings import ROOT, METHOD_NAMES, make_config, validate_config

DEFAULT_METHOD = 'slic'  # Python API 的默认方法；命令行未指定 --method 时运行全部方法
OUTPUT_DIR = ROOT / 'outputs/dense_partition'
DEVICE = 'cuda:1'  # 单卡改为 cuda:0；SLIC 使用 CPU
MIN_MASK_AREA = 16  # 原图像素数；小于此面积的最终区域并入邻居，0 关闭；适用于全部方法

UNSAMV2_PLUS = {
    'granularity': 0.2,  # [0.1, 1.0]，越小倾向越细；过小可能漏掉物体/部件
    'pred_iou_thresh': 0.77,  # Predicted IoU threshold：降低保留更多候选
    'stability_score_thresh': 0.9,  # Stability threshold：降低筛选更宽松
}

SEMANTIC_SAM = {
    'level': [1, 2, 3, 4, 5, 6],  # 1/2 偏语义，3 实例，4/5/6 部件；可只用 [3] 或 [6]；全部层级能提供更多互补候选
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
    return make_config(method, device=DEVICE, unsam=UNSAMV2_PLUS, slic=SLIC,
                       semantic_sam=SEMANTIC_SAM, min_mask_area=MIN_MASK_AREA)
