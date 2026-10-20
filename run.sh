cd /data0/yutian/SystemOneAffordance
source /home/hri/miniconda3/etc/profile.d/conda.sh
conda activate systemone-affordance
unset PYTHONPATH
python main.py --image /data0/yutian/SystemOneAffordance/data/umd/clutter/scene_01/scene_01_00000000_rgb.jpg --device cuda:1