import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}






# 数据集参数设置
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__)))
print()
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))
# 获取config.py的文件路径并组合成默认的文件路径供_get_default_path使用













