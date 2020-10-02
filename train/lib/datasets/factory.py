

"""通过数据集名字获取数据集数据"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# print('1\n')
__sets = {}
from lib.datasets.pascal_voc import pascal_voc
# from lib.datasets.coco import coco
from lib.datasets.DIY_pascal_voc import DIY_pascal_voc

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
# print('2')

# # Set up coco_2014_<split>
# for year in ['2014']:
#     for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))
#
# # Set up coco_2015_<split>
# for year in ['2015']:
#     for split in ['test', 'test-dev']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))

for year in ['2018']:
    for split in ['trainval']:
        name = 'DIY_dataset'
        __sets[name] = (lambda split=split, year=year: DIY_pascal_voc(split, year))

# print('3\n')

def get_imdb(name):
    # 通过数据集的名称返回数据集对应的类
    # 数据集对应的类分别在lib.datasets下
    # 其中,pascal_voc.py文件中对应定义了pascal_voc数据集对应的类
    # DIY_pascal_voc.py文件中对应了DIY数据集中对应的类，其实这个跟pascal_voc.py区别不大，但为了好区分还是做了两个文件
    """
    返回的类（实际上返回的是指向这个类的指针）：
    class pascal_voc
    {
    self._year
    self._image_set
    self.__devkit_path
    ...
    }
    """

    # print('sets', __sets)
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())