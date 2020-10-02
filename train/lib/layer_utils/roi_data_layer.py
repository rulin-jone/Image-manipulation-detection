"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from lib.config import config as cfg
from lib.utils.minibatch import get_minibatch



class RoIDataLayer(object):
    """Fast R-CNN data layer used for training"""

    def __init__(self, roidb, num_classes, random=False):
        """set the roidb to be used by this layer during training"""
        self._roidb = roidb
        self._num_classes = num_classes
        self._random = random
        # 设置一个random flag
        # 得到self._perm（为0---len(self._roidb)打乱顺序构成的数组）和置self._cur = 0
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """randomly permute the training roidb"""
        # 当self._random置1时，随机改变训练用roidb的排列顺序
        # 使用np.random.permutation()函数打乱数组顺序
        # 对验证集同样适用 useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)
            # 生成随机种子

        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        if self._random:
            np.random.set_state(st0)

        self._cur = 0
        # minibatch_rois索引开始的标志
        # self._cur相当于一个指向_perm的指针，每次取走图片后，他会跟着变化

    def _get_next_minibatch_inds(self):
        """return the roidb indices for the next minibatch"""
        # 获取下一个mini_batch中roidb在所有图像构成的roidb中的索引并返回
        if self._cur + cfg.FLAGS.ims_per_batch >= len(self._roidb):
            # 防止越界
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur:self._cur + cfg.FLAGS.ims_per_batch] # 下一次roidb的下标
        self._cur += cfg.FLAGS.ims_per_batch

        return db_inds


    def _get_next_minibatch(self):

        """return the blobs to be used for the next minibatch

        if cfg.TRAIN.USE_PREFETCH （好像没有这个参数）
        is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        """
        获取下一个mini_batch中roidb（即为minibatch.py中roidb，部分图像的rois相关信息），
        调用_get_next_minibatch_inds()获取下一个mini_batch中roidb的索引--->获取minibatch_db（即minibatch.py中的roidb）
        --->以minibatch_db作为参数调用get_minibatch(minibatch_db, self._num_classes)构造网络输入blobs
        （默认训练阶段使用RPN时，blob含'data'、'gt_boxes'、'gt_ishard'、'dontcare_area'、'im_info'、'im_name'字段），被forward(...)调用
        """
        # 获取下一批图片的索引，即获取下一个mini_batch中roidb
        db_inds = self._get_next_minibatch_inds()
        # 把对应索引的图像信息（dict）取出来，放入一个列表中
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def forward(self):
        # 调用_get_next_minibatch()得到blobs并返回
        """get blobs and copy them into this layer's top blob vector"""
        blobs = self._get_next_minibatch()
        # blobs['noise'] = SRM(blobs['data'])
        return blobs































