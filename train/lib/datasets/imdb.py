
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from lib.config import config as cfg

'''
imdb class 为所有数据集的父类，包含了所有数据集的共有特性
例如：数据集名称（name）、数据集类名列表（classes）、数据集的文件名列表（_image_index)、roi集合、config
'''

'''
roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含的roi信息，下面以roidb[img_index]为例说明：
boxes：box位置信息，box_num*4的 np array
gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
gt_classes：所有box的真实类别，box_num长度的list
filpped：是否翻转
max_overlaps：每个box的在所有类别的得分最大值，box_num长度
max_classes：每个box的得分最高对应的类，box_num长度
'''

class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name  # 数据集名称
        self._num_classes = 0  # 数据集类别个数
        if not classes:
            self._classes = []
        else:
            self._classes = classes  # 数据集类名列表
        self._image_index = []  # 数据集图片文件名列表 例如例如 data/VOCdevkit2007/VOC2007/ImageSets/Main/{image_set}.txt
        self._obj_proposer = 'gt'
        self._roidb = None  # 这是一个字典，里面包含了gt_box、真实标签、gt_overlaps和翻转标签flipped:true，代表图片被水平翻转
        self._roidb_handler = self.default_roidb  # roi数据列表
        # Use this dict for storing dataset specific config options
        self.config = {}
    # 通过property修饰器定义只读属性
    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    # 返回ground-truth每个ROI构成的数据集
    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    # 属性函数
    @property
    # self.roidb即可调用
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        #   如果已经有了，那么直接返回，如果没有就通过指针指向的函数生成
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb
    # property为属性函数，可以将类方法转化为类属性进行调用，例如本例(self.cache_path)
    # cache_path用来生成roidb缓存文件的文件夹，用来存储数据集的roi
    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.FLAGS2["data_dir"], 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    # 返回该数据集下（note：此时train、test、val是分开的）中定义由多少个图片，就是在train.txt中定义的
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list element is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    # 返回该部分数据集索引图像的size[0]，即宽度，存在一个list
    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    # 对图像进行水平翻转，进行数据增强
    def append_flipped_images(self):
        num_images = self.num_images
        # 格式：list
        widths = self._get_widths()
        # self.roidb最后调用到pascal_voc.gt_roidb最后返回的是一个列表，列表中存在图像信息dict
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            # 取出该幅图片所包含的bbox的xmin, xmax，存在oldx1，oldx2中
            # eg. boxes=([1,2,3,4],[5,6,7,8]) 则oldx1=(1,5), oldx2=(3,7)
            '''
            这里要加copy()，如果不用.copy()则会出现对应同一个存储区域的异名参数，如引用，copy则会另分出一个储存空间，对原数据无影响
            例如dic = {'name': 'liubo', 'num': [1,2,3]}
            dic1 = dic
            dic2 = dic.copy()
            dic['name'] = '123123' # 修改父对象dic
            dic['num'].remove(1) # 修改父对象dic中的[1,2,3]列表子对象
            # 输出结果
            print(dic) # {'name': '123123', 'num': [2,3]}
            print(dic1) # {'name': '123123', 'num': [2,3]}
            print(dic2) # {'name': 'liubo', 'num': [2,3]}
            # 也就是说用copy，父对象不会因为dic的改变而改变，但子对象会
            '''
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            # 变化坐标，将xmax变成xmin，xmin变成xmax关于x的对称点
            # 反转后boxes=([-1,2,1,4],[-5,6,-3,8])

            assert (boxes[:, 2] >= boxes[:, 0].all())
            # 插入一个异常，如果该式子不是所有都成立，则数据读取有异常
            # .all()函数如果对应iterable中有一个false，则返回false
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True}
            # 将新的boxes信息存入 entry字典
            self.roidb.append(entry)
            # 将entry依次加入roidb列表，回忆一下roidb列表中存的是图片信息dict，图片信息引索与self.image_index引索相对应
        self._image_index = self._image_index * 2
        # 由于数据增强（添加了水平反转数据），且水平反转数据还是按照image_index的顺序，所以只需要执行image_insdx=image_index*2


        # def evaluate_recall(self, candidate_boxes=None, thresholds=None,
        #                     area='all', limit=None):
        #     """Evaluate detection proposal recall metrics.
        #
        #     Returns:
        #         results: dictionary of results with keys
        #             'ar': average recall
        #             'recalls': vector recalls at each IoU overlap threshold
        #             'thresholds': vector of IoU overlap thresholds
        #             'gt_overlaps': vector of all ground-truth overlaps
        #     """
        #     # Record max overlap value for each gt box
        #     # Return vector of overlap values
        #     areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
        #              '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        #     area_ranges = [[0 ** 2, 1e5 ** 2],  # all
        #                    [0 ** 2, 32 ** 2],  # small
        #                    [32 ** 2, 96 ** 2],  # medium
        #                    [96 ** 2, 1e5 ** 2],  # large
        #                    [96 ** 2, 128 ** 2],  # 96-128
        #                    [128 ** 2, 256 ** 2],  # 128-256
        #                    [256 ** 2, 512 ** 2],  # 256-512
        #                    [512 ** 2, 1e5 ** 2],  # 512-inf
        #                    ]
        #     assert area in areas, 'unknown area range: {}'.format(area)
        #     area_range = area_ranges[areas[area]]
        #     gt_overlaps = np.zeros(0)
        #     num_pos = 0
        #     for i in range(self.num_images):
        #         # Checking for max_overlaps == 1 avoids including crowd annotations
        #         # (...pretty hacking :/)
        #         max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
        #         gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
        #                            (max_gt_overlaps == 1))[0]
        #         gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
        #         gt_areas = self.roidb[i]['seg_areas'][gt_inds]
        #         valid_gt_inds = np.where((gt_areas >= area_range[0]) &
        #                                  (gt_areas <= area_range[1]))[0]
        #         gt_boxes = gt_boxes[valid_gt_inds, :]
        #         num_pos += len(valid_gt_inds)
        #
        #         if candidate_boxes is None:
        #             # If candidate_boxes is not supplied, the default is to use the
        #             # non-ground-truth boxes from this roidb
        #             non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        #             boxes = self.roidb[i]['boxes'][non_gt_inds, :]
        #         else:
        #             boxes = candidate_boxes[i]
        #         if boxes.shape[0] == 0:
        #             continue
        #         if limit is not None and boxes.shape[0] > limit:
        #             boxes = boxes[:limit, :]
        #
        #         overlaps = bbox_overlaps(boxes.astype(np.float),
        #                                  gt_boxes.astype(np.float))
        #
        #         _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        #         for j in range(gt_boxes.shape[0]):
        #             # find which proposal box maximally covers each gt box
        #             argmax_overlaps = overlaps.argmax(axis=0)
        #             # and get the iou amount of coverage for each gt box
        #             max_overlaps = overlaps.max(axis=0)
        #             # find which gt box is 'best' covered (i.e. 'best' = most iou)
        #             gt_ind = max_overlaps.argmax()
        #             gt_ovr = max_overlaps.max()
        #             assert (gt_ovr >= 0)
        #             # find the proposal box that covers the best covered gt box
        #             box_ind = argmax_overlaps[gt_ind]
        #             # record the iou coverage of this gt box
        #             _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        #             assert (_gt_overlaps[j] == gt_ovr)
        #             # mark the proposal box and the gt box as used
        #             overlaps[box_ind, :] = -1
        #             overlaps[:, gt_ind] = -1
        #         # append recorded iou coverage level
        #         gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
        #
        #     gt_overlaps = np.sort(gt_overlaps)
        #     if thresholds is None:
        #         step = 0.05
        #         thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        #     recalls = np.zeros_like(thresholds)
        #     # compute recall for each iou threshold
        #     for i, t in enumerate(thresholds):
        #         recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        #     # ar = 2 * np.trapz(recalls, thresholds)
        #     ar = recalls.mean()
        #     return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
        #             'gt_overlaps': gt_overlaps}

        # def create_roidb_from_box_list(self, box_list, gt_roidb):
        #     assert len(box_list) == self.num_images, \
        #         'Number of boxes must match number of ground-truth images'
        #     roidb = []
        #     for i in range(self.num_images):
        #         boxes = box_list[i]
        #         num_boxes = boxes.shape[0]
        #         overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
        #
        #         if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
        #             gt_boxes = gt_roidb[i]['boxes']
        #             gt_classes = gt_roidb[i]['gt_classes']
        #             gt_overlaps = bbox_overlaps(boxes.astype(np.float),
        #                                         gt_boxes.astype(np.float))
        #             argmaxes = gt_overlaps.argmax(axis=1)
        #             maxes = gt_overlaps.max(axis=1)
        #             I = np.where(maxes > 0)[0]
        #             overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
        #
        #         overlaps = scipy.sparse.csr_matrix(overlaps)
        #         roidb.append({
        #             'boxes': boxes,
        #             'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        #             'gt_overlaps': overlaps,
        #             'flipped': False,
        #             'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
        #         })
        #     return roidb

    # 将a b两个roidb归并为一个roidb
    @staticmethod
    def mer_roidbs(a, b):
        assert  len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a][i]['gt_overlaps'], b[i]['gt_overlaps'])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'], b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass























