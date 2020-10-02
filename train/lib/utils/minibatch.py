"""compute minibatch blobs for training a Fast R-CNN network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    # given a roidb, construct a minibatch sampled from it
    num_images = len(roidb)
    # roidb中元素个数，就是要处理的图片个数
    # sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.FLAGS2["scales"]),
                                    size=num_images) # 根据scales的数量，为每张图片生成一个scale的索引
    assert (cfg.FLAGS.batch_size % num_images == 0), 'num_images ({}) must divide batch_size ({})'.format(num_images, cfg.FLAGS.batch_size)
    # 要求batch_size必须是图片数量的整数倍

    # rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images  # 计算平均从每个图片上要产生多少个roi输入
    # fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)  # 按比率计算每张图片的roi中需要多少个前景

    # get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    # 调用函数，读取图片数据，返回矩阵数据和每张图片的尺寸缩放信息
    # roidb中图片经过了减去均值、缩放操作

    blobs = {'data': im_blob}
    # 把blobs字典的key：data赋值为im_blob，也就是图片的矩阵数据

    assert len(im_scales) == 1, "single batch only"
    assert len(roidb) == 1, "single batch only"

    # gt boxes: (x1,y1,x2,y2,cls)
    if cfg.FLAGS.use_all_gt:
        # include all gt boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0] # 获得所有box真实类别的下标
    else:
        # for the COCO gt boxes, exclude the ones that are "iscrowd"
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    '''
    这里全部用的是roidb[0]
    传进来的roidb是通过_get_next_minibatch获取pre_minibatch里面的图片
    然后通过_get_next_minibatch_inds获取图片的下标
    虽然可以改变 cfg.FLAGS.ims_per_batch = 2,使每个pre_minibatch为2
    但再train_rpn里面将其修改为1， 所以cfg.FLAGS.ims_per_batch = 1
    也就是说其实传进来的roidb只有一个图片的信息，所以下面都用的roidb[0]
    '''
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32
    )
    '''
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    im_blob.shape[1]代表w, im_blob.shape[2]代表h, im_scales[0]代表缩放比例
    '''

    return blobs


def _get_image_blob(roidb, scale_inds):
    # 对roidb的图像进行缩放，并返回blob和缩放比例
    """builds an input blob from the images in the roidb at the specified scales"""
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image']) # 获取路径图片
        if roidb[i]['flipped']:
            # 如果之前翻转过，则水平翻转该图片
            im = im[:, ::-1, :]
        target_size = cfg.FLAGS2["scales"][scale_inds[i]]
        # cfg.FLAGS2.scales = (600,)没有多的，所有的target_size均为600
        im, im_scale = prep_im_for_blob(im, cfg.FLAGS2["pixel_means"], target_size, cfg.FLAGS.max_size)
        im_scales.append(im_scale)
        processed_ims.append(im)
        # 对图片进行缩放，保存缩放比例

    # create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    # 将缩放后的图片放入blob中

    return blob, im_scales














