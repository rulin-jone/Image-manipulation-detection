"""blob helper function"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def im_list_to_blob(ims):
    # convert a list of images into a network input
    # 输入的图片需要满足要求（通过prep_im_for_blob()函数的预处理）
    # 此函数将缩放后的图片信息存到blob中,该blob可能右边与下边值为0，因为存放时blob大小按照ims中最大的图片进行设置，小的图片进行了零填充
    # ims是缩放后的图片列表
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # 取出该batch中所有缩放后图片的长宽最大值
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    # 构建一个全0 array，3代表RGB通道
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # 存图片
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    # 对图像进行缩放，返回缩放后的image以及缩放比例
    # pixel_means, cfg.PIXEL_MEANS为np.array([[[102.9801, 115.9465, 122.7717]]])
    # target_size 为5个缩放比例的随机一个
    # max_size, cfg.TRAIN.MAX_SIZE为1000
    """mean subtract and scale an image for use in a blob"""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    # 减去3通道的平均值
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # 比较长宽获得最大最小值
    im_scale = float(target_size) / float(im_size_min)
    # 缩放比例im_scale,距离目标尺寸的比例
    # prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    # 对im进行缩放，缩放比例为im_scale

    return im, im_scale


