
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh
    # 从config文件读取配置
    # post_nms_topN（执行NMS算法后proposal的数量）
    # nms_thresh（NMS阈值）



    # 学习参数：rpn_bbox_pred
    # 原始anchor给出的proposal通过学习参数转换为与gt接近的边框，裁剪掉超出图片的部分

    im_info = im_info[0]
    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # bbox_transform_inv:
    # 每个anchor的边框学习之前得到的偏移量（这里的偏移量就是需要学习的rpn_bbox_pred）做位移和缩放，获取最终的预测边框
    # 也就是原始proposal A，通过学习rpn_bbox_pred中的参数，得到一个与ground truth G相近的预测边框G'
    proposals = clip_boxes(proposals, im_info[:2])
    # clip_boxes:
    # 裁剪掉超出原始图片边框的部分

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    # 执行NMS算法，获取最终的proposals
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
