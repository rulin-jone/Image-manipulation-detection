from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from detection import Window

CLASSES = ('__background__',
           'tampered')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_7400.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_dections(im, class_name, dets, thresh=0.5):
    # 画出检测到的bounding box
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} '.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        ax.set_title(('{} detections '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()


def demo(sess, net, image_name):


    # 加载目标图片
    im_file = os.path.join('test_images', image_name)
    im = cv2.imread(im_file)

    # 检测所有对象类并回归对象边界
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # 对每个检查的检测进行可视化
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # 因为跳过了背景background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_dections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    # 分析输入参数
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    f = open("path.txt", "r")  # 设置文件对象
    path = f.read()  # 将txt文件的所有内容读入到字符串str中
    path = path[:-1]
    f.close()  # 将文件关闭
    os.remove(r'path.txt')

    # print(path)

    args = parse_args()


    # 模型路径
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', 'DIY_dataset', 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError('{:s} not found.'.format(tfmodel + '.meta'))

    # 设置
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # 加载模型
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2, tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # for file in os.listdir("./test_images"):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            print('#########################################')
            print('Demo for lib/layer_utils/{}'.format(file))
            demo(sess, net, file)

    plt.show()












