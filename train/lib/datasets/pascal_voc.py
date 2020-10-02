from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
# import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse


from lib.config import config as cfg
from lib.datasets.imdb import imdb
# from .voc_eval import voc_eval






class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        # 初始化函数，对应着的是pascal_voc的数据集访问格式
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        # 继承类imdb的初始化函数__init__()，
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path

        print(self._devkit_path)

        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year) # 组合出数据集文件路径
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        # 数据集中所包含的全部object类别
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        # 构建字典{'__background__':'0','aeroplane':'1',
        # 'bicycle':'2', 'bird':'3', 'boat':'4','bottle':'5',
        # 'bus':'6', 'car':'7', 'cat':'8', 'chair':'9','cow':'10',
        # 'diningtable':'11', 'dog':'12', 'horse':'13','motorbike':'14',
        # 'person':'15', 'pottedplant':'16','sheep':'17', 'sofa':'18',
        # 'train':'19', 'tvmonitor':'20'}
        # self.classes（数据集类名列表）和self.num_classes（数据集类别个数）继承自imdb.py
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # 加载样本的list文件
        # 读取Main里txt记录的图像名字，建立图像数据集的索引列表，就是名字集合。
        self._seg_index = self._load_seg_set_index()
        # 读取分割图像的名，建立分割图像索引列表
        self._roidb_handler = self.gt_roidb
        # 这是一个调用gt_roidb()函数的句柄
        # 读取图像的ground-truth数据，但不读取roi，roi通过RPN来提取
        self._salt = str(uuid.uuid4())
        # 撒盐，使用uuid保证空间时间的唯一性
        self._comp_id = 'comp4'

        # # PASCAL specific config options
        # self.config = {'cleanup': True,
        #                'use_salt': True,
        #                'use_diff': False,
        #                'matlab_eval': False,
        #                'rpn_file': None
        #                }

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

        print(self.config)




        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        # 返回某一张图像的路径信息
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        # 从图像文件的名字集合中获取图片序号并组合成该图像的路径
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def seg_path_at(self, i):
        # 返回某一张seg的路径信息
        print('i\n', i)
        print('num of _seg_index\n', len(self._seg_index))
        return self.seg_path_from_index(self._seg_index[i])

    def seg_path_from_index(self, index):
        # 同image_path_from_index
        seg_path = os.path.join(self._data_path, 'SegmentationObject',
                                index + '.png')  # seg图像以png格式储存
        assert os.path.exists(seg_path), \
            'Path does not exist: {}'.format(seg_path)
        return seg_path


    def _load_image_set_index(self):
        # 读取Main里txt记录的图像名字，建立图像数据集的索引列表，就是名字集合。
        # load the indexes listed in this dataset's image set file
        # eg. path to image set file: self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt

        print(self._data_path)

        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        # 检查路径是否存在，否则抛出错误信息
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
            # 读内容（图片名字）放在image_index中，注意，这里的图片名没有后缀.jpg
        return image_index

    def _load_seg_set_index(self):
        # 同_load_image_set_index()
        # 读取分割图像的图像名字，建立分割图像数据的索引列表
        # load the indexes listed in this dataset's image set file
        # eg. path to image set file: self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Segmentation/trainval.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Segmentation',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            seg_index = [x.strip() for x in f.readlines()]
            # 读内容（图片名字）放在image_index中，注意，这里的图片名没有后缀.jpg
        return seg_index

    def _get_default_path(self):
        # 从config.py中获取默认的数据路径
        return os.path.join(cfg.FLAGS2["data_dir"], 'VOCdevkit' + self._year)

    def gt_roidb(self):
        # 读取并返回图片的ground-truth的db。将图片的gt加载进来
        # 图片gt可能已经被加载过并放在路径cache_file下，因此先生成一下路径
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            #如果路径存在，则可以直接加载存放图片gt的.pkl文件，能够提升加载速度
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        # 如果找不到自己预先生成的.pkl文件，那么就需要重新从文件中读取图片gt并保存在.pkl文件中以便下一次快速加载
        # pascal_voc图片的ground-truth信息保存在annotation文件夹下的XML文件中，因此先打开XML文件
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ground-truth to {}'.format(cache_file))
        return gt_roidb
    # ----------------------注意------------------------------
    # 这个函数的存在会直接加载先前的gt数据，并不会检查数据库是否被修改
    # 如果再次训练的时候修改了train数据库，增加或者删除了一些数据，再想重新训练的时候，一定要先删除之前的.pkl文件。
    # 否则，就会自动加载旧的pkl文件，而不会生成新的pkl文件。

    # def rpn_roidb(self):
    #     if int(self._year) == 2007 or self._image_set != 'test':
    #         gt_roidb = self.gt_roidb()
    #         rpn_roidb = self._load_rpn_roidb(gt_roidb)
    #         roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    #     else:
    #         roidb = self._load_rpn_roidb(None)
    #
    #     return roidb

    # def _load_rpn_roidb(self, gt_roidb):
    #     filename = self.config['rpn_file']
    #     print('loading {}'.format(filename))
    #     assert os.path.exists(filename), \
    #         'rpn data not found at: {}'.format(filename)
    #     with open(filename, 'rb') as f:
    #         box_list = pickle.load(f)
    #     return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # 要读取的xml文件路径eg. VOCdevkit2007/VOC2007/Annotations/000005.xml
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # 排除标记为difficult的样例
            non_diff_objs = [
                obj for obj in objs if int(obj.find('diffcult').text) == 0
            ]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs

        num_objs = len(objs)  # 输进来的图片上的object的个数

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            # 对于一张图片上的所有object
            bbox = obj.find('bndbox')
            # pascal_voc的XML文件给出的图片gt的形式是左上角和右下角的坐标
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            # 减1？ 因为voc数据坐标默认从0开始
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            # 找到该object的对应类别并转换成对应序号
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            # seg_areas计算该object的面积

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    # def _get_comp_id(self):
    #     comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
    #                else self._comp_id)
    #     return comp_id

    # def _get_voc_results_file_template(self):
    #     # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     path = os.path.join(
    #         self._devkit_path,
    #         'results',
    #         'VOC' + self._year,
    #         'Main',
    #         filename)
    #     return path

    # def _write_voc_results_file(self, all_boxes):
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print('Writing {} VOC results file'.format(cls))
    #         filename = self._get_voc_results_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in range(dets.shape[0]):
    #                     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                             format(index, dets[k, -1],
    #                                    dets[k, 0] + 1, dets[k, 1] + 1,
    #                                    dets[k, 2] + 1, dets[k, 3] + 1))

    # def _do_python_eval(self, output_dir='output'):
    #     annopath = self._devkit_path + '\\VOC' + self._year + '\\Annotations\\' + '{:s}.xml'
    #     imagesetfile = os.path.join(
    #         self._devkit_path,
    #         'VOC' + self._year,
    #         'ImageSets',
    #         'Main',
    #         self._image_set + '.txt')
    #     cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self._year) < 2010 else False
    #     print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    #     if not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(self._classes):
    #         if cls == '__background__':
    #             continue
    #         filename = self._get_voc_results_file_template().format(cls)
    #         rec, prec, ap = voc_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #             use_07_metric=use_07_metric)
    #         aps += [ap]
    #         print(('AP for {} = {:.4f}'.format(cls, ap)))
    #         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
    #             pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #     print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print(('{:.3f}'.format(ap)))
    #     print(('{:.3f}'.format(np.mean(aps))))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')

    # def _do_matlab_eval(self, output_dir='output'):
    #     print('-----------------------------------------------------')
    #     print('Computing results with the official MATLAB eval code.')
    #     print('-----------------------------------------------------')
    #     path = os.path.join(cfg.FLAGS2["root_dir"], 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #         .format(self._devkit_path, self._get_comp_id(),
    #                 self._image_set, output_dir)
    #     print(('Running:\n{}'.format(cmd)))
    #     status = subprocess.call(cmd, shell=True)

    # def evaluate_detections(self, all_boxes, output_dir):
    #     self._write_voc_results_file(all_boxes)
    #     self._do_python_eval(output_dir)
    #     if self.config['matlab_eval']:
    #         self._do_matlab_eval(output_dir)
    #     if self.config['cleanup']:
    #         for cls in self._classes:
    #             if cls == '__background__':
    #                 continue
    #             filename = self._get_voc_results_file_template().format(cls)
    #             os.remove(filename)

    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True






















