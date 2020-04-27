import os
from PIL import Image
import numpy as np
from random import randint
from shutil import copyfile
import xml.etree.ElementTree as ET

from lib.factory import get_imdb
from lib.xml_op import *


DATASET_SIZE = 1
# 定义生成的训练集的大小，即要生成的新数据集图片的数量

dataset_path = os.sep.join(['data', 'VOCDevkit2007', 'VOC2007'])
images_path = os.sep.join([dataset_path, 'JPEGImages'])
image_annotation_path = os.sep.join([dataset_path, 'Annotations'])

save_path = os.sep.join(['data', 'DIY_dataset', 'VOC2007'])
save_image_path = os.sep.join([save_path, 'JPEGImages'])
save_annotation_path = os.sep.join([save_path, 'Annotations'])
# 定义读取的VOC数据集路径与生成的数据集保存路径

imdb = get_imdb("voc_2007_trainval")
print('imdb', imdb)
roidb = imdb.roidb

image_index = imdb._load_image_set_index()
seg_index = imdb._load_seg_set_index()

def generate_seg_img_map():
    # 将分割图像seg与原图像进行配对
    # 即将sge_index中的名字序号与image_index中的名字序号保存在map{}中
    # eg. seg_index=['003','001','002'] image_index=['002','003','001']则map=[1,2,0]
    map = {}
    idx1 = 0
    for i in seg_index:
        idx2 = 0
        for j in image_index:
            if i == j:
                map[idx1] = idx2
            idx2 += 1
        idx1 += 1
    return map

def random_seg_idx():
    # 随机选一张seg（某一图像对应的seg，eg:000032.jpg，是image文件夹下000032图像对应的seg图像）
    return randint(0, len(seg_index)-1)

def random_obj_idx(s):
    # 从选择的seg中随机选一个obj用于抠图备用
    return randint(1, len(s)-2)

def random_obj_loc(img_h, img_w, obj_h, obj_w):
    # 生成目标图片中的一个随机位置
    # 以左下角顶点为标准
    return randint(0, img_h - obj_h), randint(0, img_w - obj_w)

def find_obj_vertex(mask):
    # 确定box的四个顶点
    hor = np.where(np.sum(mask, axis=0) > 0)
    ver = np.where(np.sum(mask, axis=1) > 0)
    return hor[0][0], hor[0][-1], ver[0][0], ver[0][-1]

def modify_xml(filename, savefile, xmin, ymin, xmax, ymax):
    # 定义一个XML初始化函数，用来保存生成的图像的图像信息
    def create_node(tag, property_map, content):
        element = Element(tag, property_map)
        element.text = content
        return element
    copyfile(filename, savefile)
    # 复制要被PS的图片的XML信息到savefile路径下
    tree = ET.parse(savefile)
    root = tree.getroot()
    # 打开刚创建的savlfile下的XML文件进行修改
    for obj in root.findall('object'):
        root.remove(obj)
        # 将新建的XML文件中原本的obj信息全部删除，为下一步存入新的obj对象信息做准备
    new_obj = Element('object', {})
    new_obj.append(create_node('name', {}, 'tampered'))
    bndbox = Element('bndbox', {})
    bndbox.append(create_node('xmin', {}, str(xmin)))
    bndbox.append(create_node('ymin', {}, str(ymin)))
    bndbox.append(create_node('xmax', {}, str(xmax)))
    bndbox.append(create_node('ymax', {}, str(ymax)))
    new_obj.append(bndbox)
    root.append(new_obj)
    tree.write(savefile)
    # 往XML文件中写入被粘贴过来的obj的顶点信息


if __name__ == '__main__':
    map = generate_seg_img_map()
    count = 0
    while count < DATASET_SIZE:
        if count % 100 == 0:
            print('>>> %d / %d' % (count, DATASET_SIZE))
        img_idx = count % len(image_index)
        print('img_idx\n', img_idx)
        print('num of image_index\n', len(image_index))
        # 按照序号顺序取待PS图片
        # 如果要生成的数据集图片数量大于原本VOC中的图片数量那就再来一轮
        seg_idx = random_seg_idx()
        # 随机选取一张seg图片
        img = Image.open(imdb.image_path_at(img_idx))
        # 打开图像img_idx（base image）
        seg = Image.open(imdb.seg_path_at(seg_idx)).convert('P')
        print('seg_array\n', seg)
        # 打开分割图像seg_idx（这张seg图像是随机选取的，用于抠图）
        # 使用convert('P')转化成8bit彩色，即每个像素用8位，0-255表示
        seg_img = Image.open(imdb.image_path_at(map[seg_idx]))
        # 打开分割图像seg_idx对应的原始图像seg_img

        seg_np = np.asarray(seg)
        obj_idx = random_obj_idx(set(seg_np.flatten()))
        # randomly pic an obj from seg img
        # 随机的从seg图像中选取一个obj
        # 因为seg中用一种颜色代表一个obj，所以只需要随机选取一种颜色就行
        # 首先flatten降为一维，通过set()去除重复元素，即只剩下了几个代表不同颜色的obj的数字，然后在这几个数字中随机一个，就能随机的选择一个obj
        mask2 = (seg_np == obj_idx)
        min_x, max_x, min_y, max_y = find_obj_vertex(mask2)
        # 确定mask的顶点
        loop_counter = 0
        while(max_x - min_x) * (max_y - min_y) < img.size[0] * img.size[1] * 0.005 or \
                (max_x - max_y) * (max_y - min_y) >img.size[0] * img.size[1] *0.3 or \
                max_x - min_x >= img.size[0] or max_y - min_y >= img.size[1] or loop_counter > 1000:
            loop_counter +=1
            seg_idx = random_seg_idx()
            seg = Image.open(imdb.seg_path_at(seg_idx)).convert('P')
            seg_img = Image.open(imdb.image_path_at(map[seg_idx]))
            seg_np = np.asarray(seg)
            obj_idx = random_obj_idx(set(seg_np.flatten()))
            mask2 = (seg_np == obj_idx)
            min_x, max_x, min_y, max_y = find_obj_vertex(mask2)
        # 如果粘贴的obj过大或过小都不好，所以排除这种情况的obj，重新随机一个seg和obj
        if loop_counter > 1000:
            continue
        mask2 = mask2[min_y:max_y, min_x:max_x]
        mask = np.stack((mask2, mask2, mask2), axis=2)
        seg_img_np = np.asarray(seg_img).copy()[min_y:max_y, min_x:max_x, :]
        # 复制seg_img中对应seg_obj的图像，即seg_img中将要被抠图并粘贴到img中的部分
        img_np = np.asarray(img).copy()
        loc_y, loc_x = random_obj_loc(img.size[1], img.size[0], max_y - min_y, max_x - min_x)
        img_np[loc_y: loc_y + max_y - min_y, loc_x: loc_x + max_x - min_x, :] = img_np[loc_y:loc_y+max_y - min_y, loc_x:loc_x+max_x - min_x, :] * (1-mask) + seg_img_np * mask
        new_img = Image.fromarray(img_np, mode='RGB')
        # 将抠图部分粘贴到img上，并以RGB格式输出图片

        new_img.save(os.sep.join([save_image_path,image_index[img_idx] + '.jpg']))
        modify_xml(os.sep.join([image_annotation_path, image_index[img_idx] + '.xml']),
                   os.sep.join([save_annotation_path, image_index[img_idx] + '.xml']),
                   loc_x+1, loc_y+1, loc_x+max_x - min_x, loc_y+max_y - min_y)
        count += 1



