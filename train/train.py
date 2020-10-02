import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
import lib.config.config as cfg
from lib.datasets import roidb as rd1_roidb
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

# try:
#     import cPickle as pickle
# except ImportError:
import pickle
import os

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""

    if True:
        print('Appending horizontally-filpped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rd1_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

def combined_roidb(imdb_names):
    """Combine multiple roidbs"""
    # 融合roidb，roidb来自于数据集（实验可能用到多个），所以需要combine多个数据集的roidb

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)

        print('imdb*******/n', imdb)

        print('Loaded dataset `{:s}` for training'.format(imdb_name))
        imdb.set_proposal_method("gt")
        # 设置proposal方法
        print('Set proposal method: gt')
        roidb = get_training_roidb(imdb)
        # 得到用于训练的roidb，定义在train.py，进行了水平翻转，以及为原始roidb添加了一些说明性的属性
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        # 如果大于一个，则进行combine roidb
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
        # get_imdb方法定义在factory.py中，通过名字得到imdb
    return imdb, roidb


class Train:
    def __init__(self):
        # create network
        # 初始化，准备网络与数据载入
        if cfg.FLAGS.net == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        else:
            raise NotImplementedError

        self.imdb, self.roidb = combined_roidb("DIY_dataset")

        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.output_dir = cfg.get_output_dir(self.imdb, 'default')

    def train(self):

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess = tf.Session(config=tfconfig)
        # 创建一个session并对其进行配置
        # allow_soft_placement允许自动分配GPU，allow_growth允许慢慢增加分配的内存

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)  # 固定随机数种子
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag='default')
            # 展开网络  create_architecture在network.py中被定义
            loss = layers['total_loss']

            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)
            # 通过tf.Variable()申请一个常驻内存的量，作为learning_rate
            momentum = cfg.FLAGS.momentum
            optimizer = tf.train.MomentumOptimizer(lr, momentum)
            # 使用动量梯度下降优化器

            gvs = optimizer.compute_gradients(loss)  # 对loss进行优化

            # double bias
            # 通过cfg.FLAGS.double_bias进行控制
            # 加倍gradient
            if cfg.FLAGS.double_bias:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            # 自己处理snapshots保存模型参数
            self.saver = tf.train.Saver(max_to_keep=100000)
            # 向tensorboard中写入训练和检验信息
            writer = tf.summary.FileWriter('default/', sess.graph)
            # valwriter = tf.summary.FileWriter(self.tbvaldir)

        # 加载权重
        # 直接从ImageNet weights 中更新训练
        # 加载预训练模型vgg16，路径cfg.FLAGS.pretrained_model
        print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
        variables = tf.global_variables()
        # 获取全部学习变量，以便进行初始化

        # print('###################')
        # print(variables)
        # print('###################')

        sess.run(tf.variables_initializer(variables, name='init'))
        # 初始化变量
        var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)
        # Get the variables to restore, ignorizing the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic, sess,
                                                                 cfg.FLAGS.pretrained_model)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, cfg.FLAGS.pretrained_model)
        print('Loaded.')
        # 需要在加载前fix variables，以便将RGB数据型的权重转变成BGR表示
        # 同样对vgg16网络中fc6和fc7的权重也进行了改变
        # 全连接层权重
        self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
        print('Fixed.')
        sess.run(tf.assign(lr, cfg.FLAGS.learning_rate))
        # 随着训练的进行，将cfg.FLAGS.learning_rate的值赋给lr
        last_snapshot_iter = 0

        timer = Timer()
        # 添加一个计时器，计算训练时间
        iter = last_snapshot_iter + 1
        # last_summary_time = time.time()
        print('****************start training*****************')
        while iter < cfg.FLAGS.max_iters + 1:
            # learning rate
            if iter == cfg.FLAGS.step_size + 1:
                # 在更新learning rate之前，保存模型snapshot
                # self.snapshot(sess, iter)
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma))


            timer.tic()
            # 获取训练数据，一次获取一个batch
            blobs = self.data_layer.forward()
            iter += 1
            # 在没有summary的情况下计算图
            if iter % 100 == 0:
                # 没100次保存一下
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = self.net.train_step_with_summary(
                    sess, blobs, train_op)
                timer.toc()

                run_metadata = tf.RunMetadata()
                writer.add_run_metadata(run_metadata, 'step%03d' % iter)
                writer.add_summary(summary, iter)
            else:
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(
                    sess, blobs, train_op)
                timer.toc()

            # 在每个cfg.FLAGS.display处进行print，展示训练loss等
            if iter % (cfg.FLAGS.display) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                      (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if iter % cfg.FLAGS.snapshot_iterations == 0:
                # 在cfg.FLAGS.snapshot_iterations处保存snapshot
                self.snapshot(sess, iter)

    def get_variables_in_checkpoint_file(self, file_name):
        # 读取与训练模型
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            # 如果预训练文件是压缩状态，抛出错误
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")



    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            # 检查output路径
            os.makedirs(self.output_dir)

        # 保存模型
        filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('write snapshot to: {:s}'.format(filename))

        # 保存 meta information, random state等数据
        nfilename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # 当前np随机数current state of numpy random
        st0 = np.random.get_state()
        # 当前layer
        cur = self.data_layer._cur
        # 数据库的当前无序索引shuffled indeces of the database
        perm = self.data_layer._perm

        # 保存数据
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename


















if __name__ == '__main__':
    train = Train()
    train.train()