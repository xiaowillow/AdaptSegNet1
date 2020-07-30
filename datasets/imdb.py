# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import pdb

np.random.seed(1234)

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        # self.group=0
        # self.num_folds= 4
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb

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

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')   # method=self.gt_roidb
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()  # call self.gt_roidb()
        return self._roidb

    # @property
    # def cache_path(self):
    #     cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    #     if not os.path.exists(cache_path):
    #         os.makedirs(cache_path)
    #     return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def image_id_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def get_pair_images(self):
        self.group=0
        self.num_folds= 4
        cats = self.get_cats(self.split, self.fold)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        sample_img_ids = np.random.choice(len(self.grouped_imgs[rand_cat]), 2, replace=False)
        return (self.grouped_imgs[rand_cat][sample_img_ids[0]],
                self.grouped_imgs[rand_cat][sample_img_ids[1]])

    def get_triple_images(self, split, group, num_folds=4):#得到三对图片
        cats = self.get_cats(split, group, num_folds)#目的是得到对应split和group的所含数据集的类别
        rand_cat = np.random.choice(cats, 2, replace=False)#从所含类中选择两类
        #print(len(self.grouped_imgs[18]))
        sample_img_ids_1 = np.random.choice(len(self.grouped_imgs[rand_cat[0]]), 2, replace=False)#从第一类中，随机选择两张图片
        sample_img_ids_2 = np.random.choice(len(self.grouped_imgs[rand_cat[1]]), 1, replace=False)#从第二类中，随机选择一张图片

        anchor_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[0]]#将所选取的第一类的第一张图片作为查询集
        pos_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[1]]#第而这一类的第二张图片作为正样本，因为该图片和上面的图片属于同一类
        neg_img = self.grouped_imgs[rand_cat[1]][sample_img_ids_2[0]]#第二类图片作为负样本，因为跟上面两张图片的类别不一样

        return (anchor_img, pos_img, neg_img)

    def get_multiclass_train(self, split, group, num_folds=4):#针对多个类
        cats = self.get_cats('train', group, num_folds)#获取group对应的训练集类别
        rand_cat = np.random.choice(cats, 1, replace=False)[0]#从其中随机选择一个类
        cat_list = self.multiclass_grouped_imgs[rand_cat]#获取的是对应rand_cat类的多标签集合，这里面每张图片都有对应的rand_cat类，但也可能包含训练集的其他类别
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)#从上面的类别的图片中随机选择两张图片
        query_img = cat_list[sample_img_ids_1[0]]#一张作为查询集
        support_img = cat_list[sample_img_ids_1[1]]#一张作为支持集
        return query_img, support_img, rand_cat

    def get_multiclass_val(self, split, group, num_folds=4):
        cats = self.get_cats('val', group, num_folds)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]#4
        cat_list = self.multiclass_grouped_imgs[rand_cat]
        sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]#从第4类中随机选择一张图片的ids
        query_img = cat_list[sample_img_ids_1]#将该图片作为查询集
        sup_img_list=[]
        for cat_id in cats:#[0,1,2,3,4]
            cat_list = self.grouped_imgs[cat_id]#得到当前类的所有图片
            sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]#随机在该类中选择一张
            img_dict = cat_list[sample_img_ids_1]
            sup_img_list.append(img_dict)#将该张作为支持集，因为是多类，由于测试集部分也有5类，所以每一类选择一张图片，共5类图片，对应5张支持集
        return (query_img, sup_img_list)


    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

