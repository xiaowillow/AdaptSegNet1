from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
import cv2
from .imdb import imdb
from collections import OrderedDict
import util
import PIL.Image as Image
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class PASCAL_READ_MODES:
    #Returns list of DBImageItem each has the image and one object instance in the mask
    INSTANCE = 0
    #Returns list of DBImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    #Returns list of DBImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2

class pascal_voc_seg(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_ids = self._load_image_set_ids()

        self.read_mode = PASCAL_READ_MODES.SEMANTIC_ALL
        #self.read_mode = PASCAL_READ_MODES.SEMANTIC
        #self.get_seg_items()

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _remove_small_objects(self, items):
        filtered_item = []
        for item in items:
            m = np.array(Image.open(item['mask_path']))
            if util.change_coordinates(m, 8.0, 0.0).sum() > 2:  ##相当于对mask进行下采样到32倍，下采样后的大小变为(13, 17)，然后判断下采样的图片的像素值之和是否大于2
                filtered_item.append(item)  # 将大于2的图片放入filtered_item中,删掉小的图片
        return filtered_item  # 365个

    def get_seg_items(self, group, num_folds=4):
        pkl_file = os.path.join(self._data_path, 'cache', 'aaai_pascal_voc_seg_img_db.pkl')
        if os.path.exists(pkl_file):
            pkl_file = pkl_file.encode()
            with open(pkl_file,'rb') as f:
                #self.img_db = pickle.load(f,encoding='iso-8859-1')#1464
                self.img_db = pickle.load(f)
        else:
            self.img_db = self.getItems()

            if not os.path.exists(os.path.join(self._data_path, 'cache')):
                os.mkdir(os.path.join(self._data_path, 'cache'))
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.img_db, f)

        self.get_seg_items_single_clalss()#相当于将数据集中只有单个标签的图片提出来,共5813张,可分成20类
        self.get_seg_items_multiclalss(group, num_folds)##相当于将所有图片按照训练集合和测试集合进行分类,只有当图片的所有标签都在训练集中时才作为训练集,只有当图片的所有标签全部在测试类中才作为测试集

        print('Total images: %d'%(len(self.img_db)))



    def get_seg_items_single_clalss(self):
        self.single_img_db = self.filter_single_class_img(self.img_db)#self.single_img_db只含有单个标签的图片共有6746张,self.img_db对应所有的训练集10582张
        ######删掉mask比较小的目标
        self.single_img_db = self._remove_small_objects(self.single_img_db)
        print('Total images after filtering: %d'%(len(self.single_img_db)))
        self.grouped_imgs = self.group_images(self.single_img_db)#共20类,将同类放入一个list中

    def get_seg_items_multiclalss(self, group, num_folds):
        train_cats = self.get_cats('train', group, num_folds)# [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        val_cats = self.get_cats('val', group, num_folds)#[5, 6, 7, 8, 9]
        print('Train Categories:', train_cats)
        print('Val Categories:', val_cats)

        multiclass_grouped_imgs = self.group_images(self.img_db)#相当于总共10582张图片,总共20类,然后只要包含当前类的图片,就将其放在当前类,当图片中还有其他类时,也可以放在其他类,
        # for cat_id, key in enumerate(multiclass_grouped_imgs.keys()):
        #     print(len(multiclass_grouped_imgs[key]))
        self.multiclass_grouped_imgs = self.filter_multi_class_img(multiclass_grouped_imgs, train_cats, val_cats)##返回的是同一类的图片集合，当图片有多个标签时，需要判断图片的其他标签是否在训练集中，只有在的才放入该类中，也就是这里面的集合对应的图片可以包含多个类别，如果是单个类的分割，那么每张图片只有一个标签，而多类分割，则图片可以对应多个标签，获取多个类的group，#self.multiclass_grouped_imgs[rand_cat]
        #相当于将所有图片按照训练集合和测试集合进行分类,只有当图片的所有标签都在训练集中时才作为训练集,只有当图片的所有标签全部在测试类中才作为测试集
        # for cat_id, key in enumerate(self.multiclass_grouped_imgs.keys()):
        #     print('after filter:',len(self.multiclass_grouped_imgs[key]))


    def getItems(self):
        # items = OrderedDict()
        items = []
        for i in range(len(self._image_ids)):#共10582张图片
            # img_path = osp.join(self.db_path, 'JPEGImages', ann['image_name'] + '.jpg')
            item = {}
            #print(self._image_ids[i])
            item['img_id'] = self._image_ids[i]
            item['img_path'] = self.img_path_at(i)
            item['mask_path'] = self.mask_path_at(i)
            item['labels'] = self.get_labels(item['mask_path'])#得到对应图片的标签个数
            items.append(item)

        return items

    def mask_path_at(self, i):
        if self.read_mode == PASCAL_READ_MODES.INSTANCE:
            mask_path = os.path.join(self._data_path, 'SegmentationObject', self._image_ids[i]+'.png')
        else:
            mask_path = os.path.join(self._data_path, 'SegmentationClassAug', self._image_ids[i]+'.png')
            #mask_path = os.path.join(self._data_path, 'SegmentationClass', self._image_ids[i] + '.png')

        return mask_path

    def img_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_ids[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i


    def filter_multi_class_img(self, grouped_dict_list, train_cats, val_cats):
        grouped_imgs = OrderedDict()
        for key in grouped_dict_list.keys():
            grouped_imgs[key] = []

        for key in grouped_dict_list.keys():#对应每个类，共20类
            cat_list = grouped_dict_list[key]#获取第key类的图片信息,获取当前类的list
            for img_dict in cat_list:#获取每一类的每一张图片信息,包括img_id,img_path,mask_path,labels
                if key in set(train_cats):#判断该类是否在训练集中
                    labels = img_dict['labels']#[0,14]获取该图片的标签
                    if set(labels).issubset(train_cats):#图片有可能有两个标签，需要判断两个标签是否同时在训练类别中
                        grouped_imgs[key].append(img_dict)#如果该类图片在训练集中，则将图片放入grouped_imgs中
                elif key in set(val_cats):#当当前类不在训练集类别中时，判断是否在测试集中，
                    labels = img_dict['labels']#[1,7,15]
                    if set(labels).issubset(val_cats):#判断该图片的所有标签是否在测试集类别中
                        grouped_imgs[key].append(img_dict)#只有所有标签在测试集类中,才放入测试集中作为测试集

        return grouped_imgs#返回的是同一类的图片集合，当图片有多个标签时，需要判断图片的其他标签是否在训练集中，只有在的才放入该类中


    def filter_single_class_img(self, img_db):#得到单个类的图片，相当于找出图片中只有一个目标的图片，因为前面查看标签时错误，导致所有标签为0
        filtered_db = []
        for img_dict in img_db:
            if len(img_dict['labels']) == 1:
                filtered_db.append(img_dict)
        return filtered_db


    def group_images(self, img_db):
        '''
        Images of the same label cluster to one list有相同标签的图片聚类
        Images with multicalsses will be copyed to each class's list
        :return:
        :return:
        '''
        grouped_imgs = OrderedDict()
        for cls in self._classes:
            grouped_imgs[self._class_to_ind[cls]] = []
        for img_dict in img_db:
            for label in img_dict['labels']:
                grouped_imgs[label].append(img_dict)

        return grouped_imgs#共有20类,返回的是将具有相同类的图片放在一起

    def read_mask(self, mask_path):
        assert os.path.exists(mask_path), "%s does not exist"%(mask_path)
        mask = cv2.imread(mask_path)
        return mask

    def get_labels(self, mask_path):
        mask = self.read_mask(mask_path)#(281, 500, 3)
        labels = np.unique(mask)
        #print(labels)
        labels = [label-1 for label in labels if label != 255 and label != 0]
        #print('labels:',labels)
        return labels

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def get_cats(self, split, group,  num_folds):
        '''
          Returns a list of categories (for training/test) for a given fold number

          Inputs:
            split: specify train/val
            fold : fold number, out of num_folds
            num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

        '''
        num_cats = self.num_classes#20
        assert(num_cats%num_folds==0)
        val_size = int(num_cats/num_folds)#5
        assert(group<num_folds), 'group: %d, num_folds:%d'%(group, num_folds)
        val_set = [group*val_size+v for v in range(val_size)]# 验证集的类别[5, 6, 7, 8, 9]
        train_set = [x for x in range(num_cats) if x not in val_set]# [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        if split=='train':
            # print('Classes for training:'+ ','.join([self.classes[x] for x in train_set]))
            return train_set
        elif split=='val':
            # print('Classes for testing:'+ ','.join([self.classes[x] for x in train_set]))
            return val_set


    def split_id(self, path):
        return path.split()[0].split('/')[-1].split('.')[0]

    def _load_image_set_ids(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        #image_set_file = os.path.join(self._data_path, 'list',self._image_set + '.txt')
        image_set_file = os.path.join(self._data_path,  self._image_set + 'aug.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [self.split_id(x) for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

