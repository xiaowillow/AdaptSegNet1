from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
import random
# from .transforms import functional
# random.seed(1234)
from .transforms import functional
import cv2
import math
#from .transforms import transforms
from torch.utils.data import DataLoader
from utils.config import cfg
from datasets.factory import get_imdb
import torchvision.transforms.functional as tr_F
class mydataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, args, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_db = get_imdb('voc_2012_train')
        self.img_db.get_seg_items(args.group, args.num_folds)
        self.transform = transform
        self.split = args.split
        self.group = args.group
        self.num_folds = args.num_folds
        self.is_train = is_train


    def __len__(self):
        # return len(self.image_list)
        return 100000000

    def read_img(self, path):
        return cv2.imread(path)

    def _read_data(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']
        instance_mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat>0] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)#返回的是图片及mask

    def _read_mlclass_val(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat!=category+1] = 0
        mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def get_item_mlclass_val(self, query_img, sup_img_list):
        que_img, que_mask = self._read_mlclass_val(query_img)
        supp_img = []
        supp_mask = []
        for img_dit in sup_img_list:
            tmp_img, tmp_mask = self._read_mlclass_val(img_dit)
            supp_img.append(tmp_img)
            supp_mask.append(tmp_mask)

        supp_img_processed = []
        if self.transform is not None:
            que_img = self.transform(que_img)
            for img in supp_img:
                supp_img_processed.append(self.transform(img))

        return que_img, que_mask, supp_img_processed, supp_mask

    def get_item_mlclass_train(self, query_img, support_img, category):
        que_img, que_mask = self._read_mlclass_train(query_img, category)
        supp_img, supp_mask = self._read_mlclass_train(support_img, category)
        if self.transform is not None:
            que_img = self.transform(que_img)
            supp_img = self.transform(supp_img)

        return que_img, que_mask, supp_img, supp_mask

    def get_item_single_train(self,dat_dicts):#
        #print(dat_dicts[0]['img_id'])
        #print(dat_dicts[0])
        first_img, first_mask = self._read_data(dat_dicts[0])#基准图片
        second_img, second_mask = self._read_data(dat_dicts[1])#正样本图片，和基准图片属于同一类
        third_img, third_mask = self._read_data(dat_dicts[2])#负样本图片，与基准类别不一样

        #print(first_img)
        if self.transform is not None:
            first_img = self.transform(functional.to_pil_image(first_img))
            second_img = self.transform(functional.to_pil_image(second_img))
            third_img = self.transform(functional.to_pil_image(third_img))
            first_mask = tr_F.resize(functional.to_pil_image(first_mask.reshape(first_mask.shape[0],
                                                                                first_mask.shape[1], 1)), [416, 416],
                                     interpolation=Image.NEAREST)
            second_mask = tr_F.resize(functional.to_pil_image(second_mask.reshape(second_mask.shape[0],
                                                                                  second_mask.shape[1], 1)), [416, 416],
                                      interpolation=Image.NEAREST)
            third_mask = tr_F.resize(functional.to_pil_image(third_mask.reshape(third_mask.shape[0],
                                                                                third_mask.shape[1], 1)), [416, 416],
                                     interpolation=Image.NEAREST)
        ##############根据前景和实例分割构造boundingbox

        #########################
        return first_img, np.array(first_mask).astype(np.float32), second_img, np.array(second_mask).astype(np.float32), third_img, np.array(third_mask).astype(np.float32)

    def get_item_rand_val(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        third_img, third_mask = self._read_data(dat_dicts[2])

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            third_img = self.transform(third_img)

        # return first_img, first_mask, second_img,second_mask
        return first_img, first_mask, second_img,second_mask, third_img, third_mask, dat_dicts

    def __getitem__(self, idx):
        if self.split == 'train':
            dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)#返回的是从split和group中对应数据集，随机选择两类图片，(anchor_img基准, pos_img正样本（和基准的类别一样）, neg_img（负样本，和基准图片不一样）)
            return self.get_item_single_train(dat_dicts)#如果split=train，则调用单个train函数，返回的是3张图片以及对应的mask
        elif self.split == 'random_val':
            dat_dicts = self.img_db.get_triple_images(split='val', group=self.group, num_folds=4)
            return self.get_item_rand_val(dat_dicts)#对应的是随机验证
        elif self.split == 'mlclass_val':#多个类
            query_img, sup_img_list = self.img_db.get_multiclass_val(split='val', group=self.group, num_folds=4)
            return self.get_item_mlclass_val(query_img, sup_img_list)
        elif self.split == 'mlclass_train':#多个类的训练
            query_img, support_img, category = self.img_db.get_multiclass_train(split='train', group=self.group, num_folds=4)#返回的是从某个类别category中随机选取的两张图片，这两种图片可能还有其他标签，但是其他标签都在当前训练类别中
            return self.get_item_mlclass_train(query_img, support_img, category)#获取对应图片以及mask


    # def __getitem__(self, idx):
    #     if self.split == 'train':
    #         dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
    #
    #     first_img, first_mask = self._read_data(dat_dicts[0])
    #     second_img, second_mask = self._read_data(dat_dicts[1])
    #     thrid_img, thrid_mask = self._read_data(dat_dicts[2])
    #
    #     if self.transform is not None:
    #         first_img = self.transform(first_img)
    #         second_img = self.transform(second_img)
    #         thrid_img = self.transform(thrid_img)
    #
    #     return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask


