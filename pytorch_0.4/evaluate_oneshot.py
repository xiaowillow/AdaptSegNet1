import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG

from modeling.deeplab import DeepLab
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from ss_datalayer import SSDatalayer
from tqdm import tqdm
import torch.nn as nn
from utils.save_mask import mask_to_img
import cv2
from utils.save_atten import SAVE_ATTEN
from utils.segscorer import SegScorer
from utils import Metrics
from model.discriminator import FCDiscriminator
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 2
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM ='/home/prlab/willow/Segmentation/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_1_100000.pth'
D1_RESTORE_FROM ='/home/prlab/willow/Oneshot/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_1_100000_D1.pth'
#D2_RESTORE_FROM ='/home/prlab/willow/Oneshot/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_ADV25000_D2.pth'
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
#RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
RESTORE_FROM_ORC = '/home/prlab/willow/Segmentation/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_1_100000.pth'

SET = 'val'

MODEL = 'Oracle'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def get_org_img(img):
    img = np.transpose(img, (1,2,0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img*std_vals + mean_vals
    img = img*255
    return img

def measure(y_in, pred_in):
    # thresh = .5
    thresh = .5
    y = y_in>thresh
    pred = pred_in>thresh
    tp = np.logical_and(y,pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from1", type=str, default=D1_RESTORE_FROM,
                        help="Where restore model parameters from.")

    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")

    parser.add_argument("--arch", type=str, default='onemodel_sg_one_resnet')
    parser.add_argument("--disp_interval", type=int, default=100)
    #parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=100000)

    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        #model = Res_Deeplab(num_classes=args.num_classes)
        model = DeepLab(backbone='resnet', output_stride=8)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)
    model.eval()

    num_classes = 20
    tp_list = [0] * num_classes
    fp_list = [0] * num_classes
    fn_list = [0] * num_classes
    iou_list = [0] * num_classes

    hist = np.zeros((21, 21))
    group = 1
    scorer = SegScorer(num_classes=21)
    datalayer = SSDatalayer(group)
    cos_similarity_func = nn.CosineSimilarity()
    for count in tqdm(range(1000)):
        dat = datalayer.dequeue()
        ref_img = dat['second_img'][0]  # (3, 457, 500)
        query_img = dat['first_img'][0]  # (3, 375, 500)
        query_label = dat['second_label'][0]  # (1, 375, 500)
        ref_label = dat['first_label'][0]  # (1, 457, 500)
        # query_img = dat['second_img'][0]
        # ref_img = dat['first_img'][0]
        # ref_label = dat['second_label'][0]
        # query_label = dat['first_label'][0]
        deploy_info = dat['deploy_info']
        semantic_label = deploy_info['first_semantic_labels'][0][0] - 1  # 2

        ref_img, ref_label = torch.Tensor(ref_img).cuda(), torch.Tensor(ref_label).cuda()
        query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0, :, :]).cuda()
        #ref_img, ref_label = torch.Tensor(ref_img), torch.Tensor(ref_label)
        #query_img, query_label = torch.Tensor(query_img), torch.Tensor(query_label[0, :, :])

        # ref_img = ref_img*ref_label
        ref_img_var, query_img_var = Variable(ref_img), Variable(query_img)
        query_label_var, ref_label_var = Variable(query_label), Variable(ref_label)

        ref_img_var = torch.unsqueeze(ref_img_var, dim=0)  # [1, 3, 457, 500]
        ref_label_var = torch.unsqueeze(ref_label_var, dim=1)  # [1, 1, 457, 500]
        query_img_var = torch.unsqueeze(query_img_var, dim=0)  # [1, 3, 375, 500]
        query_label_var = torch.unsqueeze(query_label_var, dim=0)  # [1, 375, 500]

        samples = torch.cat([ref_img_var,query_img_var], 0)
        pred = model(samples, ref_label_var)
        w, h = query_label.size()
        pred = F.upsample(pred, size=(w, h), mode='bilinear')#[2, 416, 416]
        pred = F.softmax(pred, dim=1).squeeze()
        values, pred = torch.max(pred, dim=0)
        #print(pred.shape)
        pred = pred.data.cpu().numpy().astype(np.int32)  # (333, 500)
        #print(pred.shape)
        org_img = get_org_img(query_img.squeeze().cpu().data.numpy())  # 查询集的图片(375, 500, 3)
        #print(org_img.shape)
        img = mask_to_img(pred, org_img)  # (375, 500, 3)mask和原图加权后的彩色图片
        cv2.imwrite('save_bins/que_pred/query_set_1_%d.png' % (count), img)

        query_label = query_label.cpu().numpy().astype(np.int32)  # (333, 500)
        class_ind = int(deploy_info['first_semantic_labels'][0][0]) - 1  # because class indices from 1 in data layer,0
        scorer.update(pred, query_label, class_ind + 1)
        tp, tn, fp, fn = measure(query_label, pred)
        # iou_img = tp/float(max(tn+fp+fn,1))
        tp_list[class_ind] += tp
        fp_list[class_ind] += fp
        fn_list[class_ind] += fn
        # max in case both pred and label are zero
        iou_list = [tp_list[ic] /
                    float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1))
                    for ic in range(num_classes)]

        tmp_pred = pred
        tmp_pred[tmp_pred > 0.5] = class_ind + 1
        tmp_gt_label = query_label
        tmp_gt_label[tmp_gt_label > 0.5] = class_ind + 1

        hist += Metrics.fast_hist(tmp_pred, query_label, 21)

    print("-------------GROUP %d-------------" % (group))
    print(iou_list)
    class_indexes = range(group * 5, (group + 1) * 5)
    print('Mean:', np.mean(np.take(iou_list, class_indexes)))

    '''
    for group in range(2):
        datalayer = SSDatalayer(group+1)
        restore(args, model, group+1)

        for count in tqdm(range(1000)):
            dat = datalayer.dequeue()
            ref_img = dat['second_img'][0]#(3, 457, 500)
            query_img = dat['first_img'][0]#(3, 375, 500)
            query_label = dat['second_label'][0]#(1, 375, 500)
            ref_label = dat['first_label'][0]#(1, 457, 500)
            # query_img = dat['second_img'][0]
            # ref_img = dat['first_img'][0]
            # ref_label = dat['second_label'][0]
            # query_label = dat['first_label'][0]
            deploy_info = dat['deploy_info']
            semantic_label = deploy_info['first_semantic_labels'][0][0] - 1#2

            ref_img, ref_label = torch.Tensor(ref_img).cuda(), torch.Tensor(ref_label).cuda()
            query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0,:,:]).cuda()
            #ref_img, ref_label = torch.Tensor(ref_img), torch.Tensor(ref_label)
            #query_img, query_label = torch.Tensor(query_img), torch.Tensor(query_label[0, :, :])

            # ref_img = ref_img*ref_label
            ref_img_var, query_img_var = Variable(ref_img), Variable(query_img)
            query_label_var, ref_label_var = Variable(query_label), Variable(ref_label)

            ref_img_var = torch.unsqueeze(ref_img_var,dim=0)#[1, 3, 457, 500]
            ref_label_var = torch.unsqueeze(ref_label_var, dim=1)#[1, 1, 457, 500]
            query_img_var = torch.unsqueeze(query_img_var, dim=0)#[1, 3, 375, 500]
            query_label_var = torch.unsqueeze(query_label_var, dim=0)#[1, 375, 500]

            logits  = model(query_img_var, ref_img_var, ref_label_var,ref_label_var)

            # w, h = query_label.size()
            # outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
            # out_side = F.softmax(outB_side, dim=1).squeeze()
            # values, pred = torch.max(out_side, dim=0)
            values, pred = model.get_pred(logits, query_img_var)#values[2, 333, 500]
            pred = pred.data.cpu().numpy().astype(np.int32)#(333, 500)

            query_label = query_label.cpu().numpy().astype(np.int32)#(333, 500)
            class_ind = int(deploy_info['first_semantic_labels'][0][0])-1 # because class indices from 1 in data layer,0
            scorer.update(pred, query_label, class_ind+1)
            tp, tn, fp, fn = measure(query_label, pred)
            # iou_img = tp/float(max(tn+fp+fn,1))
            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn
            # max in case both pred and label are zero
            iou_list = [tp_list[ic] /
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1))
                        for ic in range(num_classes)]


            tmp_pred = pred
            tmp_pred[tmp_pred>0.5] = class_ind+1
            tmp_gt_label = query_label
            tmp_gt_label[tmp_gt_label>0.5] = class_ind+1

            hist += Metrics.fast_hist(tmp_pred, query_label, 21)


        print("-------------GROUP %d-------------"%(group))
        print(iou_list)
        class_indexes = range(group*5, (group+1)*5)
        print('Mean:', np.mean(np.take(iou_list, class_indexes)))

    print('BMVC IOU', np.mean(np.take(iou_list, range(0,20))))

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))
    '''

    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(), hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)

    scores = scorer.score()
    for k in scores.keys():
        print(k, np.mean(scores[k]), scores[k])


if __name__ == '__main__':
    main()
