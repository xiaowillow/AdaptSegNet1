import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter

from model.deeplab_multi import DeeplabMulti
from model.deeplab import Res_Deeplab
from modeling.deeplab import DeepLab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from utils.LoadDataSeg import data_loader
from collections import OrderedDict
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
#RESTORE_FROM ='https://download.pytorch.org/models/resnet50-19c8e357.pth'
RESTORE_FROM ='./snapshots/voc2012_1_50000.pth'
RESTORE_FROM_D ='./snapshots/voc2012_1_50000_D1.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.001
LAMBDA_ADV_TARGET2 = 0.001
#LAMBDA_ADV_TARGET1 = 0.0002
#LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")

    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=RESTORE_FROM_D,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--arch", type=str, default='onemodel_sg_one')
    parser.add_argument("--max_steps", type=int, default=100001)
    parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    #parser.add_argument("--restore_from", type=str,default='/home/prlab/willow/Oneshot/SG-One-master/snapshots/onemodel_sg_one_scale/group_1_of_4/step_80000.pth.tar')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=50000)

    # parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=4)
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
def restore(model, pre_model):
    try:
        model.load_state_dict(pre_model)
    except RuntimeError:
        print("KeyError")
        model_dict = model.state_dict()
        new_model_dict = OrderedDict()
        for k_model, k_pretrained in zip(list(model_dict.keys()), list(pre_model.keys())[:len(model_dict.keys())]):#增加了list
            if model_dict[k_model].size() == pre_model[k_pretrained].size():
                print("%s\t<--->\t%s"%(k_model, k_pretrained))
                new_model_dict[k_model] = pre_model[k_pretrained]
            else:
                print('Fail to load %s'%(k_model))

        model_dict.update(new_model_dict)
        model.load_state_dict(model_dict)
    except KeyError:
        print("Loading pre-trained values failed.")
        raise
# else:
#     print("=> no checkpoint found at '{}'".format(snapshot))

def one_hot(label):
    label = label.squeeze().unsqueeze(0).cpu().numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        #model = DeeplabMulti(num_classes=args.num_classes)
        #model = Res_Deeplab(num_classes=args.num_classes)
        model = DeepLab(backbone='resnet', output_stride=16)
        '''
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        #restore(model, saved_state_dict)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not i_parts[0] == 'layer4' and not i_parts[0] == 'fc':
                #new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                new_params[i] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)
        '''
    else:
        raise NotImplementedError

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)

    # if args.restore_from_D[:4] == 'http':
    #     saved_state_dict = model_zoo.load_url(args.restore_from_D)
    # else:
    #     saved_state_dict = torch.load(args.restore_from_D)
    #     ### for running different versions of pytorch
    # model_dict = model_D1.state_dict()
    # saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    # model_D1.load_state_dict(saved_state_dict)


    model_D1.train()
    model_D1.to(device)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_loader = data_loader(args)


    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    seg_loss = torch.nn.CrossEntropyLoss()

    interp = nn.Upsample(size=(416, 416), mode='bilinear', align_corners=True)
    #interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training

    # set up tensorboard
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    count = args.start_count  # 迭代次数
    for dat in train_loader:
        if count > args.num_steps:
            break

        loss_seg_value1_anchor = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0


        optimizer.zero_grad()
        adjust_learning_rate(optimizer, count)

        optimizer_D1.zero_grad()

        adjust_learning_rate_D(optimizer_D1, count)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False


            # 相当于group=0时，训练样本对应的类有15类为[0,1,2,3,4,5,6,7,8,9,10,....]，验证集有5类，
            # 现在从训练集类中随机选择两类，然后从其中一类中选择两张图片，对应为基准图片和正样本图片，
            # 两者属于同一类，接着从另一类中选择一张图片作为负样本，属于不同类。其中基准图片对应的是查询集图片
            #############################
            anchor_img, anchor_mask, pos_img, pos_mask, neg_img, neg_mask = dat  # 返回的是基准图片以及mask，正样本以及mask（和基准图片属于同一类），负样本以及mask（和基准图片属于不同类）

            anchor_img, anchor_mask, pos_img, pos_mask, \
                = anchor_img.cuda(), anchor_mask.cuda(), pos_img.cuda(), pos_mask.cuda()  # [1, 3, 386, 500],[1, 386, 500],[1, 3, 374, 500],[1, 374, 500]

            anchor_mask = torch.unsqueeze(anchor_mask, dim=1)  # [1, 1, 386, 500]
            pos_mask = torch.unsqueeze(pos_mask, dim=1)  # [1,1, 374, 500]
            samples = torch.cat([pos_img, anchor_img], 0)

            pred = model(samples, pos_mask)  ##[2, 2, 53, 53],#[2, 2, 53, 53]
            pred = interp(pred)

            loss_seg1_anchor = seg_loss(pred, anchor_mask.squeeze().unsqueeze(0).long())
            D_out1 = model_D1(F.softmax(pred))
            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(1).to(
                device))  # 相当于将源域的标签设置为1,然后判断判别网络得到的目标预测与源域对应的损失
            '''
            s = torch.stack([s, 1-s])
            loss_s = seg_loss()
            '''
            loss = loss_seg1_anchor + args.lambda_adv_target1 * loss_adv_target1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()

            loss_seg_value1_anchor += loss_seg1_anchor.item() / args.iter_size
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size


            # train D# bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True


            # train with anchor
            pred_target1 = pred.detach()
            D_out1 = model_D1(F.softmax(pred_target1))
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(0).to(device))
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            # train with GT
            anchor_gt = Variable(one_hot(anchor_mask)).cuda()
            D_out1 = model_D1(anchor_gt)
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(1).to(device))
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

        optimizer.step()
        optimizer_D1.step()

        count = count + 1
        if args.tensorboard:
            scalar_info = {

                'loss_seg1_anchor': loss_seg_value1_anchor,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_D1': loss_D_value1,

            }

            if count % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, count)

        # print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}'.format(
            count, args.num_steps, loss_seg_value1_anchor,loss_adv_target_value1,loss_D_value1))

        if count >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'voc2012_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'voc2012_' + str(args.num_steps_stop) + '_D1.pth'))
            break

        if count % args.save_pred_every == 0 and count != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'voc2012_' + str(count) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'voc2012_' + str(count) + '_D1.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
