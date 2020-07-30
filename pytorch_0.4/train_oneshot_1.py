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
from modeling.focalloss2d import FocalLoss2d

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
NUM_STEPS = 200000
NUM_STEPS_STOP = 200000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = '/home/prlab/willow/Segmentation/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_1_200000.pth'
RESTORE_FROM_D = '/home/prlab/willow/Segmentation/AdaptSegNet-master/pytorch_0.4/snapshots/voc2012_1_200000_D1.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.001
LAMBDA_ADV_TARGET2 = 0.001
# LAMBDA_ADV_TARGET1 = 0.0002
# LAMBDA_ADV_TARGET2 = 0.001
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
    # parser.add_argument("--restore_from", type=str,default='/home/prlab/willow/Oneshot/SG-One-master/snapshots/onemodel_sg_one_scale/group_1_of_4/step_80000.pth.tar')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)

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
        for k_model, k_pretrained in zip(list(model_dict.keys()),
                                         list(pre_model.keys())[:len(model_dict.keys())]):  # 增加了list
            if model_dict[k_model].size() == pre_model[k_pretrained].size():
                print("%s\t<--->\t%s" % (k_model, k_pretrained))
                new_model_dict[k_model] = pre_model[k_pretrained]
            else:
                print('Fail to load %s' % (k_model))

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
        one_hot[:, i, ...] = (label == i)
    # handle ignore labels
    return torch.FloatTensor(one_hot)


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         p_t = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):  # [1, 2, 416, 416],[1, 416, 416]
        if inputs.dim() > 2:
            inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)  # [1, 2, 173056]
            inputs = inputs.transpose(1, 2)  # [1, 173056, 2]
            inputs = inputs.contiguous().view(-1, inputs.size(2)).squeeze()  # [173056, 2]
        if targets.dim() == 4:
            targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)  # [1, 1, 173056]
            targets = targets.transpose(1, 2)  # [1, 173056, 1]
            targets = targets.contiguous().view(-1, targets.size(2)).squeeze()  # [173056]
        elif targets.dim() == 3:
            targets = targets.view(-1)
        else:
            targets = targets.view(-1, 1)
        N = inputs.size(0)  # 416*416
        C = inputs.size(1)  # channel,2
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)  # [1, 2]
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)  # [173056, 1]
        class_mask.scatter_(1, ids.data,
                            1.)  # scatter_(dim, index, src) → Tensor1）维度dim 2）索引数组index 3）原数组src，为了方便理解，我们后文把src换成input表示。最终的输出是新的output数组。
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class AffinityFieldLoss(nn.Module):
    '''
        loss proposed in the paper: https://arxiv.org/abs/1803.10335
        used for sigmentation tasks
    '''
    def __init__(self, kl_margin, lambda_edge=1., lambda_not_edge=1., ignore_lb=255):
        super(AffinityFieldLoss, self).__init__()
        self.kl_margin = kl_margin
        self.ignore_lb = ignore_lb
        self.lambda_edge = lambda_edge
        self.lambda_not_edge = lambda_not_edge
        self.kldiv = nn.KLDivLoss(reduction='none')

    def forward(self, logits, labels):
        ignore_mask = labels.cpu() == self.ignore_lb
        n_valid = ignore_mask.numel() - ignore_mask.sum().item()
        indices = [
                # center,               # edge
            ((1, None, None, None), (None, -1, None, None)), # up
            ((None, -1, None, None), (1, None, None, None)), # down
            ((None, None, 1, None), (None, None, None, -1)), # left
            ((None, None, None, -1), (None, None, 1, None)), # right
            ((1, None, 1, None), (None, -1, None, -1)), # up-left
            ((1, None, None, -1), (None, -1, 1, None)), # up-right
            ((None, -1, 1, None), (1, None, None, -1)), # down-left
            ((None, -1, None, -1), (1, None, 1, None)), # down-right
        ]

        losses = []
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        for idx_c, idx_e in indices:
            lbcenter = labels[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            lbedge = labels[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            igncenter = ignore_mask[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            ignedge = ignore_mask[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            lgp_center = probs[:, :, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]]
            lgp_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            prob_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            kldiv = (prob_edge * (lgp_edge - lgp_center)).sum(dim=1)

            kldiv[ignedge | igncenter] = 0
            loss = torch.where(
                lbcenter == lbedge,
                self.lambda_edge * kldiv,
                self.lambda_not_edge * F.relu(self.kl_margin - kldiv, inplace=True)
            ).sum() / n_valid
            losses.append(loss)

        return sum(losses) / 8

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
        # model = DeeplabMulti(num_classes=args.num_classes)
        # model = Res_Deeplab(num_classes=args.num_classes)
        model = DeepLab(backbone='resnet', output_stride=8)
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

    # saved_state_dict = torch.load(args.restore_from)
    # ### for running different versions of pytorch
    # model_dict = model.state_dict()
    # saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    # model.load_state_dict(saved_state_dict)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    #model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)

    # saved_state_dict = torch.load(args.restore_from_D)
    # ### for running different versions of pytorch
    # model_dict = model_D1.state_dict()
    # saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    # model_D1.load_state_dict(saved_state_dict)

    # model_D1.train()
    # model_D1.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_loader = data_loader(args)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # optimizer_D1.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    #seg_loss = FocalLoss2d(gamma=2.0, weight=0.75).to(device)#alpha是用来衡量样本的正负样本不平衡的
    #seg_loss = FocalLoss2d(gamma=2.0, weight=0.75).to(device)
    # seg_loss = FocalLoss(alpha=0.75, logits=True)
    #seg_loss = FocalLoss(class_num=2).to(device)
    seg_loss = torch.nn.CrossEntropyLoss()
    affinity_loss = AffinityFieldLoss(kl_margin=3.)
    R_loss = torch.nn.MSELoss()
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training

    # set up tensor board
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
        loss_affinity_value1_anchor = 0
        loss_D_value1 = 0
        loss_R_values = 0
        loss_A_values = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, count)

        # optimizer_D1.zero_grad()
        # adjust_learning_rate_D(optimizer_D1, count)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            # for param in model_D1.parameters():
            #     param.requires_grad = False

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
            if count == 5134:
                import matplotlib.pyplot as plt
                plt.imshow(pos_img[0][0].cpu().detach().numpy())
                plt.show()
                plt.imshow(pos_mask[0][0].cpu().detach().numpy())
                plt.show()

            pred = model(samples, pos_mask)  ##[2, 2, 53, 53],#[2, 2, 53, 53]#[1,2704,2704]#[1,52,52]
            _, _, w1, h1 = pred.size()
            _, _, mask_w, mask_h = anchor_mask.size()
            ####################分割loss和对抗loss#############################################
            pred = F.interpolate(pred, [mask_w, mask_h], mode='bilinear', align_corners=False)
            # loss_seg1_anchor = seg_loss(pred.squeeze(), anchor_mask.squeeze())###针对BCELOSS
            loss_seg1_anchor = seg_loss(pred, anchor_mask.squeeze().unsqueeze(0).long())  ##SOFTMAX
            loss_affinity = affinity_loss(pred, anchor_mask.squeeze().unsqueeze(0).long())
            # D_out1 = model_D1(F.softmax(pred))
            # loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(1).to(
            #     device))  # 相当于将源域的标签设置为1,然后判断判别网络得到的目标预测与源域对应的损失
            #########################关系矩阵损失,R#################################################
            # G_q = F.interpolate(anchor_mask, [w1, h1], mode='bilinear', align_corners=True)
            # G_s = F.interpolate(pos_mask, [w1, h1], mode='bilinear', align_corners=True)
            # R_gt = G_q.reshape(w1 * h1, -1) * G_s.reshape(-1, w1 * h1)
            # loss_R1 = R_loss(R1.squeeze(), R_gt)
            # loss_R2 = R_loss(R2.squeeze(), R_gt)
            ##########################注意力矩阵A loss####################################################
            '''
            A1 = torch.cat([1 - A1, A1], 0)
            A1 = interp(A1.unsqueeze(0))
            A2 = torch.cat([1 - A2, A2], 0)
            A2 = interp(A2.unsqueeze(0))
            loss_A1 = seg_loss(A1, anchor_mask.squeeze().unsqueeze(0).long())
            loss_A2 = seg_loss(A2, anchor_mask.squeeze().unsqueeze(0).long())
            '''
            #######################总的loss#############################################
            # loss = loss_seg1_anchor + args.lambda_adv_target1 * loss_adv_target1 + 0.3 * loss_R1 +0.3 * loss_R2 + 0.2 * loss_A1 + 0.2 * loss_A2
            loss = loss_seg1_anchor + loss_affinity
            # proper normalization
            loss = loss / args.iter_size
            loss.backward()

            loss_seg_value1_anchor += loss_seg1_anchor.item() / args.iter_size
            loss_affinity_value1_anchor += loss_affinity.item() / args.iter_size
            # loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
            # loss_R_values += loss_R1.item() / args.iter_size
            # loss_R_values += loss_R2.item() / args.iter_size
            # loss_A_values += loss_A1.item() / args.iter_size
            # loss_A_values += loss_A2.item() / args.iter_size

            # train D# bring back requires_grad
            # for param in model_D1.parameters():
            #     param.requires_grad = True
            #
            # # train with anchor
            # pred_target1 = pred.detach()
            # D_out1 = model_D1(F.softmax(pred_target1))
            # loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(0).to(device))
            # loss_D1 = loss_D1 / args.iter_size / 2
            # loss_D1.backward()
            # loss_D_value1 += loss_D1.item()
            #
            # # train with GT
            # anchor_gt = Variable(one_hot(anchor_mask)).cuda()
            # D_out1 = model_D1(anchor_gt)
            # loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(1).to(device))
            # loss_D1 = loss_D1 / args.iter_size / 2
            # loss_D1.backward()
            # loss_D_value1 += loss_D1.item()

        optimizer.step()
        # optimizer_D1.step()

        count = count + 1
        if args.tensorboard:
            scalar_info = {

                'loss_seg1_anchor': loss_seg_value1_anchor,
                'loss_affinity_anchor': loss_affinity_value1_anchor,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_D1': loss_D_value1,
                'loss_R': loss_R_values,
                'loss_A': loss_A_values
            }

            if count % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, count)

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_affinity = {3:.3f}, loss_D1 = {4:.3f}, loss_R = {5:.3f} loss_A = {6:.3f}'.format(
                count, args.num_steps, loss_seg_value1_anchor, loss_affinity_value1_anchor, loss_D_value1, loss_R_values,
                loss_A_values))

        if count >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, 'voc2012_1_' + str(args.num_steps_stop) + '.pth'))
            # torch.save(model_D1.state_dict(),
            #            osp.join(args.snapshot_dir, 'voc2012_1_' + str(args.num_steps_stop) + '_D1.pth'))
            break

        if count % args.save_pred_every == 0 and count != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'voc2012_1_' + str(count) + '.pth'))
            # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'voc2012_1_' + str(count) + '_D1.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
