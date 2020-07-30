import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.graph_conv import build_graph_conv
import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn
        self.multi_graph_conv = build_graph_conv(backbone, BatchNorm)
        self.layer1 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False), BatchNorm(512), nn.ReLU())
        self.classifier_6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, dilation=1, padding=1),  # fc6
            nn.ReLU(inplace=True)
        )
        self.exit_layer = nn.Conv2d(128, 2, kernel_size=1, padding=1)

        ############################################################

        bins = [13, 26]
        self.features = []
        for bin in bins:  # [1,2,3,6]
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),  # 括号里面的大小是多少，输出来的大小就是多少
            ))
        ############################################################

    def forward1(self, input, mask):  # [2, 3, 416, 416]
        # 对特征进行自适应pooling,进行图卷积和跨图卷积,然后计算关系矩阵s,最后由关系矩阵得到相似性矩阵A
        x2, x3 = self.backbone(input)  # [2, 512, 52, 52],[2, 1024, 52, 52]
        _, _, w, h = x2.size()
        all_features = []
        for f in self.features:
            f_2 = f(x2)  # [2, 512, 13, 13][2, 512,26, 26]
            f_3 = f(x3)  # ,[2, 1024, 13, 13][2, 1024,26,26]
            f_A2, _ = self.multi_graph_conv(f_2, mask)  # [1,13,13][1,26,26]
            f_A3, _ = self.multi_graph_conv(f_3, mask)  # [1,13,13][1,26,26]
            f_x2_pos = f_2[1].unsqueeze(0) * f_A2  # [1, 512, 13,13][1, 512,26,26]
            f_x3_pos = f_3[1].unsqueeze(0) * f_A3  # [1, 1024, 13,13][1, 1024,26,26]
            f_x2_pos = F.interpolate(f_x2_pos, size=[w, h], mode='bilinear',
                                     align_corners=True)  # [1, 512,  52, 52][1, 512,  52, 52]
            f_x3_pos = F.interpolate(f_x3_pos, size=[w, h], mode='bilinear',
                                     align_corners=True)  # [1, 1024, 52, 52] [1, 1024, 52, 52]
            all_features.append(torch.cat([f_x2_pos, f_x3_pos], dim=1))  # [1, 1536, 52, 52] [1, 1536, 52, 52]
        A2, _ = self.multi_graph_conv(x2, mask)  # [1,52,52]
        A3, _ = self.multi_graph_conv(x3, mask)  # [1,52,52]
        x2_pos = x2[1].unsqueeze(0) * A2  # [1, 512, 52, 52]
        x3_pos = x3[1].unsqueeze(0) * A3  # [1, 1024, 52, 52]
        x = torch.cat([x2_pos, x3_pos], 1)  # [1, 1536, 52, 52]
        all_features.append(x)
        x = torch.sum(torch.cat(all_features), dim=0).unsqueeze(0)  # [1, 1536, 52, 52]

        x = self.aspp(x)  # [1, 256, 52, 52]
        x = self.decoder(x)  # [2,2,52, 52]
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)#[1, 2, 416, 416]

        return x

    def forward(self, input, mask):  # [2, 3, 416, 416]
        # out_step = 16, 两个特征分别 进行图卷积和跨图卷积,然后计算关系矩阵s,最后由关系矩阵得到相似性矩阵A
        x2, x3 = self.backbone(input)  # [2, 512, 52, 52],[2, 1024, 26, 26]
        x2_pos = self.multi_graph_conv(x3, mask)  # [1, 512, 52, 52]
        #x3_pos = self.multi_graph_conv(x3, mask)#[1, 512, 26, 26]
        #A2, _ = self.multi_graph_conv(x2, mask)  # [1,52,52]
        #A3, _ = self.multi_graph_conv(x3, mask)  # [1,26,26]
        #x2_pos = x2[1].unsqueeze(0) * A2  # [1, 512, 52, 52]
        #x3_pos = x3[1].unsqueeze(0) * A3  # [1, 1024,26, 26]
        x2_pos = self.aspp(x2_pos)  # [1, 256, 52, 52]
        x = self.decoder(x2_pos)  # [1,2,52, 52]
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)#[1, 2, 416, 416]

        # import matplotlib.pyplot as plt
        # plt.imshow(input[0][0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(input[1][0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(x2[1][0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(x2_pos[0][0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(x[0][1].cpu().detach().numpy())
        # plt.show()

        return x

    def forward3(self, input, mask):  # [2, 3, 416, 416]
        # 进行图卷积和跨图卷积,然后计算关系矩阵s,最后由关系矩阵得到相似性矩阵A
        x2, x3 = self.backbone(input)  # [2, 512, 52, 52],[2, 1024, 52, 52]
        x = torch.cat([x2, x3], 1)  # [2,1536,52,52]
        A, _ = self.multi_graph_conv(x, mask)  # [1,52,52]
        x = self.layer1(x)
        x = x[1].unsqueeze(0) * A  # [1, 512, 52, 52]

        # x = torch.cat([x2_pos, x3_pos], 1)  # [1, 1536, 52, 52]
        x = self.aspp(x)
        x = self.decoder(x)  # [2,2,52, 52]
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)#[1, 2, 416, 416]

        return x

    def forward2(self, input, mask):  # [2, 3, 416, 416]
        x2, x3 = self.backbone(input)  # [2, 512, 52, 52],[2, 1024, 52, 52]
        # 进行图卷积和跨图卷积,然后计算关系矩阵s,最后由关系矩阵得到相似性矩阵A
        # x = torch.cat([x2, x3], 1)
        A, _ = self.multi_graph_conv(x2, mask)  # [1,52,52]
        x = self.layer1(x2)
        x_pos = x[1].unsqueeze(0) * A  # [1, 512, 52, 52]
        x = x[1].unsqueeze(0) + x_pos
        x = self.classifier_6(x)
        x = self.exit_layer(x)
        # x = torch.cat([x2_pos, x3_pos], 1)  # [1, 1536, 52, 52]
        # x_pos = self.aspp(x_pos)
        # x = self.decoder(x_pos)  # [2,2,52, 52]
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)#[1, 2, 416, 416]
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    # model.eval()
    input = torch.rand(2, 3, 53, 53)
    mask = torch.rand(2, 3, 53, 53)
    output = model(input, mask)
    print(output.size())
