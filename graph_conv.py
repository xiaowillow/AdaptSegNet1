import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.PCA.gconv import Siamese_Gconv
from model.PCA.affinity_layer import Affinity


class Graph_conv(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(Graph_conv, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            # low_level_inplanes = 256
            inplanes = 512
        elif backbone == 'xception':
            inplanes = 1024
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(inplanes, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(inplanes * 2, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.bn2 = BatchNorm(256)
        self.relu = nn.ReLU()
        # self.l2norm = nn.LocalResponseNorm(256 * 2, alpha=256 * 2, beta=0.5,
        #                                   k=0)
        self.gnn_layer = 2
        # self.cos_similarity_func = nn.CosineSimilarity()
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(256, 256 * 2)
            else:
                gnn_layer = Siamese_Gconv(256 * 2, 256 * 2)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(256 * 2))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i),
                                nn.Sequential(nn.Linear(256 * 2 * 2, 256 * 2), nn.BatchNorm1d(512), nn.ReLU()))

        self._init_weight()

    def get_adjacency(self, x, y):  # [1,512,52,52]
        x_reshape = F.normalize(x)  # [1,2704, 128]
        y_reshape = F.normalize(y)  # [1,2704, 128]
        # x_reshape = x.reshape(w * h, -1).unsqueeze(0)

        '''
        A = torch.ones([w*h, w*h])#[2704, 2704]
        for i in range(w*h):
            similar = self.cos_similarity_func(x, x_reshape[:, i].unsqueeze(2).unsqueeze(3))
            A[i] = similar.reshape(1, -1)
        '''
        A = torch.matmul(x_reshape, y_reshape.transpose(1, 2))

        return A / A.max()  # [1, 2704, 2704]

    def get_adjacency_eu(self, x, y):  # x,y:[1, 2704, 256]
        # A = torch.matmul(x, y.transpose(1, 2))
        A = torch.rand(x.shape[1], y.shape[1])  # [2704, 2704]
        for i in range(x.shape[1]):
                # A[i][j] = torch.sqrt(torch.sum((x[0][i] - y[0][j])**2))
                A[i] = torch.pairwise_distance(x[0][i].unsqueeze(0).cpu(), y[0].cpu(), p=2)

        temp = torch.ones_like(A) * 100
        A = temp - A

        _, idx = torch.topk(A, k=10, dim=-1, largest=True)
        for i in range(idx.shape[0]):  # [2704, 10]
            for index in idx[i]:
                A[i][index] = 1

        zero = torch.zeros_like(A)
        A = torch.where(A == 1, A, zero).unsqueeze(0)
        return A

    def forward(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对特征上采样与mask相乘后,再下采样到原特征大小,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv

        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        # x = self.conv1(x)
        ##################################################################
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性

        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        #########均值系列###########
        #vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)#
        #vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)
        ##########################
        pos_node = F.interpolate(pos_node * mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1,256,52,52]
        x_s = pos_node.reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        A_s = self.get_adjacency(x_s, x_s)
        A_q = self.get_adjacency(x_q, x_q)  ## [1, 2704,2704][1, 169, 169]

        ###################################################################
        # graph_conv
        # 得到关系矩阵R
        emb1 = x_s  # [1, 2704, 256][1, 169, 256]
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            R = self.get_adjacency(emb1, emb2)  # [1, 2704, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(
                    torch.cat((emb1.squeeze(), torch.bmm(R.cuda(), emb2).squeeze()), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(
                    torch.cat((emb2.squeeze(), torch.bmm(R.transpose(1, 2).cuda(), emb1).squeeze()), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new.unsqueeze(0)
                emb2 = emb2_new.unsqueeze(0)
        return emb2.reshape(1, -1, w, h)

    def forward_copy(self, x, mask):
        b, dim, w, h = mask.size()
        b1, dim1, w1, h1 = x.size()

        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = pos_node * mask
        vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w1, h1)
        pos_node = F.interpolate(pos_node, [w1, h1], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)

        x_s = x[0].reshape(1, w1 * h1, -1)
        x_q = x[1].reshape(1, w1 * h1, -1)

        emb_vec = vec_pos.reshape(1, w1 * h1, -1)
        emb1 = x_s
        emb2 = x_q
        for i in range(self.gnn_layer):
            R1 = F.softmax(torch.matmul(emb1, emb2.transpose(1, 2)), 1)
            R2 = F.softmax(torch.matmul(emb2, emb1.transpose(1, 2)), 1)
            R_vec = F.softmax(torch.matmul(emb2, emb_vec.transpose(1, 2)), 1)

            emb1_new = emb1 + torch.bmm(R1, emb2)
            emb2_new = emb2 + torch.bmm(R2, emb1)
            emb_vec_new = emb2 + torch.bmm(R_vec, emb_vec)
            emb1 = emb1_new
            emb2 = emb2_new
            emb_vec = emb_vec_new

        return emb2.reshape(1, -1, w1, h1) + emb_vec.reshape(1, -1, w1, h1)

    def forward0(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################复现金字塔那篇文章,先将特征进行卷积操作,然后对特征上采样与mask相乘后再变回原图大小,
        # 计算查询集中每个查询点与支持集中每个点的相似性,然后进行特征加权操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv
        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        else:
            x = self.conv2(x)
        #######对支持集乘上mask处理
        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        x_s = F.interpolate(pos_node * mask, [w, h], mode='bilinear', align_corners=True).reshape(1, w * h,
                                                                                                  -1)  # [1, 2704, 256][1, 169, 256]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        #########计算q中每个点到s中所有点的相似性,然后用softmax归一化
        A_q_s = F.softmax(torch.matmul(x_q, x_s.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        x_q = x_q + torch.bmm(A_q_s, x_s)
        return x_q.reshape(1, -1, w, h)

    def forward1(self, x, pos_mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv
        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        else:
            x = self.conv2(x)
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性
        pos_mask = F.interpolate(pos_mask, [w, h], mode='bilinear', align_corners=True)  # [1, 1, 13, 13]
        x_s = (x[0] * pos_mask).reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        A_s = F.softmax(torch.matmul(x_s, x_s.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        A_q = F.softmax(torch.matmul(x_q, x_q.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        x = self.l2norm(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        emb1 = x_s
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]
        # s = torch.zeros([w * h, w * h])
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            affinity = getattr(self, 'affinity_{}'.format(i))
            R = affinity(emb1, emb2)  # [1, 2704, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R, emb2)), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R.transpose(1, 2), emb1)), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new
                emb2 = emb2_new

        # s = torch.sum(s, 1).reshape(1, w, h)  # [1, 53, 53]
        # s = (s - s.min()) / (s.max() - s.min())
        mask_inver = torch.matmul(pos_mask.reshape(-1, 1),
                                  torch.matmul(pos_mask.reshape(1, -1), pos_mask.reshape(-1, 1)).pinverse())
        R = (R - R.min()) / (R.max() - R.min())
        A = torch.matmul(R, mask_inver)
        # A = (A - A.min()) / (A.max() - A.min())

        return A.reshape(1, w, h), R

    def forward4(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对特征上采样与mask相乘后,再下采样到原特征大小,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv

        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        # x = self.conv1(x)
        ##################################################################
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear', align_corners=True)  # [1, 1, 13, 13]
        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        #########均值系列###########
        # vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)#
        # vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)
        ##########################
        pos_node = F.interpolate(pos_node * mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1,512,52,52]
        x_s = pos_node.reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        A_s = self.get_adjacency(x_s, x_s)
        A_q = self.get_adjacency(x_q, x_q)  ## [1, 2704,2704][1, 169, 169]
        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        # x = self.l2norm(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        # b, f, w, h = out_node.size()
        ###################################################################
        # graph_conv
        # 得到关系矩阵R

        emb1 = x_s  # [1, 2704, 256][1, 169, 256]
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]
        R = torch.zeros([w * h, w * h])
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]

            affinity = getattr(self, 'affinity_{}'.format(i))
            R = affinity(emb1, emb2)  # [1, 2704, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(
                    torch.cat((emb1.squeeze(), torch.bmm(R, emb2).squeeze()), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(
                    torch.cat((emb2.squeeze(), torch.bmm(R.transpose(1, 2), emb1).squeeze()), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new.unsqueeze(0)
                emb2 = emb2_new.unsqueeze(0)

        # s = torch.sum(s, 1).reshape(1, w, h)  # [1, 53, 53]
        # s = (s - s.min()) / (s.max() - s.min())
        mask_inver = torch.matmul(pos_mask.reshape(-1, 1),
                                  torch.matmul(pos_mask.reshape(1, -1), pos_mask.reshape(-1, 1)).pinverse())
        R = (R - R.min()) / (R.max() - R.min())
        A = torch.matmul(R, mask_inver)
        A = (A - A.min()) / (A.max() - A.min())

        return A.reshape(1, w, h), R

    def forward3(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################复现局部变换那篇文章
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv

        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        else:
            x = self.conv2(x)

        # x = self.conv1(x)
        #########均值系列###########
        # vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)#
        # vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)
        ##########################
        ##################################################################
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear', align_corners=True)  # [1, 1, 13, 13]
        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = F.interpolate(pos_node * mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1,512,52,52]
        x_s = pos_node.rehsape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        ###通过当前特征进行计算R矩阵
        R = self.get_adjacency(x_s, x_q)
        mask_inver = torch.matmul(pos_mask.reshape(-1, 1),
                                  torch.matmul(pos_mask.reshape(1, -1), pos_mask.reshape(-1, 1)).pinverse())
        A = torch.matmul(R, mask_inver)
        A = (A - A.min()) / (A.max() - A.min())

        return A.reshape(1, w, h), R

    def forward2(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对特征上采样与mask相乘后,然后计算支持集的均值向量,\
        # 利用均值向量填充,构造支持集特征map,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv
        x = self.conv1(x)
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear', align_corners=True)  # [1, 1, 13, 13]
        ############################################################################
        pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)

        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]

        A_q = F.softmax(torch.matmul(x_q, x_q.transpose(1, 2)), 2)  ## [1, 2704,2704][1, 169, 169]
        A_s = torch.ones_like(A_q)
        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        # x = self.l2norm(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        # b, f, w, h = out_node.size()
        emb1 = vec_pos.reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]

        # s = torch.zeros([w * h, w * h])
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            affinity = getattr(self, 'affinity_{}'.format(i))
            R = affinity(emb1, emb2)  # [1, 2704, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R, emb2)), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R.transpose(1, 2), emb1)), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new
                emb2 = emb2_new

        # s = torch.sum(s, 1).reshape(1, w, h)  # [1, 53, 53]
        # s = (s - s.min()) / (s.max() - s.min())
        mask_inver = torch.matmul(pos_mask.reshape(-1, 1),
                                  torch.matmul(pos_mask.reshape(1, -1), pos_mask.reshape(-1, 1)).pinverse())
        R = (R - R.min()) / (R.max() - R.min())
        A = torch.matmul(R, mask_inver)
        # A = (A - A.min()) / (A.max() - A.min())

        return A.reshape(1, w, h), R

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_graph_conv(backbone, BatchNorm):
    return Graph_conv(backbone, BatchNorm)
