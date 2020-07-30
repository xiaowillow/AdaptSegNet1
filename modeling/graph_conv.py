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
        self.cls_scale = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # fc6
        )
        self.padding = nn.ZeroPad2d(1)
        self.theta = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # fc6
        )
        self.phi = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # fc6
        )
        self.g = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # fc6
        )
        self.eth = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # fc6
        )

        # self.relu = nn.ReLU()
        # self.l2norm = nn.LocalResponseNorm(256 * 2, alpha=256 * 2, beta=0.5,
        #                                   k=0)
        self.gnn_layer = 2
        # self.cos_similarity_func = nn.CosineSimilarity()

        # for i in range(self.gnn_layer):
        #     if i == 0:
        #         # self.add_module('affinity_{}'.format(i), Affinity(512 * 2))
        #         self.add_module('cross_graph_{}'.format(i),
        #                         nn.Sequential(nn.Linear(3072, 256 * 2)))

        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(512, 512 * 2)
            else:
                gnn_layer = Siamese_Gconv(512 * 2, 512 * 2)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(512 * 2))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i),
                                nn.Sequential(nn.Linear(512 * 2 * 2, 256 * 2)))

        self._init_weight()

    def get_adjacency(self, x, y):  # [1,512,52,52]
        # b, w, h = x.size()
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

    def forward5(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对特征上采样与mask相乘后,再下采样到原特征大小,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        # 先进行1*1conv

        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]

        else:
            x = self.conv2(x)

        # x = self.conv1(x)
        ##################################################################
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性

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
                    torch.cat((emb1.squeeze(), torch.bmm(R, emb2).squeeze()), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(
                    torch.cat((emb2.squeeze(), torch.bmm(R.transpose(1, 2), emb1).squeeze()), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new.unsqueeze(0)
                emb2 = emb2_new.unsqueeze(0)
        return emb2.reshape(1, -1, w, h)

    def forward3(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对特征上采样与mask相乘后,再下采样到原特征大小,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        import matplotlib.pyplot as plt
        plt.imshow(mask[0][0].cpu().detach().numpy())
        plt.show()
        plt.imshow(x[0][0].cpu().detach().numpy())
        plt.show()
        b, dim, w, h = x.size()
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = x[0].unsqueeze(0) * pos_mask
        #####################
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)
        # 先进行1*1conv
        if dim == 512:
            x = self.conv1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
        else:
            x = self.conv2(x)
        plt.imshow(x[0][0].cpu().detach().numpy())
        plt.show()
        ##################################################################
        # 进行graph conv
        # 构建邻接矩阵A
        # 先考虑特征相似性

        # pos_node = F.interpolate(x[0].unsqueeze(0), mask.shape[2:], mode='bilinear',
        #                         align_corners=True)  # [1, 1, 13, 13]
        #########均值系列###########
        # vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)#
        # vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)
        ##########################
        # pos_node = F.interpolate(pos_node * mask, [w, h], mode='bilinear',
        #                         align_corners=True)  # [1,512,52,52]

        x_s = x[0].reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
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
                    torch.cat((emb1.squeeze(), torch.bmm(R, emb2).squeeze()), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(
                    torch.cat((emb2.squeeze(), torch.bmm(R.transpose(1, 2), emb1).squeeze()), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new.unsqueeze(0)
                emb2 = emb2_new.unsqueeze(0)

        return emb2.reshape(1, -1, w, h)

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

    def forward_00(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = x[0].unsqueeze(0) * pos_mask
        ##########################################################
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)
        # 先进行1*1conv
        if dim == 512:
            x = self.cls1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
            # x = self.relu(x)
        else:
            x = self.cls2(x)
            # x = self.relu(x)

        x_s = x[0].reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]

        A_s = F.softmax(torch.matmul(x_s, x_s.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        A_q = F.softmax(torch.matmul(x_q, x_q.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        emb1 = x_s
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]
        # s = torch.zeros([w * h, w * h])
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            affinity = getattr(self, 'affinity_{}'.format(i))
            R = affinity(emb2, emb1)  # R = Gq*Gs [1, 2704, 2704]
            R = (R - R.min()) / (R.max() - R.min())
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R.transpose(1, 2), emb2)), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R, emb1)), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new
                emb2 = emb2_new
                A_s = F.softmax(torch.matmul(emb1, emb1.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
                A_q = F.softmax(torch.matmul(emb2, emb2.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        return R, emb2.reshape(1, -1, w, h)

    def gcn(self, x):  # [2,512,52,52]
        # x_patch = F.unfold(x, kernel_size=5, padding=2, stride=1)#[2, 512*25, 2704=52*52]
        batch, _, w, h = x.size()
        x1 = self.padding(x)
        x_patch = F.unfold(x1, kernel_size=3, dilation=1, stride=1)  # [2, 512*9, 2704=52*52]
        x_patch = x_patch.reshape(batch, w * h, 3 * 3, -1)  # [2, 169, 9, 1536]
        x_similar = torch.matmul(x_patch, x_patch.transpose(3, 2))  # [2, 169, 9, 9]
        x_similar = x_similar / x_similar.max()
        x_patch = torch.matmul(x_patch.transpose(2, 3), x_similar[:, :, 4, :].unsqueeze(3))
        x_patch = x_patch.reshape(batch, -1, w, h)
        return x_patch

    def forward_resize_feature(self, pos, anchor, mask):  # pos [1, 1536, 52, 52]anchor[1, 1536, 13, 13]
        # 复现scale金字塔,多尺度cat[x2,x3]
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = mask.size()
        b1, dim1, w1, h1 = pos.size()  # [1,1536,52,52]
        _, _, w2, h2 = anchor.size()
        theta_s = self.theta(pos)  # [1, 512, 47, 63]
        ####################支持图片先上次采样与mask相乘,再下采样################
        theta_s = F.interpolate(theta_s, [w, h], mode='bilinear', align_corners=False)  # [1, 512, 374, 500]
        theta_s = theta_s * mask  # [1, 512, 374, 500]
        theta_s = F.interpolate(theta_s, [w1, h1], mode='bilinear', align_corners=False)  # [1, 512, 47, 63]
        inf_mask = torch.where(theta_s[0][0] == 0, torch.full_like(theta_s[0][0], 1),
                               torch.full_like(theta_s[0][0], 0))  # [1, 512, 47, 63]
        #################计算权重e_s ###############################################################
        theta_s_1 = theta_s.view(1, -1, w1 * h1)  # [1, 512, 2961][1, 512, 2704]
        phi_q = self.phi(anchor)  # [1, 512, 13, 13] [1, 512, 52, 52]
        phi_q_1 = phi_q.view(1, w2 * h2, -1)  # [1, 169, 512]
        e_s = torch.matmul(phi_q_1, theta_s_1)  # [1, 169,2704] [1, 2704, 2704]
        ################对权重进行归一化#########
        # import matplotlib.pyplot as plt
        # plt.imshow(mask[0][0].cpu().detach().numpy())
        # plt.show()
        e_s_0 = e_s.reshape(w2 * h2, 1, w1 * h1)  # [169, 1, 2704]
        inf_mask = inf_mask.reshape(1, w1 * h1)  # [1,2704]
        e_s_masked = torch.where(inf_mask == 1, torch.full_like(e_s_0, float('-inf')), e_s_0)  # [169, 1, 2704]
        e_s = F.softmax(e_s_masked, dim=-1).reshape(1, w2 * h2, w1 * h1)  # [169, 1, 2704]
        print(e_s.sum())
        ######################计算最终的q#######################################
        g_s = self.g(pos).view(1, w1 * h1, -1)  # [1, 2704, 512]
        g_q = self.g(anchor)  # [1, 512, 13, 13][1, 512, 52, 52]

        v_q = torch.matmul(e_s, g_s)  # [1, 169, 512][1,2704,512]
        v_q = v_q.view(1, -1, w2, h2)  # [1, 512, 13, 13]
        finall_q = self.eth(torch.cat([g_q, v_q], 1))  # [1, 512, 13, 13]

        ################均值pooling part###############################
        if w2 == h2 == 52:
            pos_node = F.interpolate(theta_s, size=(w, h), mode='bilinear', align_corners=False)  # [1, 512, 416, 416]
            vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w2, h2)  # [1,512,52,52]
            #####每个权重都一样,然后在做加权求和,其实就相当于他自己
            # vec_q = self.g(vec_pos)
            vec_q = self.eth(torch.cat([g_q, vec_pos], 1))  # [1, 512, 52, 52]
            ######################
            return finall_q + vec_q
        return finall_q

    def forward_pyramid(self, pos, anchor, mask):  # pos [1, 1536, 52, 52]anchor[1, 1536, 13, 13]
        # 复现scale金字塔,多尺度cat[x2,x3],mask_resize
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = mask.size()
        b1, dim1, w1, h1 = pos.size()  # [1,1536,52,52]
        _, _, w2, h2 = anchor.size()
        ######################theta_s,phi_q, g_s,g_q#####################################
        theta_s = self.theta(pos)  # [1, 512, 52, 52]
        phi_q = self.phi(anchor)  # [1, 512, 13, 13] [1, 512, 52, 52]

        theta_s_1 = theta_s.view(1, -1, w1 * h1)  # [1, 512, 2704]
        phi_q_1 = phi_q.view(1, w2 * h2, -1)  # [1, 169, 512]

        g_s = self.g(pos)  # [1, 512, 13, 13]
        g_q = self.g(anchor)  # [1, 512, 13, 13][1, 512, 52, 52]

        ##############均值pooling part################################
        #####相当于不进行尺度降低时,就计算该特征map的均值相似性就行了

        ####################################################################

        # #################计算权重e_s ############################################
        e_s = torch.matmul(phi_q_1, theta_s_1)  # [1, 169,2704] [1, 2704, 2704]
        #########################归一化#########################################
        pos_mask = F.interpolate(mask, [w1, h1], mode='bilinear', align_corners=False)  # [1, 1, 52, 52]
        e_s_0 = e_s.reshape(w2 * h2, 1, w1 * h1)  # [169, 1, 2704]
        pos_mask_0 = pos_mask.reshape(1, w1 * h1)  # [1,2704]
        e_s_masked = torch.where(pos_mask_0 == 1, e_s_0, torch.full_like(e_s_0, float('-inf')))  # [169, 1, 2704]
        e_s = F.softmax(e_s_masked, dim=-1).reshape(1, w2 * h2, w1 * h1)  # [169, 1, 2704]
        #######################################################################
        g_s_0 = g_s.view(1, w1 * h1, -1)  # [1, 2704, 512]
        v_q = torch.matmul(e_s, g_s_0)  # [1, 169, 512][1,2704,512]
        v_q = v_q.view(1, -1, w2, h2)  # [1, 512, 13, 13]

        finall_q = self.eth(torch.cat([g_q, v_q], 1))  # [1, 512, 13, 13]
        if w2 == w1:
            pos_node = F.interpolate(theta_s, size=(w, h), mode='bilinear', align_corners=False)  # [1, 512, 416, 416]
            vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
            vec_pos = vec_pos.unsqueeze(dim=2).repeat(1, 1, w1 * h1)  # [1,512,52*52]
            #####计算权重
            e_s_1 = torch.matmul(phi_q_1, vec_pos)  # [1,16,16]
            e_s_1 = e_s_1 / torch.sum(e_s_1, 2)
            ###########计算g_s
            g_s_1 = F.interpolate(g_s, size=(w, h), mode='bilinear', align_corners=False)
            g_pos = torch.sum(torch.sum(g_s_1 * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
            g_pos = g_pos.unsqueeze(dim=2).repeat(1, 1, w1 * h1)  # [1,512,52*52]
            v_q_1 = torch.matmul(e_s_1, g_pos.reshape(1, w1 * h1, -1))
            v_q_1 = v_q_1.view(1, -1, w2, h2)
            finall_q_1 = self.eth(torch.cat([g_q, v_q_1], 1))  # [1, 512, 52, 52]
            return finall_q_1 + finall_q
        # ##############均值pooling part################################
        # pos_node = F.interpolate(theta_s, size=(w, h), mode='bilinear', align_corners=False)  # [1, 512, 416, 416]
        # vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
        # vec_pos = vec_pos.unsqueeze(dim=2).repeat(1, 1, w1 * h1)  # [1,512,52*52]
        # # vec_pos = torch.sum(torch.sum(theta_s, dim=3), dim=2)
        # #####计算权重
        # e_s_1 = torch.matmul(phi_q_1, vec_pos)
        # ###########计算g_s
        # g_s_1 = F.interpolate(g_s, size=(w, h), mode='bilinear', align_corners=False)
        # g_pos = torch.sum(torch.sum(g_s_1 * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
        # g_pos = g_pos.unsqueeze(dim=2).repeat(1, 1, w1 * h1)  # [1,512,52*52]
        # v_q_1 = torch.matmul(e_s_1, g_pos.reshape(1, w1*h1, -1))
        # v_q_1 = v_q_1.view(1, -1, w2, h2)
        # finall_q_1 = self.eth(torch.cat([g_q, v_q_1], 1))  # [1, 512, 52, 52]
        # ######################################################################################
        # return finall_q + finall_q_1

        # if w2 == h2 == 52:
        #     pos_node = F.interpolate(theta_s, size=(w, h), mode='bilinear', align_corners=False)  # [1, 512, 416, 416]
        #     vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)  # [1, 512]
        #     vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w2, h2)  # [1,512,52,52]
        #     #####每个权重都一样,然后在做加权求和,其实就相当于他自己
        #     # vec_q = self.g(vec_pos)
        #     vec_q = self.eth(torch.cat([g_q, vec_pos], 1))  # [1, 512, 52, 52]
        #     ######################
        #     return finall_q + vec_q
        return finall_q

    def forward(self, pos, anchor, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        # graph_conv
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = mask.size()
        b1, dim1, w1, h1 = pos.size()  # [1, 1024, 52, 52]
        _, _, w2, h2 = anchor.size()  # [1, 1024, 6, 6]
        x_s = self.cls_scale(pos)  # [1, 512, 52, 52]
        x_q = self.cls_scale(anchor)  # [1, 512, 13, 13]
        ############################################################

        x_s = x_s.reshape(1, w1 * h1, -1)  ## [1, 2704,512]
        x_q = x_q.reshape(1, w2 * h2, -1)  # [1, 2704, 512][1, 169, 512]
        A_s = self.get_adjacency_eu(x_s, x_s)
        A_q = self.get_adjacency_eu(x_q, x_q)

        emb1 = x_s
        emb2 = x_q  # [1, 2704, 1536][1, 169, 1536]

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            affinity = getattr(self, 'affinity_{}'.format(i))
            R = affinity(emb2, emb1)  # R = Gq*Gs [1, 2704, 2704]
            pos_mask = F.interpolate(mask, [w1, h1], mode='bilinear', align_corners=False)  # [1, 1, 52, 52]
            R = R.reshape(w2 * h2, 1, w1 * h1)  # [169, 1, 2704]
            pos_mask_0 = pos_mask.reshape(1, w1 * h1)  # [1,2704]
            R = torch.where(pos_mask_0 == 1, R, torch.full_like(R, float('-inf')))  # [169, 1, 2704]
            R = F.softmax(R, dim=-1).reshape(1, w2 * h2, w1 * h1)  # [169, 1, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R.transpose(1, 2), emb2)), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R, emb1)), dim=-1))  # [1, 2704, 512]
                emb2 = emb2_new
                emb1 = emb1_new
        return emb2.reshape(1, -1, w2, h2), R

    def forward_11(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        # graph_conv
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = mask.size()
        b1, dim1, w1, h1 = x.size()

        pos_node = F.interpolate(x[0].unsqueeze(0), [w, h], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = pos_node * mask
        ############################################################
        ########将均值pooling的相似性也要加上
        vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)  #
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w1, h1)
        ###########################################################
        pos_node = F.interpolate(pos_node, [w1, h1], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        ############################################################
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)
        x = self.gcn(x)
        x_s = x[0].reshape(1, w1 * h1, -1)  ## [1, 2704,1536][1, 169, 1536]
        x_q = x[1].reshape(1, w1 * h1, -1)  # [1, 2704, 1536][1, 169, 1536]

        # A_s = self.get_adjacency(x_s, x_s)
        # A_q = self.get_adjacency(x_q, x_q)
        # A_s = F.softmax(torch.matmul(x_s, x_s.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        # A_q = F.softmax(torch.matmul(x_q, x_q.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        emb_vec = vec_pos.reshape(1, w1 * h1, -1)
        emb1 = x_s
        emb2 = x_q  # [1, 2704, 1536][1, 169, 1536]
        for i in range(self.gnn_layer):
            R1 = F.softmax(torch.matmul(emb1, emb2.transpose(1, 2)), 1)
            R2 = F.softmax(torch.matmul(emb2, emb1.transpose(1, 2)), 1)
            R_vec = F.softmax(torch.matmul(emb2, emb_vec.transpose(1, 2)), 1)
            # R = self.get_adjacency(emb2, emb1)  # [1, 169, 169]
            # R_vec = self.get_adjacency(emb2, emb_vec)  # [1, 169, 169]
            # R = (R - R.min()) / (R.max() - R.min())
            # R1 = torch.where(R >= 0.8, torch.full_like(R, 1), torch.full_like(R, 0))
            # R2 = torch.where(R >= 0.6,torch.full_like(R, 1), torch.full_like(R, 0))
            # R_vec = (R_vec - R_vec.min()) / (R_vec.max() - R_vec.min())
            # R_vec = torch.where(R_vec >= 0.5, torch.full_like(R_vec, 1), torch.full_like(R_vec, 0))
            # if i == self.gnn_layer - 2:
            # cross_graph = getattr(self, 'cross_graph_{}'.format(i))
            # emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R.transpose(1, 2), emb2)), dim=-1))  # [1, 2704, 512]
            # emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R, emb1)), dim=-1))  # [1, 2704, 512]
            # emb_vec_new = cross_graph(torch.cat((emb2, torch.bmm(R_vec, emb_vec)), dim=-1))
            emb1_new = emb1 + torch.bmm(R1, emb2)  # [1, 2704, 512]
            emb2_new = emb2 + torch.bmm(R2, emb1)  # [1, 2704, 512]
            emb_vec_new = emb2 + torch.bmm(R_vec, emb_vec)
            emb1 = emb1_new
            emb2 = emb2_new
            emb_vec = emb_vec_new

        # s = torch.zeros([w * h, w * h])
        # for i in range(self.gnn_layer):
        #     gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
        #     emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
        #                            [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
        #     #affinity = getattr(self, 'affinity_{}'.format(i))
        #     #R = affinity(emb2, emb1)  # R = Gq*Gs [1, 2704, 2704]
        #     R = self.get_adjacency(emb2, emb1)
        #     R = (R - R.min()) / (R.max() - R.min())
        #     if i == self.gnn_layer - 2:
        #         cross_graph = getattr(self, 'cross_graph_{}'.format(i))
        #         emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R.transpose(1, 2), emb2)), dim=-1))  # [1, 2704, 512]
        #         emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R, emb1)), dim=-1))  # [1, 2704, 512]
        #         emb1 = emb1_new
        #         emb2 = emb2_new
        #         A_s = self.get_adjacency(emb1, emb1)
        #         A_q = self.get_adjacency(emb2, emb2)
        # A_s = F.softmax(torch.matmul(emb1, emb1.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        # A_q = F.softmax(torch.matmul(emb2, emb2.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        # print(R)
        return emb2.reshape(1, -1, w1, h1) + emb_vec.reshape(1, -1, w1, h1)

    def forward_0(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################先将特征进行卷积操作,然后对mask下采样与特征相乘后,然后计算邻接矩阵,然后在进行图卷积等操作
        ###################
        b, dim, w, h = x.size()
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = x[0].unsqueeze(0) * pos_mask
        ##########################################################
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)
        # 先进行1*1conv
        if dim == 512:
            x = self.cls1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
            # x = self.relu(x)
        else:
            x = self.cls2(x)
            # x = self.relu(x)

        x_s = x[0].reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        A_s = self.get_adjacency(x_s, x_s)
        A_q = self.get_adjacency(x_q, x_q)
        # A_s = F.softmax(torch.matmul(x_s, x_s.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
        # A_q = F.softmax(torch.matmul(x_q, x_q.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        # 位置相近点
        # L_f = torch.zeros(w*h, w*h)
        emb1 = x_s
        emb2 = x_q  # [1, 2704, 256][1, 169, 256]
        # s = torch.zeros([w * h, w * h])
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_s.cuda(), emb1],
                                   [A_q.cuda(), emb2])  # [1, 2704, 512][1, 2704, 512]
            # affinity = getattr(self, 'affinity_{}'.format(i))
            # R = affinity(emb2, emb1)  # R = Gq*Gs [1, 2704, 2704]
            R = self.get_adjacency(emb2, emb1)
            R = (R - R.min()) / (R.max() - R.min())
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(R.transpose(1, 2), emb2)), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(R, emb1)), dim=-1))  # [1, 2704, 512]
                emb1 = emb1_new
                emb2 = emb2_new
                A_s = self.get_adjacency(emb1, emb1)
                A_q = self.get_adjacency(emb2, emb2)
                # A_s = F.softmax(torch.matmul(emb1, emb1.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]
                # A_q = F.softmax(torch.matmul(emb2, emb2.transpose(1, 2)), 1)  ## [1, 2704,2704][1, 169, 169]

        return R, emb2.reshape(1, -1, w, h)

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
            R = affinity(emb2, emb1)  # R=Gq*Gs [1, 2704, 2704]
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(
                    torch.cat((emb1.squeeze(), torch.bmm(R.transpose(1, 2), emb2).squeeze()), dim=-1))  # [1, 2704, 512]
                emb2_new = cross_graph(
                    torch.cat((emb2.squeeze(), torch.bmm(R, emb1).squeeze()), dim=-1))  # [1, 2704, 512]
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

    def forward_1(self, x, mask):  # [2, 512, 52, 52][1, 1, 416, 416]
        ###################复现局部变换那篇文章
        ###################
        b, dim, w, h = x.size()
        pos_mask = F.interpolate(mask, [w, h], mode='bilinear',
                                 align_corners=True)  # [1, 1, 13, 13]
        pos_node = x[0].unsqueeze(0) * pos_mask

        '''
        plt.imshow(pos_node[0][0].cpu().detach().numpy())
        plt.show()
        '''
        ##########################################################
        x = torch.cat([pos_node, x[1].unsqueeze(0)], 0)
        # 先进行1*1conv
        if dim == 512:
            x = self.cls1(x)  # [2, 256, 52, 52][2, 256, 13, 13]
            # x = self.relu(x)
        else:
            x = self.cls2(x)
            # x = self.relu(x)
        '''
        plt.imshow(x[0][0].cpu().detach().numpy())
        plt.show()
        '''
        ##################################################################
        # x = self.conv1(x)
        #########均值系列###########
        # vec_pos = torch.sum(torch.sum(pos_node * mask, dim=3), dim=2) / torch.sum(mask)#
        # vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, w, h)
        ##########################
        ##################################################################

        x_s = x[0].reshape(1, w * h, -1)  ## [1, 2704,2704][1, 169, 169]
        x_q = x[1].reshape(1, w * h, -1)  # [1, 2704, 256][1, 169, 256]
        ###通过当前特征进行计算R矩阵
        R = self.get_adjacency(x_q, x_s)  # [1,2704,2704]
        pos_mask_reshape = pos_mask.reshape(1, -1)
        mask_inver = torch.matmul(pos_mask_reshape.t(),
                                  torch.matmul(pos_mask_reshape, pos_mask_reshape.t()).pinverse())

        # mask_inver = (mask_inver - mask_inver.min()) / (mask_inver.max() - mask_inver.min())
        A = torch.matmul(R, mask_inver)

        # A = (A - A.min()) / (A.max() - A.min())

        '''
        import matplotlib.pyplot as plt
        plt.imshow(A.reshape(52, 52).cpu().detach().numpy())
        plt.show()
        '''
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
                                 align_corners=False)  # [1, 1, 13, 13]
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
        print(A.max())
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
