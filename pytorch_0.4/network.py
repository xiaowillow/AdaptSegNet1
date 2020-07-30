
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils import optim_or_not



def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = True)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = True)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = True)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out











class ResNet(nn.Module):#Bottleneck,[3, 4, 23, 3], num_classes
    def __init__(self, block, layers, args):
        num_classes=2

        self.args=args
        self.stop_layer='layer4'# layer that is not load from pretrained model
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)#
        self.bn1 = nn.BatchNorm2d(64, affine = True)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # 
        self.layer1 = self._make_layer(block, 64, layers[0])#
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)#
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)#


        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias = True),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.5))



        self.layer55_0 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))











        self.gau1=GraphAttUnit(1536)
        self.gau2 = GraphAttUnit(1536)
        self.gau3 = GraphAttUnit(1536)

        self.layer6_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.5)
                                      )

        self.layer6_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer6_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

        )

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.layer9 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = True

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = True))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)



    def gau_compute(self,gau,query_feat,support_feat,support_embedding,support_mask):
        out=gau(query_feat,support_feat,support_embedding,support_mask)
        return out




    def forward(self, query_rgb,support_rgb,support_mask):

        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb
        query_rgb = self.layer3(query_rgb)
        # query_rgb = self.layer4(query_rgb)
        query_rgb_for_attention=torch.cat([query_feat_layer2,query_rgb],dim=1)

        query_rgb = self.layer5(query_rgb_for_attention)

        feature_size = query_rgb.shape[-2:]



        #side branch,get latent embedding z
        support_rgb = self.conv1(support_rgb)
        support_rgb = self.bn1(support_rgb)
        support_rgb = self.relu(support_rgb)
        support_rgb = self.maxpool(support_rgb)
        support_rgb = self.layer1(support_rgb)
        support_rgb = self.layer2(support_rgb)
        support_feat_layer2 = support_rgb
        support_rgb = self.layer3(support_rgb)

        #support_rgb = self.layer4(support_rgb)
        support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)

        support_embedding=self.layer5(support_rgb)
        support_mask = F.interpolate(support_mask, support_rgb.shape[-2:], mode='bilinear',align_corners=True)
        z = self.get_z(support_mask, support_embedding, feature_size) # this serve as the gau with equal attention
        support_for_attention=support_rgb#*support_mask
        support_embedding=support_embedding*support_mask

        out1=self.gau_compute(self.gau1,
                          query_rgb_for_attention,
                          support_for_attention,support_embedding,support_mask)

        out2 = self.gau_compute(self.gau2,
                            F.adaptive_avg_pool2d(query_rgb_for_attention,output_size=4),
                            support_for_attention, support_embedding,support_mask)

        out3 = self.gau_compute(self.gau3,
                            F.adaptive_avg_pool2d(query_rgb_for_attention, output_size=8),
                            support_for_attention, support_embedding,support_mask)

        out0 = self.layer55_0(torch.cat([query_rgb, z], dim=1))
        out1=self.layer55_0(torch.cat([query_rgb,out1],dim=1))
        out2 = self.layer55_0(torch.cat([query_rgb, F.interpolate(out2,feature_size,mode='bilinear',align_corners=True)], dim=1))
        out3 = self.layer55_0(torch.cat([query_rgb, F.interpolate(out3,feature_size,mode='bilinear',align_corners=True)], dim=1))


        out = out0+out1+out2+out3
        out = out + self.residule1(out)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        global_feature = F.avg_pool2d(out, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat(
            [global_feature, self.layer6_1(out), self.layer6_2(out), self.layer6_3(out), self.layer6_4(out)], dim=1)
        out = self.layer7(out)
        out = self.layer9(out)


        return out


    def get_z(self,support_mask,support_rgb,feature_size):
        support_mask = F.interpolate(support_mask, support_rgb.shape[-2:], mode='bilinear',align_corners=True)
        h, w = support_rgb.shape[-2:][0], support_rgb.shape[-2:][1]

        area = F.avg_pool2d(support_mask, support_rgb.shape[-2:]) * h * w + 0.0005
        z = support_mask * support_rgb  # elementwise multiplycation

        z = F.avg_pool2d(input=z,
                         kernel_size=support_rgb.shape[-2:]) * h * w / area  # avg pool to spatially fuse embeddings

        z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat

        return z



    def turn_off(self,model):
        optim_or_not(model.module.conv1, False)
        optim_or_not(model.module.layer1, False)
        optim_or_not(model.module.layer2, False)
        optim_or_not(model.module.layer3, False)


    def get_10x_lr_params(self,model):
        """
        return all the layers to optimize
        """

        b = []

        b.append(model.module.gau1.parameters())
        b.append(model.module.gau2.parameters())
        b.append(model.module.gau3.parameters())

        b.append(model.module.layer5.parameters())

        b.append(model.module.layer55_0.parameters())


        b.append(model.module.layer6_0.parameters())
        b.append(model.module.layer6_1.parameters())
        b.append(model.module.layer6_2.parameters())
        b.append(model.module.layer6_3.parameters())
        b.append(model.module.layer6_4.parameters())
        b.append(model.module.layer7.parameters())
        b.append(model.module.layer9.parameters())
        b.append(model.module.residule1.parameters())
        b.append(model.module.residule2.parameters())
        b.append(model.module.residule3.parameters())


        for j in range(len(b)):
            for i in b[j]:
                yield i







def Res_Deeplab(num_classes=2):
    model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes)
    return model



class GraphAttUnit(nn.Module):
    def __init__(self, in_channels):
        super(GraphAttUnit, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = 256
        self.temperature=4
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.Dropout=nn.Dropout(0.1)


    def forward(self, x_query,x_support,support_embedding,support_mask):
        batch_size = x_query.size(0)
        g_x = support_embedding.view(batch_size, 256, -1)  # (batch,C//2,H2*W2)
        g_x = g_x.permute(0, 2, 1)  # (batch,H2*W2,C//2)
        theta_x = self.theta(x_query).view(batch_size, self.inter_channels, -1)  # (batch,C//2,H1*W1)
        theta_x = theta_x.permute(0, 2, 1)  # (batch,H1*W1,C//2)
        phi_x =  self.phi(x_support).view(batch_size, self.inter_channels, -1)  # (batch,C//2,H2*W2)
        f = torch.matmul(theta_x, phi_x)  # (batch,H1*W1,H2*W2)\
        support_mask=support_mask.view(batch_size,1,-1)
        f=f*support_mask
        f = f.masked_fill(f == 0, -1e9)
        f=f/self.temperature
        f_div_C = F.softmax(f, dim=-1)  # normalize the last dim by softmax
        f_div_C=self.Dropout(f_div_C)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, 256, *x_query.size()[2:])



        return y





if __name__ == '__main__':
    import torchvision


    def load_resnet50_param(model, stop_layer='layer4'):
        resnet50 = torchvision.models.resnet50(pretrained=True)
        saved_state_dict = resnet50.state_dict()
        new_params = model.state_dict().copy()

        for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer

            i_parts = i.split('.')

            if not i_parts[0] == stop_layer:

                new_params['.'.join(i_parts)] = saved_state_dict[i]
            else:
                break
        model.load_state_dict(new_params)
        model.train()
        return model


    model = Res_Deeplab(num_classes=2).cuda()
    model = load_resnet50_param(model)

    query_rgb = torch.FloatTensor(1, 3, 321, 321).cuda()
    support_rgb = torch.FloatTensor(1, 3, 321, 321).cuda()
    support_mask = torch.FloatTensor(1, 1, 321, 321).cuda()


    pred = (model(query_rgb, support_rgb, support_mask))




