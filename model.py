import math
import torch
import torch.nn as nn
from torch.nn import init, Softmax
import torch.nn.functional as F
from utils import weights_init_classifier, weights_init_kaiming
from resnet import resnet50

class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


class visible_module(nn.Module):
    def __init__(self, pretrained=True):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        # x = self.visible.layer1(x)
        # x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        # x = self.thermal.layer1(x)
        # x = self.thermal.layer2(x)
        return x


class base_module12(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module12, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        return x


class base_module34(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module34, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channel, 2) + b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        # self.channel_attention1 = nn.Sequential(
        #     nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )
        # self.channel_attention2 = nn.Sequential(
        #     nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )
        self.channel_attention1 = ECA(dim)
        self.channel_attention2 = ECA(dim)

        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)

        x1_IN = self.IN(x1)
        x1_ = x1 - x1_IN
        # pooled1 = F.avg_pool2d(x1_, x1_.size()[2:])
        mask1 = self.channel_attention1(x1_)
        x1 = x1_ * mask1 + x1_IN

        x2_IN = self.IN(x2)
        x2_ = x2 - x2_IN
        # pooled2 = F.avg_pool2d(x2_, x2_.size()[2:])
        mask2 = self.channel_attention2(x2_)
        x2 = x2_ * mask2 + x2_IN

        x1 = x1 * mask2
        x2 = x2 * mask1

        x = torch.cat((x1, x2), 1)

        return x



class EMA(nn.Module):
    def __init__(self, channels):
        super(EMA, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        x2 = self.conv3x3(x)
        x11 = self.softmax(self.agp(x1).reshape(b, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b, c, -1)
        x21 = self.softmax(self.agp(x2).reshape(b, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b, c, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)
        return (x * weights.sigmoid()).reshape(b, c, h, w)


class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.base12_v = base_module12(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base12_t = base_module12(pretrained=pretrained)

        self.base34_1 = base_module34(pretrained=pretrained)
        self.base34_2 = base_module34(pretrained=pretrained)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.relu = nn.ReLU()
        self.pool = GeMP()
        # self.gap = nn.AdaptiveAvgPool2d(1)

        self.mam3 = MAM(512)
        self.mam4 = MAM(1024)
        self.ema1 = EMA(2048)
        self.ema2 = EMA(2048)
        self.ema3 = EMA(2048)

        self.bottleneck_p1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck_p1.bias.requires_grad_(False)  # no shift
        self.bottleneck_p1.apply(weights_init_kaiming)
        self.classifier_p1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier_p1.apply(weights_init_classifier)

        self.bottleneck_p2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck_p2.bias.requires_grad_(False)  # no shift
        self.bottleneck_p2.apply(weights_init_kaiming)
        self.classifier_p2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier_p2.apply(weights_init_classifier)

        self.bottleneck_p3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck_p3.bias.requires_grad_(False)  # no shift
        self.bottleneck_p3.apply(weights_init_kaiming)
        self.classifier_p3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier_p3.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible(x1)
            x2 = self.thermal(x2)

            x1 = self.base12_v(x1)
            x2 = self.base12_t(x2)

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible(x1)
            x = self.base12_v(x)
        elif modal == 2:
            x = self.thermal(x2)
            x = self.base12_t(x)

        xm = self.base34_1.base.layer3(x)
        xm = self.mam3(xm)
        xm = self.base34_1.base.layer4(xm)
        xm = self.mam4(xm)

        xl = self.base34_2.base.layer3(x)
        xl = self.base34_2.base.layer4(xl)
        p1, p2, p3 = xl.chunk(3, 2)
        p1 = self.ema1(p1)
        p2 = self.ema2(p2)
        p3 = self.ema3(p3)
        xl = torch.cat((p1, p2, p3), 2)

        x = torch.cat((xm, xl), 0)

        x = self.relu(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x_pool = self.pool(x)
        feat = self.bottleneck(x_pool)

        if self.training:
            p1_pool = self.avgpool(p1)
            p1_pool = p1_pool.view(p1_pool.size(0), p1_pool.size(1))
            p2_pool = self.avgpool(p2)
            p2_pool = p2_pool.view(p2_pool.size(0), p2_pool.size(1))
            p3_pool = self.avgpool(p3)
            p3_pool = p3_pool.view(p3_pool.size(0), p3_pool.size(1))
            feat_p1 = self.bottleneck_p1(p1_pool)
            feat_p2 = self.bottleneck_p2(p2_pool)
            feat_p3 = self.bottleneck_p3(p3_pool)

            return x_pool, self.classifier(feat), self.classifier_p1(feat_p1), self.classifier_p2(feat_p2), self.classifier_p3(feat_p3)
        else:
            return F.normalize(x_pool, p=2.0, dim=1), F.normalize(feat, p=2.0, dim=1)
