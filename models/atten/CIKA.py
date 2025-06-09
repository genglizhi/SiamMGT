# coding=gbk
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x

class MyDepthWiseConv(nn.Module):
    def __init__(self, ch, kernel_size, stride=1, padding=0, dilation=1):
        super(MyDepthWiseConv, self).__init__()
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
        self.k = kernel_size
        self.ch = ch
        self.w = nn.Parameter(torch.empty(ch, kernel_size, kernel_size))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, ksa):
        n, c, h, w = ksa.shape
        ksa = ksa.unsqueeze(4).transpose(1, 4).view(n, 1, h, w, self.k, self.k)
        x = self.unfold(x).reshape(n, self.ch, self.k, self.k, h, w)
        x = x.transpose(2, 4).transpose(3, 5)
        x = torch.einsum('nchwkj,ckj->nchw', x*ksa, self.w)
        x = x + self.b
        return x


class KSA(nn.Module):
    def __init__(self, ch, kernel_size=5, dilation=1):    #5  1
        super(KSA, self).__init__()
        self.conv = nn.Conv2d(ch, ch, (5, 5), padding=2*dilation, dilation=dilation, groups=ch)
        # self.conv = nn.Conv2d(ch, ch, (3, 3), padding=1, dilation=dilation, groups=ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, 2*kernel_size**2, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*kernel_size**2, kernel_size**2, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.mlp(x)
        return x


class KCA(nn.Module):
    def __init__(self, ch, dilation=1):   #1
        super(KCA, self).__init__()
        self.conv = nn.Conv2d(ch, ch, (5, 5), padding=2*dilation, dilation=dilation, groups=ch)
        # self.conv = nn.Conv2d(ch, ch, (3, 3), padding=1, dilation=dilation, groups=ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch//4, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//4, ch, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.mlp(x)
        return x


class UpConv(nn.Module):
    def __init__(self, chin, chout, dilation=1, kernel_size=5):
        super(UpConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(chin, chin, (kernel_size, kernel_size),
                                        padding=dilation*(kernel_size//2),
                                        groups=chin, dilation=(dilation, dilation))
        self.conv1 = nn.Conv2d(chin, chout, (1, 1))

    def forward(self, x, kca):
        x = self.depthwise_conv(x)
        x = self.conv1(x * kca)
        return x


class LowConv(nn.Module):
    def __init__(self, chin, chout, dilation=1, kernel_size=5):
        super(LowConv, self).__init__()
        self.depthwise_conv = MyDepthWiseConv(chin, kernel_size, padding=dilation*(kernel_size//2),
                                              dilation=dilation)
        self.conv1 = nn.Conv2d(chin, chout, (1, 1))

    def forward(self, x, ksa):
        x = self.depthwise_conv(x, ksa)
        x = self.conv1(x)
        return x

class fuConv(nn.Module):
    def __init__(self, chin, chout, dilation=1, kernel_size=5):
        super(fuConv, self).__init__()
        self.depthwise_conv = MyDepthWiseConv(chin, kernel_size, padding=dilation*(kernel_size//2),
                                              dilation=dilation)
        self.conv1 = nn.Conv2d(chin, chout, (1, 1))


    def forward(self, x, ksa, kca):
        x0 = x
        x = self.depthwise_conv(x, ksa)
        x = self.conv1(x * kca)
        x = self.conv1(x)
        return x + x0

class CIKAConv(nn.Module):
    def __init__(self, dilation=1):
        super(CIKAConv, self).__init__()
        self.lowconv1 = LowConv(32, 32, dilation)
        self.lowconv2 = LowConv(32, 32, dilation)
        # self.upconv = UpConv(32, 32, dilation)
        # self.fuConv = fuConv(32, 32, dilation)
        self.kca_v = KCA(32, dilation)
        self.kca_t = KCA(32, dilation)
        self.ksa_v = KSA(32, 5, dilation)
        self.ksa_t = KSA(32, 5, dilation)

    def forward(self, v, t):
        # kca_v = self.kca_v(v)
        # kca_t = self.kca_t(t)
        ksa_v = self.ksa_v(v)
        ksa_t = self.ksa_t(t)
        att_v = self.lowconv1(v, ksa_t)
        att_t = self.lowconv2(t, ksa_v)
        # upper = self.upconv(upper, kca)
        # att_v = self.fuConv(v, ksa_t, kca_t)
        # att_t = self.fuConv(t, ksa_v, kca_v)
        # upper = self.fuConv(upper, ksa, kca)
        return att_v, att_t


class CIKABlk(nn.Module):
    def __init__(self, size):
        super(CIKABlk, self).__init__()
        self.conv1 = CIKAConv(1)
        # self.conv2 = CIKAConv(2)
        # self.ln = LRN()
        # self.ln = nn.LayerNorm([bach, 32, size, size], elementwise_affine=False)
        self.ln = nn.LayerNorm([32, size, size], elementwise_affine=False)
        # self.ln4 = nn.LayerNorm([64, size, size], elementwise_affine=False)

    def forward(self, v, t):
        v1, t1 = self.conv1(v, t)
        v1, t1 = self.ln(v1), self.ln(t1)
        # v2, t2 = self.conv2(v1, t1)
        # lower, upper = F.relu(lower2 + lower), F.relu(upper2 + upper)
        v, t = F.relu(v1), F.relu(t1)
        return v, t


# class CIKANetd(nn.Module):
#     def __init__(self, ch, size=25): # bach=32,
#         super(CIKANetd, self).__init__()
#         self.upper = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
#         self.lower = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
#         self.blk1 = CIKABlk(size)
#         self.blk2 = CIKABlk(size)
#         self.blk3 = CIKABlk(size)
#         self.blk4 = CIKABlk(size)
#         self.fuse = nn.Conv2d(64, ch, (3, 3), padding=(1, 1))
#
#     def forward(self, ms, pan):
#         # lms = F.interpolate(ms, scale_factor=4, mode='bicubic')
#         # _input = torch.subtract(pan, ms)
#         upper_features = F.relu(self.upper(_input))
#         lower_features = F.relu(self.lower(ms))
#         lower_features, upper_features = self.blk1(lower_features, upper_features)
#         lower_features, upper_features = self.blk2(lower_features, upper_features)
#         lower_features, upper_features = self.blk3(lower_features, upper_features)
#         lower_features, upper_features = self.blk4(lower_features, upper_features)
#         features = torch.cat([upper_features, lower_features], dim=1)
#         # sr = self.fuse(features)+lms
#         sr = self.fuse(features) + ms
#         return sr
class CIKANetd(nn.Module):
    def __init__(self, ch, size=25):
        super(CIKANetd, self).__init__()
        self.upper = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.lower = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.blk1 = CIKABlk(size)
        # self.blk2 = CIKABlk(size)
        # self.blk3 = CIKABlk(size)
        # self.blk4 = CIKABlk(size)
        self.fusev = nn.Conv2d(32, ch, (3, 3), padding=(1, 1))
        self.fuset = nn.Conv2d(32, ch, (3, 3), padding=(1, 1))
    def forward(self, v, t):
        # lms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        # _input = torch.subtract(pan, ms)
        v_features = F.relu(self.upper(v))
        t_features = F.relu(self.lower(t))
        v_features, t_features = self.blk1(v_features, t_features)
        # v_features, t_features = self.blk2(v_features, t_features)
        # v_features, t_features = self.blk3(v_features, t_features)
        # v_features, t_features = self.blk4(v_features, t_features)
        # features = torch.cat([v_features, t_features], dim=1)
        # v_features = v_features + v
        # t_features = t_features + t
        v_features = self.fusev(v_features) + v
        t_features = self.fuset(t_features) + t
        # features = torch.cat([v_features, t_features], 3)
        # sr = self.fuse(features) + ms
        # return features
        return v_features, t_features
class CIKANett(nn.Module):
    def __init__(self, ch, size=13):
        super(CIKANett, self).__init__()
        self.upper = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.lower = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.blk1 = CIKABlk(size)
        # self.blk2 = CIKABlk(size)
        # self.blk3 = CIKABlk(size)
        # self.blk4 = CIKABlk(size)
        self.fusev = nn.Conv2d(32, ch, (3, 3), padding=(1, 1))
        self.fuset = nn.Conv2d(32, ch, (3, 3), padding=(1, 1))
    def forward(self, v, t):
        # lms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        # _input = torch.subtract(pan, ms)
        v_features = F.relu(self.upper(v))
        t_features = F.relu(self.lower(t))
        v_features, t_features = self.blk1(v_features, t_features)
        # v_features, t_features = self.blk2(v_features, t_features)
        # v_features, t_features = self.blk3(v_features, t_features)
        # v_features, t_features = self.blk4(v_features, t_features)
        # features = torch.cat([v_features, t_features], dim=1)
        # v_features = v_features + v
        # t_features = t_features + t
        v_features = self.fusev(v_features) + v
        t_features = self.fuset(t_features) + t
        # sr = self.fuse(features) + ms
        # return features
        return v_features, t_features
'''class CIKANett(nn.Module):
    def __init__(self, ch, size=13):
        super(CIKANett, self).__init__()
        self.upper = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.lower = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.blk1 = CIKABlk(size)
        self.blk2 = CIKABlk(size)
        self.blk3 = CIKABlk(size)
        self.blk4 = CIKABlk(size)
        self.fuse = nn.Conv2d(64, ch, (3, 3), padding=(1, 1))

    def forward(self, v, t):
        # lms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        # _input = torch.subtract(pan, ms)
        v_features = F.relu(self.upper(v))
        t_features = F.relu(self.lower(t))
        v_features, t_features = self.blk1(v_features, t_features)
        v_features, t_features = self.blk2(v_features, t_features)
        # v_features, t_features = self.blk3(v_features, t_features)
        # v_features, t_features = self.blk4(v_features, t_features)
        features = torch.cat([v_features, t_features], dim=1)
        features = self.fuse(features)
        # sr = self.fuse(features) + ms
        return features'''

# if __name__ == '__main__':
#     x = torch.randn(1, 256, 13, 13)
#     y = torch.randn(1, 256, 13, 13)
#     model   = CIKANett(256)
#     sr = model(x,y)
#     print(sr.shape)