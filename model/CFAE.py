import math
import torch
import torch.nn as nn
from thop import profile

'''
     Cross-Feature Attention Enhancement Module
'''

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

def kernel_size(in_channel):

    k = int((math.log2(in_channel) + 1) // 2)
    if k % 2 == 0:
        return k + 1
    else:
        return k


class CFAE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.ca = ChannelAttention(in_channel)
        self.conv1 = nn.Conv2d(4 * in_channel, in_channel, kernel_size=1, bias=False)
        self.local_refine = nn.Conv2d(in_channel, in_channel, kernel_size=3,
                                      padding=1, groups=in_channel, bias=False)
        self.alfa = nn.Parameter(torch.tensor(0.5))
        self.bate = nn.Parameter(torch.tensor(0.5))

    def forward(self, t1, t2):
        # b, c, h, w = t1.size()
        t1_channel_avg = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max = self.max_pool(t1)
        t2_channel_avg = self.avg_pool(t2)
        t2_channel_max = self.max_pool(t2)

        channel_pool = torch.cat(
            [t1_channel_avg, t1_channel_max, t2_channel_avg, t2_channel_max],  dim=1)
        tmp = self.conv1(channel_pool)
        t2_channel_att = self.ca(tmp) * tmp
        t2_local = self.local_refine(t2)
        t2_enhanced = t2 + self.bate * t2_channel_att * t2 + self.alfa * t2_local
        return t2_enhanced



if __name__ == '__main__':
    model = CFAE(64)
    inputs1 = torch.rand(1, 64, 128, 128)
    inputs2 = torch.rand(1, 64, 128, 128)
    output = model(inputs1, inputs2)
    flops, params = profile(model, (inputs1, inputs2))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

