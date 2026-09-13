import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Embedding(nn.Module):
    def __init__(self, in_ch):
        super(Embedding, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
        )
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x, y):
        y = self.up(y)      # up20262300042
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return out

class DT_PCF(nn.Module):

    def __init__(self, channels, kernel=3, f=0.9, g=0.8, h=20., k=10., radio=4):
        super().__init__()
        self.f = nn.Parameter(torch.tensor(f))
        self.g = nn.Parameter(torch.tensor(g))
        self.h = nn.Parameter(torch.tensor(h))
        self.k  = k                      # 固定大斜率即可
        c4 = channels // radio

        self.embb = Embedding(channels)

        self.reduce = nn.Conv2d(channels, c4, 3, padding=1,bias=False)

        self.W = nn.Conv2d(c4, c4, kernel, padding=kernel//2, groups=c4, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(2 * c4, channels, 3, padding=1, bias=False),
        )
        self.gate = nn.Conv2d(c4, c4, kernel_size=3, padding=1, groups=c4, bias=False)
        self.alfa = nn.Parameter(torch.tensor(0.1))

    def __static__(self, x):
        B, C, H, W = x.shape
        global_mean = torch.mean(x, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        global_std = torch.std(x, dim=[2, 3], keepdim=True)    # (B, C, 1, 1)

        global_mean = global_mean.expand(B, C, H, W)
        global_std = global_std.expand(B, C, H, W)
        return (x - global_mean) / (global_std + 1e-8)

    def forward(self, x, g, pcnn_steps=4):
        x = self.embb(x, g)
        S = self.reduce(x)
        F = torch.zeros_like(S)
        T = torch.ones_like(S) * (self.h * 5)
        Y_list = []


        steps = pcnn_steps
        for _ in range(steps):
            F = self.f * F  + S + self.W(Y_list[-1] if Y_list else torch.zeros_like(S))
            Y = torch.sigmoid(self.k * (F - T))

            normalized_Y = self.__static__(Y)
            local_max = torch.nn.functional.max_pool2d(Y, 3, padding=1, stride=1)
            local_mean = torch.nn.functional.avg_pool2d(Y, 3, padding=1, stride=1)
            small_target_score = torch.clamp(local_max / (local_mean + 1e-8) - 1, 0, 1)
            gate_signal = self.gate(normalized_Y)
            enhanced_gate = gate_signal * (1 + small_target_score)

            T = self.g * T + self.h * Y + self.gate(Y) + self.alfa * enhanced_gate
            T = self.g * T + self.h * Y
            Y_list.append(Y)



        att = Y_list[-1]
        out = self.final(torch.cat([S, att], dim=1))

        return out

if __name__ == '__main__':
    model = DT_PCF(32)
    inputs = torch.rand(1, 32, 256, 256)
    inputs_g = torch.rand(1, 32, 128, 128)
    output = model(inputs, inputs_g)
    flops, params = profile(model, (inputs, inputs_g))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')