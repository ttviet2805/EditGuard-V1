import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConvINRelu(nn.Module):
    """Depthwise → InstanceNorm → ReLU → Pointwise → InstanceNorm → ReLU"""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        # depthwise conv
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.dw_norm = nn.InstanceNorm2d(in_ch)
        self.dw_act = nn.ReLU(inplace=True)
        # pointwise conv
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_norm = nn.InstanceNorm2d(out_ch)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_norm(self.dw(x)))
        x = self.pw_act(self.pw_norm(self.pw(x)))
        return x


class DSConvBlock(nn.Module):
    """tương tự ConvBlock nhưng dùng DSConvINRelu"""
    def __init__(self, in_ch, out_ch, blocks=1, stride=1):
        super().__init__()
        layers = []
        # layer đầu với stride có thể ≠1
        if blocks >= 1:
            layers.append(DSConvINRelu(in_ch, out_ch, stride=stride))
        # các layer tiếp theo luôn stride=1 và giữ số channel
        for _ in range(blocks - 1):
            layers.append(DSConvINRelu(out_ch, out_ch, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)