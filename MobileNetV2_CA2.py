import torch
import torch.nn as nn
import math

class MobileNetV2_CA(nn.Module):
    def __init__(self, num_classes=200, width_mult=1.0):
        super().__init__()
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 1],
            [6,  96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        input_channel = self._make_divisible(32 * width_mult, 8)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        for t, c, n, s in self.cfgs:
            output_channel = self._make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    SandGlassBlock(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, mip, 1, bias=False),
            nn.BatchNorm2d(mip),
            nn.Dropout(0.2),
            nn.Hardswish(inplace=True)
        )
        
        self.conv_h = nn.Conv2d(mip, inp, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, 1, bias=False)
        
    def forward(self, x):
        identity = x
        n,c,h,w = x.shape
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h

class SandGlassBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        
        layers = []
        layers.extend([
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Depth-wise卷积（SandGlass核心结构）
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            CoordAtt(hidden_dim),  # 在深度卷积后插入CA模块
        ])
        
        # Point-wise卷积降维
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)