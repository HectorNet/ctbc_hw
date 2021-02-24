import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torchsummary import summary

import data_utils

# torch.set_grad_enabled(False)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.conv0 = nn.Conv2d(inplanes, planes, 3, padding=1, stride=(2 if downsample else 1))
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_shortcut = nn.Conv2d(inplanes, planes, 3, padding=1, stride=(2 if downsample else 1))
    
    def forward(self, x):
        y = self.conv0(x)
        y = self.bn(y)
        y = self.relu(y)
        
        y = self.conv1(y)
        y = self.bn(y)

        shortcut = self.conv_shortcut(x)

        y += shortcut
        y = self.relu(y)

        return y


class Model(nn.Module):
    def __init__(self, char_count=4803, in_channels=1, out_channels=1):
        super(Model, self).__init__()
        self.res0 = ResBlock(1, 32, downsample=True)
        self.res1 = ResBlock(32, 32*4)
        self.res2 = ResBlock(32*4, 32*16)

        self.res3 = ResBlock(32*16, 32*32, downsample=True)
        self.res4 = ResBlock(32*32, 32*64)
        
        self.conv = nn.Conv2d(32*64, char_count, 3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d([1, 3])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.res0(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)

        y = self.conv(y)
        y = self.relu(y)
        y = self.global_avgpool(y)
        y = self.softmax(y)
        return y

if __name__ == '__main__':
    model = Model(char_count=data_utils.char_count)
    x = torch.from_numpy(np.random.normal(size=[1, 1, 50, 150]).astype(np.float32))
    y = model(x)
    output = torch.squeeze(torch.argmax(y, dim=1))
    print(x.size(), y.size(), output)

    # print(model)
    print(f'total parameters: {sum(p.numel() for p in model.parameters())}')
    summary(model, (1, 50, 150))