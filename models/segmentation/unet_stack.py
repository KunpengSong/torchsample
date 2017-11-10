# Source: https://github.com/doodledood/carvana-image-masking-challenge/blob/master/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ConvBNReluStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ConvBNReluStack, self).__init__()

        in_dim = int(in_dim)
        out_dim = int(out_dim)

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        # nn.init.xavier_normal(self.conv.weight.data)

        self.bn = nn.InstanceNorm2d(out_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, inputs_):
        x = self.conv(inputs_)
        x = self.bn(x)
        x = self.activation(x)

        return x


class UNetDownStack(nn.Module):
    def __init__(self, input_dim, filters, kernel_size=3, pool=True):
        super(UNetDownStack, self).__init__()

        self.stack0 = ConvBNReluStack(input_dim, filters, 1, stride=1, padding=0)
        self.stack1 = ConvBNReluStack(filters, filters // 2, kernel_size, stride=1, padding=1)
        self.stack2 = ConvBNReluStack(filters // 2, filters, kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2) if pool else None

    def forward(self, inputs_):
        x = self.stack0(inputs_)
        x1 = self.stack1(x)
        x = x + self.stack2(x1)

        if self.pool:
            return x, self.pool(x)

        return x


class UNetUpStack(nn.Module):
    def __init__(self, input_dim, filters, kernel_size=3):
        super(UNetUpStack, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.stack0 = ConvBNReluStack(input_dim, filters, 1, stride=1, padding=0)
        self.stack1 = ConvBNReluStack(filters, filters // 2, kernel_size, stride=1, padding=1)
        self.stack2 = ConvBNReluStack(filters // 2, filters, kernel_size, stride=1, padding=1)
        self.stack3 = ConvBNReluStack(filters, filters, kernel_size, stride=1, padding=1)

    def forward(self, inputs_, down):
        x = self.upsample(inputs_)
        x = torch.cat([x, down], dim=1)

        y0 = self.stack0(x)
        y = self.stack1(y0)
        y = self.stack2(y)
        y = y0 + self.stack3(y)

        return y


class UNet_stack(nn.Module):
    def get_n_stacks(self, input_size):
        n_stacks = 0
        width, height = input_size
        while width % 2 == 0 and height % 2 == 0:
            n_stacks += 1
            width = width // 2
            height = height // 2

        return n_stacks

    def __init__(self, input_size, filters, kernel_size=3, max_stacks=6):
        super(UNet_stack, self).__init__()

        self.n_stacks = max(self.get_n_stacks(input_size), max_stacks)

        # dynamically create stacks
        self.down1 = UNetDownStack(3, filters, kernel_size)
        prev_filters = filters
        for i in range(2, self.n_stacks + 1):
            n = i
            layer = UNetDownStack(prev_filters, prev_filters * 2, kernel_size)
            layer_name = 'down' + str(n)
            setattr(self, layer_name, layer)
            prev_filters *= 2

        self.center = UNetDownStack(prev_filters, prev_filters * 2, kernel_size, pool=False)

        prev_filters = prev_filters * 3
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            layer = UNetUpStack(prev_filters, prev_filters // 3, kernel_size)
            layer_name = 'up' + str(n)
            setattr(self, layer_name, layer)
            prev_filters = prev_filters // 2

        self.classify = nn.Conv2d(prev_filters * 2 // 3, 1, kernel_size, stride=1, padding=1)
        # nn.init.xavier_normal(self.classify.weight.data)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)

        downs = [down1]

        # execute down nodes
        prev_down_pool = down1_pool
        for i in range(2, self.n_stacks + 1):
            layer_name = 'down' + str(i)
            layer = getattr(self, layer_name)
            down, prev_down_pool = layer(prev_down_pool)
            downs.append(down)

        center = self.center(prev_down_pool)

        # excute up nodes
        prev = center
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            matching_down = downs.pop()
            layer_name = 'up' + str(n)
            layer = getattr(self, layer_name)
            prev = layer(prev, matching_down)

        x = self.classify(prev)

        return torch.squeeze(x, dim=1)


class UNet960(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super(UNet960, self).__init__()

        # 960
        self.down1 = UNetDownStack(3, filters, kernel_size)
        # 480
        self.down2 = UNetDownStack(filters, filters * 2, kernel_size)
        # 240
        self.down3 = UNetDownStack(filters * 2, filters * 4, kernel_size)
        # 120
        self.down4 = UNetDownStack(filters * 4, filters * 8, kernel_size)
        # 60
        self.down5 = UNetDownStack(filters * 8, filters * 16, kernel_size)
        # 30
        self.down6 = UNetDownStack(filters * 16, filters * 32, kernel_size)
        # 15
        self.center = UNetDownStack(filters * 32, filters * 64, kernel_size, pool=False)
        # 15
        self.up6 = UNetUpStack(filters * 96, filters * 32, kernel_size)
        # 30
        self.up5 = UNetUpStack(filters * 48, filters * 16, kernel_size)
        # 60
        self.up4 = UNetUpStack(filters * 24, filters * 8, kernel_size)
        # 120
        self.up3 = UNetUpStack(filters * 12, filters * 4, kernel_size)
        # 240
        self.up2 = UNetUpStack(filters * 6, filters * 2, kernel_size)
        # 480
        self.up1 = UNetUpStack(filters * 3, filters, kernel_size)
        # 960
        self.classify = nn.Conv2d(filters, 1, kernel_size, stride=1, padding=1)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)
        down2, down2_pool = self.down2(down1_pool)
        down3, down3_pool = self.down3(down2_pool)
        down4, down4_pool = self.down4(down3_pool)
        down5, down5_pool = self.down5(down4_pool)
        down6, down6_pool = self.down6(down5_pool)

        center = self.center(down6_pool)

        up6 = self.up6(center, down6)
        up5 = self.up5(up6, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        x = self.classify(up1)

        return torch.squeeze(x, dim=1)


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, filters=16):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, filters, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),
            nn.Conv2d(filters * 2, filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),
            nn.Conv2d(filters * 4, filters * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters * 8, filters * 4, 3, padding=1),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters * 4, filters * 2, 3, padding=1),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters * 2, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters, 1, 3, padding=1)
        )

    def forward(self, input_):
        # print(input_.size())
        output = self.main(input_)
        return torch.squeeze(output, dim=1)