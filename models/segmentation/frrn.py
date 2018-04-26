# Source: https://github.com/meetshah1995/pytorch-semseg (MIT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# from ptsemseg.models.utils import *

frrn_specs_dic = {
    'A':
        {
            'encoder': [[3, 96, 2],
                        [4, 192, 4],
                        [2, 384, 8],
                        [2, 384, 16]],

            'decoder': [[2, 192, 8],
                        [2, 192, 4],
                        [2, 96, 2]],
        },

    'B':
        {
            'encoder': [[3, 96, 2],
                        [4, 192, 4],
                        [2, 384, 8],
                        [2, 384, 16],
                        [2, 384, 32]],

            'decoder': [[2, 192, 16],
                        [2, 192, 8],
                        [2, 192, 4],
                        [2, 96, 2]],
        }, }


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)


    return loss / float(batch_size)


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """
    def __init__(self, channels, kernel_size=3, strides=1):
        super(RU, self).__init__()

        self.conv1 = conv2DBatchNormRelu(channels, channels, k_size=kernel_size, stride=strides, padding=1)
        self.conv2 = conv2DBatchNorm(channels, channels, k_size=kernel_size, stride=strides, padding=1)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming

class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """
    def __init__(self, prev_channels, out_channels, scale):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels

        self.conv1 = conv2DBatchNormRelu(prev_channels + 32, out_channels, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, k_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s*self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode='nearest')
        z_prime = z + x

        return y_prime, z_prime


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323

    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=21, model_type=None):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.K = 64 * 512
        self.loss = functools.partial(bootstrapped_cross_entropy2d, K=self.K)

        self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48, kernel_size=3, strides=1))
            self.down_residual_units.append(RU(channels=48, kernel_size=3, strides=1))

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(48, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        self.merge_conv = nn.Conv2d(prev_channels + 32, 48,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        self.classif_conv = nn.Conv2d(48, self.n_classes,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv(x)

        prev_channels = 48
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear')
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([F.upsample(y, scale_factor=2, mode='bilinear'), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x
