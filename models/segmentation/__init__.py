from .duc_hdc import ResNetDUC, ResNetDUCHDC
from .fcn16s import FCN16VGG
from .fcn32s import FCN32VGG
from .fcn8s import FCN8s
from .fusionnet import FusionNet
from .gcn import GCN
from .psp_net import PSPNet
from .seg_net import SegNet
from .u_net import UNet
from .deeplab import Res_Deeplab, Res_Ms_Deeplab, get_1x_lr_params_NOscale, get_10x_lr_params
from .carvana_unet import UNet128, UNet256, UNet512, UNet1024
from .unet_dilated import uNetDilated
from .pspnet_lex import PSPNetLex
from .resnet_gcn import ResnetGCN
from .unet_simple import UNetSimple
from .unet_stack import UNet960, UNet_stack
from .unet_elu import UNet_Elu
from .unet_res import UNetRes
from .deeplab_lg_fov import Deeplab_LG_FOV
from .resnet_dilated import Resnet18_8s, Resnet18_16s, Resnet18_32s, Resnet34_8s, Resnet34_16s, Resnet34_32s, Resnet50_8s, Resnet50_32s, Resnet101_8s
