from .deeplabv3 import DeepLabV3 as TEST_DLV3
from .deeplabv2 import DeepLabV2 as TEST_DLV2
from .deeplabv2_speeding import Res_Ms_Deeplab as TEST_Ms_DLV2
from .deeplabv3_xception import create_DLX_V3_pretrained
from .deeplabv3_resnet import create_DLR_V3_pretrained
from .dilated_linknet import DilatedLinkNet34 as TEST_DiLinknet
from .icnet import icnet as TEST_icnet
from .linknext import LinkNext as TEST_Linknext
from .standard_fc_densenets import FCDenseNet103 as TEST_FCDensenet
from .psp_saeed import PSPNet as TEST_PSPNet2
from .tiramisu_test import FCDenseNet57 as TEST_Tiramisu57
