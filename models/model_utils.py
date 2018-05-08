from .classification import *
from .segmentation import *

def get_model(type, model_name, num_classes, input_size, pretrained_path=None):
    '''
    :param type: str
        one of {'classification', 'segmentation'}
    :param model_name: str
        name of the model
    :param num_classes: int
        number of classes to segment
    :param input_size: (int,int)
        what size of input the network will accept e.g. (256, 256), (512, 512)
    :param pretrained_path: str
        path to a pretrained network to load weights from [NOTE: currently unsupported]
    :return:
    '''
    print("Loading Model:   --   " + model_name + "  with number of classes: " + str(num_classes) + ' and input size: ' + str(input_size))

    if model_name == 'Enet':                                            # standard enet
        net = ENet(num_classes=num_classes)
    elif model_name == 'deeplabv2_ASPP':                                # Deeplab Atrous Convolutions
        net = DeepLabv2_ASPP(num_classes=num_classes, pretrained=True)
    elif model_name == 'deeplabv2_FOV':                                 # Deeplab FOV
        net = DeepLabv2_FOV(num_classes=num_classes, pretrained=True)
    elif model_name == 'deeplabv3':                                     # Deeplab V3!
        net = DeepLabv3(num_classes=num_classes, pretrained=True)
    elif model_name == 'deeplabv3_Plus':  # Deeplab V3!
        net = DeepLabv3_plus(num_classes=num_classes, pretrained=True)
    elif model_name == 'DRN_C_42':
        net = drn_c_42(pretrained=True, num_classes=num_classes)
    elif model_name == 'DRN_C_58':
        net = drn_c_58(pretrained=True, num_classes=num_classes)
    elif model_name == 'DRN_D_56':
        net = drn_d_56(pretrained=True, num_classes=num_classes)
    elif model_name == 'DRN_D_107':
        net = drn_d_107(pretrained=True, num_classes=num_classes)
    elif model_name == 'FPN':                                     # FPN
        net = FPN101()
    elif model_name == 'FRRN_A':                                     # FRRN
        net = frrn(n_classes=num_classes, model_type='A')
    elif model_name == 'FRRN_B':                                     # FRRN
        net = frrn(n_classes=num_classes, model_type='B')
    elif model_name == 'FusionNet':                                     # FusionNet
        net = FusionNet(num_classes=num_classes)
    elif model_name == 'GCN':                                           # GCN Resnet
        net = GCN(num_classes=num_classes)
    elif model_name == 'GCN_VisDa':                                     # Another GCN Implementation
        net = GCN_VisDa(num_classes=num_classes, input_size=input_size)
    elif model_name == 'GCN_Densenet':                                     # Another GCN Implementation
        net = GCN_DENSENET(num_classes=num_classes, input_size=input_size)
    elif model_name == 'GCN_PSP':                                     # Another GCN Implementation
        net = GCN_PSP(num_classes=num_classes, input_size=input_size)
    elif model_name == 'GCN_NASNetA':                                     # Another GCN Implementation
        net = GCN_NASNET(num_classes=num_classes, input_size=input_size)
    elif model_name == 'GCN_Resnext':                                     # Another GCN Implementation
        net = GCN_RESNEXT(num_classes=num_classes, input_size=input_size)
    elif model_name == 'Linknet':                                       # Linknet34
        net = LinkNet34(num_classes=num_classes)
    elif model_name == 'PSPNet':                                          # standard pspnet
        net = PSPNet(num_classes=num_classes, model='resnet152')
    elif model_name == 'Resnet_DUC':
        net = ResNetDUC(num_classes=num_classes, pretrained=True)
    elif model_name == 'Resnet_DUC_HDC':
        net = ResNetDUCHDC(num_classes=num_classes, pretrained=True)
    elif model_name == 'Resnet_GCN':                                    # GCN Resnet 2
        net = ResnetGCN(num_classes=num_classes)
    elif model_name == 'Retina_FPN':
        net = RetinaFPN101()
    elif model_name == 'Segnet':                                          # standard segnet
        net = SegNet(num_classes=num_classes, pretrained=True)
    elif model_name == 'Tiramisu67':                                     # Tiramisu
        net = FCDenseNet67(n_classes=num_classes)
    elif model_name == 'Tiramisu103':                                   # Tiramisu
        net = FCDenseNet103(n_classes=num_classes)
    elif model_name == 'Unet':                                          # standard unet
        net = UNet(num_classes=num_classes)
    elif model_name == 'UNet256':                                       # Unet for 256px square imgs
        net = UNet256(in_shape=(3,256,256))
    elif model_name == 'UNet512':                                       # Unet for 512px square imgs
        net = UNet512(in_shape=(3, 512, 512))
    elif model_name == 'UNet1024':                                      # Unet for 1024px square imgs
        net = UNet1024(in_shape=(3, 1024, 1024))
    elif model_name == 'UNet960':                                       # Another Unet specifically with 960px resolution
        net = UNet960(filters=12)
    elif model_name == 'unet_dilated':                                  # dilated unet
        net = uNetDilated(num_classes=num_classes)
    elif model_name == 'Unet_res':                                      # residual unet
        net = UNetRes(num_class=num_classes)
    elif model_name == 'UNet_stack':                                    # Stacked Unet variation with resnet connections
        net = UNet_stack(input_size=(input_size, input_size), filters=12)
    else:
        raise Exception('Combination of type: {} and model_name: {} is not valid'.format(type, model_name))

    return net

def get_supported_models(type):
    '''

    :param type: str
        one of {'classification', 'segmentation'}
    :return: list (strings) of supported models
    '''

    if type == 'segmentation':
        return ['Enet',
                'deeplabv2_ASPP',
                'deeplabv2_FOV',
                'deeplabv3',
                'deeplabv3_Plus',
                'DRN_C_42',
                'DRN_C_58',
                'DRN_D_56',
                'DRN_D_107',
                'FPN',
                'FRRN_A',
                'FRRN_B',
                'FusionNet',
                'GCN',
                'GCN_VisDa',
                'GCN_Densenet',
                'GCN_PSP',
                'GCN_NASNetA',
                'GCN_Resnext',
                'Linknet',
                'PSPNet',
                'Resnet_DUC',
                'Resnet_DUC_HDC',
                'Resnet_GCN',
                'Retina_FPN',
                'Segnet',
                'Tiramisu67',
                'Tiramisu103',
                'Unet',
                'UNet256',
                'UNet512',
                'UNet1024',
                'UNet960',
                'unet_dilated',
                'Unet_res',
                'UNet_stack']
    elif type == 'classification':
        return []
    else:
        raise Exception('Incorrect type specified. Expected one of: {classification, segmentation}')
