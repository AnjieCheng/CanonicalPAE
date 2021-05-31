from correspondence.encoder import (
    pointnet, # conv, 
)


encoder_dict = {
    # 'simple_conv': conv.ConvEncoder,
    # 'resnet18': conv.Resnet18,
    # 'resnet34': conv.Resnet34,
    # 'resnet50': conv.Resnet50,
    # 'resnet101': conv.Resnet101,
    # 'pointnet_simple': pointnet.SimplePointnet,
    # 'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet_atlas': pointnet.PointNetfeat,
    # 'vgg16': conv.VGG16
}
