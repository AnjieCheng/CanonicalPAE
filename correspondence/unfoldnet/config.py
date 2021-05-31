import os
from correspondence.encoder import encoder_dict
from correspondence.unfoldnet import models, training #, generation
from correspondence import data

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Unfold Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    encoder = cfg['model']['encoder']
    z_dim = cfg['model']['z_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    encoder = encoder_dict[encoder](
        c_dim=z_dim,
        **encoder_kwargs
    )

    Fold = models.decoder_dict['ImplicitFun'](z_dim=z_dim)
    Unfold = models.decoder_dict['ImplicitFun'](z_dim=z_dim)

    model = models.ImplicitNet(encoder=encoder, fold=Fold, unfold=Unfold, device=device)

    return model

def get_trainer(model, optimizer, out_dir, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    # out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir,
        config=cfg,
    )

    return trainer


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    if cfg['data']['dataset'] == 'KeypointNet':
        fields = {}
        points_transform = data.SubsamplePointcloud(2048)
        fields['points'] = data.KpnPointsField(
            cfg['data']['points_folder'], points_transform,
            with_transforms=False,
            with_rotation=cfg['data']['with_rotation'],
            angle_sigma=cfg['data']['angle_sigma'],
            angle_clip=cfg['data']['angle_clip']
        )     

        
        if mode in ('val', 'test'):
            eval_points_transform = data.SubsamplePointcloud(2048)
            fields['eval_pts'] = data.KpnPointsField(
                cfg['data']['points_folder'], None,
                with_transforms=False, with_kp=True,
                with_rotation=cfg['data']['with_rotation'],
                angle_sigma=cfg['data']['angle_sigma'],
                angle_clip=cfg['data']['angle_clip']
            )

    return fields