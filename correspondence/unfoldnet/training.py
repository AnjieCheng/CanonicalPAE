import os
import numpy as np
from tqdm import trange
from collections import OrderedDict
import scipy.io
import scipy.spatial
import torch
import torch.optim as optim

from correspondence.training import BaseTrainer
from correspondence.template import SphereTemplate, SquareTemplate
from correspondence.utils import visualize as vis
from correspondence.unfoldnet.loss import *

from correspondence.utils.common import *

class Trainer(BaseTrainer):
    ''' Trainer object for the Implicit Network.

    Args:
        model (nn.Module): Implicit Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pcl',
                 vis_dir=None, eval_sample=False, config=None):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.eval_sample = eval_sample
        self.smoothed_total_loss = 0
        self.config = config

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, epoch_it, it):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, epoch_it, it)
        loss.backward()
        if self.config['training']['no_training']:
            return loss.item()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data, epoch_it, save=None):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        if self.config['data']['dataset'] == 'KeypointNet':
            inputs = data.get('eval_pts').to(self.device)
            kp = data.get('eval_pts.kp').to(self.device)
            inputs_t = data.get('eval_pts.target').to(self.device)
            kp_t = data.get('eval_pts.target_kp').to(self.device)

            batch_size = inputs.size(0)
            dis_list = np.array([])

            for b in range(batch_size):
                shape = torch.cat([inputs[b].unsqueeze(0), inputs_t[b].unsqueeze(0)], dim=0)

                latent = self.model.Encoder(shape)
                unfold_pts = self.model.Unfold(latent, shape)
                cross_rec_shapes = self.model.Fold(latent, torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0))

                e_shape_a = cross_rec_shapes[0:1].squeeze(0).data.cpu().numpy()
                e_shape_b = cross_rec_shapes[1:2].squeeze(0).data.cpu().numpy()
                shape_a = shape[0:1].squeeze(0).data.cpu().numpy()
                shape_b = shape[1:2].squeeze(0).data.cpu().numpy()
                land_a = kp[b].data.cpu().numpy()
                land_b = kp_t[b].data.cpu().numpy()

                # KNN
                nn_land_on_shape_b = shape_b[scipy.spatial.KDTree(e_shape_a).query(land_a)[1]]
                nn_land_on_shape_a = shape_a[scipy.spatial.KDTree(e_shape_b).query(land_b)[1]]

                dis_land_a = np.sqrt(np.sum((land_a - nn_land_on_shape_a)**2, axis=1))
                dis_land_b = np.sqrt(np.sum((land_b - nn_land_on_shape_b)**2, axis=1))

                # # Normlization
                diameter_shape_a = np.sqrt(np.sum((np.max(shape_a, axis=0)-np.min(shape_a, axis=0))**2))
                diameter_shape_b = np.sqrt(np.sum((np.max(shape_b, axis=0)-np.min(shape_b, axis=0))**2))
                dis_land_a = dis_land_a/diameter_shape_a
                dis_land_b = dis_land_b/diameter_shape_b

                # remove none-existance keypoints
                for i in range(land_a.shape[0]):
                    if np.array_equal(land_a[i], [-999, -999, -999]) or np.array_equal(land_b[i], [-999, -999, -999]):
                        dis_land_a[i] = -1
                        dis_land_b[i] = -1

                dis_land_a = dis_land_a[dis_land_a != -1]
                dis_land_b = dis_land_b[dis_land_b != -1]

                dis_list = np.append(dis_list, dis_land_a, axis=0)
                dis_list = np.append(dis_list, dis_land_b, axis=0)

            rt = {'by_uv_dist': np.mean(dis_list),
                }

            raw = {'by_uv_dist': dis_list,
                }

        return rt, raw # retrun raw data

    def visualize(self, data, logger=None, it=None, epoch_it=None):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        if epoch_it <= 1000:
            return
        device = self.device
        self.model.eval()

        if self.config['data']['dataset'] == 'KeypointNet':
            inputs = data.get('eval_pts').to(self.device)
            colors = data.get('eval_pts.colors').to(self.device)
            kp = data.get('eval_pts.kp').to(self.device)
            inputs_t = data.get('eval_pts.target').to(self.device)
            colors_t = data.get('eval_pts.target_colors').to(self.device)
            kp_t = data.get('eval_pts.target_kp').to(self.device)
            # import pdb; pdb.set_trace()
            mesh_path_t = data.get('eval_pts.target_mesh_path')
            mesh_path = data.get('eval_pts.mesh_path')
            pts_loc = data.get('eval_pts.loc')
            pts_scale = data.get('eval_pts.scale')
            pts_loc_t = data.get('eval_pts.target_loc')
            pts_scale_t = data.get('eval_pts.target_scale')
            # mesh_pts = data.get('eval_pts.mesh_pts').to(self.device)
            # mesh_pts_t = data.get('eval_pts.target_mesh_pts').to(self.device)

            batch_size = inputs.size(0)
            dis_list = np.array([])

            # generate template
            template = SphereTemplate(device=self.device)
            grid = template.get_regular_points(device=self.device).squeeze().transpose(0,1)
            batch_p_2d = batch_sample_from_2d_grid(grid, 2048, batch_size, without_sample=True)

            for b in trange(batch_size):
                shape = torch.cat([inputs[b].unsqueeze(0), inputs_t[b].unsqueeze(0)], dim=0)

                latent = self.model.Encoder(shape)
                unfold_pts = self.model.Unfold(latent, shape)

                cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)
                _, cross_matched_batch = match_source_to_target_points(unfold_pts.detach(), cross_unfold_pts.detach(), device=self.device)
                cross_matched_a = cross_matched_batch[0:1].squeeze(0).data.cpu().numpy()
                cross_matched_b = cross_matched_batch[1:2].squeeze(0).data.cpu().numpy()

                cross_rec_shapes = self.model.Fold(latent, torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0))
                self_rec_shape = self.model.Fold(latent, unfold_pts)

                self_rec_shape_a = self_rec_shape[0:1].squeeze(0).data.cpu().numpy()
                self_rec_shape_b = self_rec_shape[1:2].squeeze(0).data.cpu().numpy()
                unfold_shape_a = unfold_pts[0:1].squeeze(0).data.cpu().numpy()
                unfold_shape_b = unfold_pts[1:2].squeeze(0).data.cpu().numpy()
                e_shape_a = cross_rec_shapes[0:1].squeeze(0).data.cpu().numpy()
                e_shape_b = cross_rec_shapes[1:2].squeeze(0).data.cpu().numpy()
                shape_a = shape[0:1].squeeze(0).data.cpu().numpy()
                shape_b = shape[1:2].squeeze(0).data.cpu().numpy()
                land_a = kp[b].data.cpu().numpy()
                land_b = kp_t[b].data.cpu().numpy()
                color_a = colors[b].data.cpu().numpy()
                color_b = colors_t[b].data.cpu().numpy()

                # KNN
                nn_land_on_shape_b = shape_b[scipy.spatial.KDTree(e_shape_a).query(land_a)[1]]
                nn_land_on_shape_a = shape_a[scipy.spatial.KDTree(e_shape_b).query(land_b)[1]]


    def compute_loss(self, data, epoch_it, it):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        shape = data.get('points').to(self.device).float()
        batch_size = shape.size(0)

        # Forward model
        latent = self.model.Encoder(shape)

        self.loss_dict = {}

        # forward
        unfold_pts = self.model.Unfold(latent, shape)

        # generate template
        template = SphereTemplate(device=device)
        grid = template.get_regular_points(device=device).squeeze().transpose(0,1)
        batch_p_2d = batch_sample_from_2d_grid(grid, 2048, batch_size, without_sample=True)

        # unfold loss
        use_full_loss = False if it >= 6000 and epoch_it >= 1500 else True
        unfold_loss_alpha = 20 if it >= 6000 and epoch_it >= 1500 else 10

        loss_unfold = unfold_loss(unfold_pts, batch_p_2d, full_loss=use_full_loss)
        self.loss_dict['loss_unfold'] = unfold_loss_alpha*loss_unfold

        if epoch_it >= 0:
            # self-reconstruction loss
            self_rec_shape = self.model.Fold(latent, unfold_pts)
            loss_sr = 1000*selfrec_loss(self_rec_shape, shape) + 10*CD_loss(self_rec_shape, shape) + EMD_loss(self_rec_shape, shape)
            self.loss_dict['loss_sr'] = loss_sr
        else:
            self.loss_dict['loss_sr'] = torch.zeros(1)

        if self.config['training']['no_cross'] == True:
            self.loss_dict['loss_cr'] = torch.zeros(1)
        else:
            if epoch_it >= 1500 and it >= 6000:
                # cross-reconstruction loss
                cross_unfold_pts = torch.cat((unfold_pts[1:,:,:], unfold_pts[:1,:,:]), dim=0)
                cross_rec_shapes = self.model.Fold(latent, cross_unfold_pts)
                loss_cr = 10 * CD_loss(cross_rec_shapes, shape)
                self.loss_dict['loss_cr'] = loss_cr
            else:
                self.loss_dict['loss_cr'] = torch.zeros(1)

        # organize loss
        self.loss = 0
        self.loss_dict_mean = OrderedDict()
        for loss_name, loss in self.loss_dict.items():
            single_loss = loss.mean()
            self.loss += single_loss
            self.loss_dict_mean[loss_name] = single_loss.item()

        self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.loss.item()
        
        return self.loss

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('loss_unfold', self.loss_dict['loss_unfold'].item()),
            ('loss_sr', self.loss_dict['loss_sr'].item()),
            ('loss_cr', self.loss_dict['loss_cr'].item()),
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.loss.item()),
        ])
        return sc_dict
