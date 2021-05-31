# from im2mesh import icp
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from functools import partial

class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader, epoch_it, save=None):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        raw_np_list = defaultdict(partial(np.ndarray, 0))

        for data in tqdm(val_loader):
            eval_step_dict, eval_step_raw_np  = self.eval_step(data, epoch_it)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

            for k, v in eval_step_raw_np.items():
                raw_np_list[k] = np.append(raw_np_list[k], v, axis=0)

        eval_dict = {k: np.nanmean(v) for k, v in eval_list.items()}
        if save is not None:
            save_file = os.path.join(save, str(epoch_it)+'_test.npy')
            with open(save_file, 'wb') as f:
                np.save(f, raw_np_list)

            save_file = os.path.join(save, 'best.npy')
            with open(save_file, 'wb') as f:
                np.save(f, raw_np_list)
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
