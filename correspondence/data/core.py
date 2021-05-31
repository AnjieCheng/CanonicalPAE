import os
import logging
from torch.utils import data
from collections import defaultdict
import numpy as np
import yaml, h5py
from tqdm import tqdm
import json

from correspondence.utils.common import NAMES2ID, ID2NAMES


logger = logging.getLogger(__name__)

# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError



class KeypointNetDataset(data.Dataset):
    ''' KeypointNet dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None, categories=None, no_except=True, transform=None):
        ''' Initialization of the the KeypointNet dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If evaluate, load data pairs
        self.by_pair = True if split in ['val', 'test'] else False

        # If categories is None, use all subfolders
        # TODO: Can define all here
        if categories is None:
            categories = os.listdir(dataset_folder)
            self.categories = categories
        else:
            self.categories = categories

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            logger.warning('Dataset metadata file does not exist, setting name to n/a.')
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models from split file
        self.tmp_models = []
        split_pool = defaultdict(list) # for sanity check
        split_file = os.path.join(dataset_folder, 'splits', split + '.txt')
        with open(split_file, 'r') as f:
            for line in f.read().split('\n'):
                if '-' in line:
                    c, m = line.split('-')
                    split_pool[c].append(m) # for sanity check

                    if c in self.categories:
                        self.tmp_models += [{'category': c, 'model': m, 'model_t': None}]

            import random
            random.Random(1111).shuffle(self.tmp_models) #uncomment this (only for confidence chair)

        print("split: %s, length: %d" % (split, len(self.tmp_models)))

        if self.by_pair:
            self.models = []
            for i in range(len(self.tmp_models)):
                for j in range(i+1, len(self.tmp_models)):
                    c = self.tmp_models[i]['category']
                    m1 = self.tmp_models[i]['model']
                    m2 = self.tmp_models[j]['model']
                    self.models += [{'category': c, 'model': m1, 'model_t': m2}]
            print("split: %s, paired-length: %d" % (split, len(self.models)))
        else:
            self.models = self.tmp_models

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        model_t = self.models[idx]['model_t']
        c_idx = self.metadata[category]['idx']

        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(self.dataset_folder, idx, c_idx, category, model, model_t)
            except Exception as e:
                if self.no_except:
                    print(e)
                    logger.warn(
                        'Error occured when loading field %s of model %s %s'
                        % (field_name, model, category)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

class PartNetDataset(data.Dataset):
    ''' KeypointNet dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None, categories=None, no_except=True, transform=None):
        ''' Initialization of the the KeypointNet dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        dataset_folder = "/home/vslab2018/3d/data/data/"
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If evaluate, load data pairs
        self.by_pair = True if split in ['val', 'test'] else False

        # If categories is None, use all subfolders
        # TODO: Can define all here
        if categories is None:
            categories = os.listdir(dataset_folder)
            self.categories = categories
        else:
            self.categories = categories

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            logger.warning('Dataset metadata file does not exist, setting name to n/a.')
            self.metadata = {
                c: {'id': c, 'name': ID2NAMES[c]} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models from split file ID2NAMES[c]
        self.tmp_models = []

        for c in categories:
            category_folder = os.path.join(dataset_folder, c+'_'+ID2NAMES[c])
            
            vox_data_path = os.path.join(category_folder, c+'_'+split + '_vox.hdf5')
            split_file = os.path.join(category_folder, c+'_'+split + '_vox.txt')
            model_idx = 0
            with open(split_file, 'r') as f:
                for line in tqdm(f.read().split('\n')):
                    if len(line) >= 1:
                        part_txt = os.path.join(category_folder, 'points', line + '.txt')
                        points, parts = self.naive_read_pcd(part_txt)
                        model_raw = {
                            'h5py_file': vox_data_path,
                            'model_idx': model_idx,
                            'points': points,
                            'parts': parts,
                        }
                        self.tmp_models += [{'category': c, 'model': line, 'model_t': None, 'data': model_raw, 'data_t': None}]
                        model_idx += 1

        if split in ['val', 'test']:
            self.tmp_models = sorted(self.tmp_models, key=lambda k: k['model']) 

        print("split: %s, length: %d" % (split, len(self.tmp_models)))

        if self.by_pair:
            self.models = []
            for i in range(len(self.tmp_models)):
                for j in range(i+1, len(self.tmp_models)):
                    c = self.tmp_models[i]['category']
                    m1 = self.tmp_models[i]['model']
                    m2 = self.tmp_models[j]['model']
                    d1 = self.tmp_models[i]['data']
                    d2 = self.tmp_models[j]['data']
                    self.models += [{'category': c, 'model': m1, 'model_t': m2, 'data': d1, 'data_t': d2}]
            print("split: %s, paired-length: %d" % (split, len(self.models)))
        else:
            self.models = self.tmp_models

    def naive_read_pcd(self, path):
        lines = open(path, 'r').readlines()
        lines = [line.rstrip().split(' ') for line in lines]
        data = np.asarray(lines)

        new_x = data[:,2][:,np.newaxis].astype(np.float)
        new_y = data[:,1][:,np.newaxis].astype(np.float)
        new_z = -1 * data[:,0][:,np.newaxis].astype(np.float)
        pc = np.concatenate([new_x, new_y, new_z], axis=1) # ([float(line[2]),float(line[1]),-float(line[0])])
        part = np.array(data[:, -1], dtype=np.float)
        return pc, part.astype(np.int)

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        model_t = self.models[idx]['model_t']
        raw_data = self.models[idx]['data']
        raw_data_t = self.models[idx]['data_t']
        c_idx = self.metadata[category]['idx']

        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(self.dataset_folder, idx, c_idx, category, model, model_t, raw_data, raw_data_t)
            except Exception as e:
                if self.no_except:
                    print(e)
                    logger.warn(
                        'Error occured when loading field %s of model %s %s'
                        % (field_name, model, category)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]



def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)