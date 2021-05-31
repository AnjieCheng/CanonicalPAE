import os
import glob
import random
import h5py, json
import scipy.io
from scipy.io import savemat
from PIL import Image
import numpy as np
import trimesh
# import open3d as o3d
from sklearn import preprocessing
from correspondence.data.core import Field

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

class PartPointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=True, normalize=True, with_kp=False, with_occ=False):
        self.file_name = file_name
        self.transform = transform
        self.normalize = normalize
        self.with_transforms = with_transforms
        self.with_kp = with_kp
        self.with_occ = with_occ
        self.occ_path = "/home/vslab2018/3d/data/data/occ/"

        # if self.with_kp:
        #     self.json_file_path = os.path.join("/home/vslab2018/3d/data/KeypointNet/annotations/all.json")     
        #     self.json_annots = json.load(open(self.json_file_path)) 

    def load(self, model_path, idx, c_idx, category, model, model_t, raw_data, raw_data_t):
        ''' Loads the data point.

        Args:
            model_path (str): dataset_folder
            idx (int): ID of data point
            c_idx (int): index of category
            category (str): category id
            model (str): model id
            model_t (str): model target id
        '''
        # print(raw_data)
        data = self.load_data_by_model_name(model_path, idx, c_idx, category, model, raw_data)

        if model_t is not None:
            data_t = self.load_data_by_model_name(model_path, idx, c_idx, category, model_t, raw_data_t)
            data['target'] = data_t[None]
            data['target_labels'] = data_t['labels']

            if self.with_transforms:
                data['target_loc'] = data_t['loc']
                data['target_scale'] = data_t['scale']
        
        return data

    def load_data_by_model_name(self, model_path, idx, c_idx, category, model, raw_data):
        ''' Loads the data point.

        Args:
            model_path (str): dataset_folder
            idx (int): ID of data point
            c_idx (int): index of category
            category (str): category id
            model (str): model id
        '''
        points = raw_data['points'].astype(np.float32)
        labels = raw_data['parts'].astype(np.int32)

        p_loc = points.mean(0)
        p_scale = np.max(np.linalg.norm(points, axis=-1))

        if self.normalize:
            points = points - p_loc
            points /= p_scale

        data = {
            None: points,
            'labels': labels,
        }

        if self.with_transforms:
            data['loc'] = p_loc.astype(np.float32)
            data['scale'] = p_scale.astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        if self.with_occ:

            occ_path = os.path.join(self.occ_path, category, model+'.mat')
            if os.path.exists(occ_path) and False:
                occ_raw = scipy.io.loadmat(occ_path)
                data['occ_pts_32'] = occ_raw['occ_pts_32'].astype(np.float32) # coords
                data['occ_32'] = occ_raw['occ_32'].astype(np.int32) # occ

                indices = np.random.randint(data['occ_pts_32'].shape[0], size=4096)
                data['occ_pts_32'] = data['occ_pts_32'][indices, :] 
                data['occ_32'] = data['occ_32'][indices, :]

            else:

                model_idx = raw_data['model_idx']
                vox_data_path = raw_data['h5py_file']
                vox_data = h5py.File(vox_data_path, 'r')

                # 'points_16': vox_data['points_16'][()][model_idx],  # (4096, 3)
                # 'points_32': vox_data['points_32'][()][model_idx],  # (8102, 3)
                # 'points_64': vox_data['points_64'][()][model_idx],  # (32768, 3)
                # 'values_16': vox_data['values_16'][()][model_idx],  # (4096, 1)
                # 'values_32': vox_data['values_32'][()][model_idx],  # (8102, 1)
                # 'values_64': vox_data['values_64'][()][model_idx],  # (32768, 1)

                # points_16 = raw_data['points_16']
                # points_32 = raw_data['points_32']
                points_32 = vox_data['points_32'][()][model_idx]
                # values_16 = raw_data['values_16']
                # values_32 = raw_data['values_32']
                values_32 = vox_data['values_32'][()][model_idx]

                points_32 = (points_32+0.5)/32-0.5

                if self.normalize:
                    # points_16 = points_16 - p_loc
                    # points_16 /= p_scale
                    # points_32 = points_32 - p_loc
                    # points_32 /= p_scale
                    points_32 = points_32 - p_loc
                    points_32 /= p_scale

                # data['occ_pts_16'] = points_16
                # data['occ_16'] = values_16
                # data['occ_pts_32'] = points_32
                # data['occ_32'] = values_32
                data['occ_pts_32'] = points_32
                data['occ_32'] = values_32

                if not os.path.exists(os.path.join(self.occ_path, category)):
                    os.mkdir(os.path.join(self.occ_path, category))
                saved_data = {
                    'occ_pts_32': data['occ_pts_32'],
                    'occ_32': data['occ_32'],
                }
                savemat(occ_path, saved_data)

        return data


class KpnPointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=True, normalize=True, with_kp=False, with_occ=False, with_rotation=False, angle_sigma=0.2, angle_clip=0.5):
        self.file_name = file_name
        self.transform = transform
        self.normalize = normalize
        self.with_transforms = True # with_transforms
        self.with_kp = with_kp
        self.with_occ = with_occ
        self.with_rotation = with_rotation
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip
        self.mesh_path = "/home/vslab2018/3d/data/KeypointNet/ShapeNetCore.v2.ply"
        self.occ_path = "/home/vslab2018/3d/data/KeypointNet/occ"

        if self.with_kp:
            self.json_file_path = os.path.join("/home/vslab2018/3d/data/KeypointNet/annotations/all.json")     
            self.json_annots = json.load(open(self.json_file_path)) 

        self.cat_to_kp_num = {
            '02691156':26, #airplane
            '02808440':15, #bathtub
            '02818832':8, #bed
            '02876657':9, #bottle
            '02954340':6, #cap
            '02958343':17, #car
            '03001627':15, #chair
            '03467517':13, #guitar
            '03513137':8, #helmet
            '03624134':5, #knife
            '03642806':12, #laptop
            '03790512':7, #motorcycle
            '03797390':9, #mug
            '04225987':9, #skateboard
            '04379243':17, #table
            '04530566':18, #vessel
        }

    def load(self, model_path, idx, c_idx, category, model, model_t):
        ''' Loads the data point.

        Args:
            model_path (str): dataset_folder
            idx (int): ID of data point
            c_idx (int): index of category
            category (str): category id
            model (str): model id
            model_t (str): model target id
        '''
        # model = '47ae91f47ffb34c6f7628281ecb18112'
        data = self.load_data_by_model_name(model_path, idx, c_idx, category, model)

        if model_t is not None:
            data_t = self.load_data_by_model_name(model_path, idx, c_idx, category, model_t)
            data['target'] = data_t[None]
            data['target_colors'] = data_t['colors']
            data['target_mesh_path'] = data_t['mesh_path']

            if self.with_kp:
                data['target_kp'] = data_t['kp']
            if self.with_transforms:
                data['target_loc'] = data_t['loc']
                data['target_scale'] = data_t['scale']

        return data

    def load_data_by_model_name(self, model_path, idx, c_idx, category, model):
        ''' Loads the data point.

        Args:
            model_path (str): dataset_folder
            idx (int): ID of data point
            c_idx (int): index of category
            category (str): category id
            model (str): model id
        '''
        # self.dataset_folder/pcds/category/model.pcd
        file_path = os.path.join(model_path, self.file_name, category, model+'.pcd')

        # read pcl
        points, colors = self.naive_read_pcd(file_path)
        points = points.astype(np.float32)
        colors = colors.astype(np.float32)

        p_loc = points.mean(0)
        p_scale = np.max(np.linalg.norm(points, axis=-1))

        if self.normalize:
            points = points - p_loc
            points /= p_scale

        if self.with_rotation:
            points = self.rotate_point_cloud(points, angle_sigma=self.angle_sigma, angle_clip=self.angle_clip)

        data = {
            None: points.astype(np.float32),
            'colors': colors,
        }

        if self.with_kp:
            keypoint_coords = np.full((25, 3), -999).astype(np.float32)
            label = [label for label in self.json_annots if label['model_id'] == model][0]

            for kp in label['keypoints']:
                keypoint_coords[kp['semantic_id']] = points[kp['pcd_info']['point_index']]

            data['kp'] = keypoint_coords.astype(np.float32)

        if self.with_transforms:
            data['loc'] = p_loc.astype(np.float32)
            data['scale'] = p_scale.astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        data['mesh_path'] = os.path.join(self.mesh_path, category, model+'.ply')
            
        return data

    def naive_read_pcd(self, path):
        lines = open(path, 'r').readlines()
        idx = -1
        for i, line in enumerate(lines):
            if line.startswith('DATA ascii'):
                idx = i + 1
                break
        lines = lines[idx:]
        lines = [line.rstrip().split(' ') for line in lines]
        data = np.asarray(lines)
        pc = np.array(data[:, :3], dtype=np.float)
        colors = np.array(data[:, -1], dtype=np.int)
        colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
        return pc, colors.astype(np.float)

    def rotate_point_cloud(self, data, angle_sigma=0.2, angle_clip=0.5):
        """ Randomly perturb the point clouds by small rotations
            Input:
            Nx3 array, original point clouds
            Return:
            Nx3 array, rotated point clouds
        """
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))

        rotated_data = np.dot(data, R)

        return rotated_data
