import torch
from torch.distributions import Uniform, Normal, HalfCauchy
from sklearn.preprocessing import minmax_scale
from scipy import sparse, spatial
import trimesh
import scipy.io
import os, json
import seaborn as sns
import numpy as np

ID2NAMES = {"02691156": "airplane",
            "02773838": "bag",
            "02828884": "bench",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03261776": "earphone",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03636649": "lamp",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "03948459": "pistol",
            "04090263": "rifle",
            "04099429": "rocket",
            "04225987": "skateboard",
            "04256520": "couch",
            "04379243": "table",
            "04530566": "vessel",}

NAMES2ID = {v: k for k, v in ID2NAMES.items()}

def batch_sample_from_2d_grid(grid, K, batch_size, without_sample=False):
    grid_size = grid.shape[0]

    grid = grid.unsqueeze(0)
    grid = grid.expand(batch_size, -1, -1) # BxNx2

    if without_sample:
        return grid
        
    assert(grid_size >= K)
    idx = torch.randint(
        low=0, high=grid_size,
        size=(batch_size, K),
    )

    idx, _ = torch.sort(idx, 1)

    idx = idx[:, :, None].expand(batch_size, K, 2)

    sampled_points = torch.gather(grid, dim=1, index=idx)
    assert(sampled_points.size() == (batch_size, K, 2))
    return sampled_points