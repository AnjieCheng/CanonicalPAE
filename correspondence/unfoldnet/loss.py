
import numpy as np
import torch
import external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from external.emd.emd_module import *

criterion = torch.nn.MSELoss()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

#------------------------------------------------------------------------------------------------------------#

def unfold_loss(esti_shapes, shapes, full_loss=False):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes) # idx1[16, 2048] idx2[16, 2562]
    if full_loss:
        loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    else:
        loss_cd = torch.mean(torch.sqrt(dist1))
    return loss_cd

def selfrec_loss(esti_shapes, shapes):
    return criterion(esti_shapes, shapes)

def CD_loss(esti_shapes, shapes):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    return loss_cd

def EMD_loss(esti_shapes, shapes):
    emd_dist = emdModule()
    dist, assigment = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()
    return loss_emd
