import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
# import im2mesh.common as common
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from collections import defaultdict
import seaborn as sns
import json

def tsboard_log_scalar(logger, scalars, it, prefrix='train'):
    for k, v in scalars.items():
        logger.add_scalar('%s/%s' % (prefrix, k), v, it)
        
def print_current_scalars(epoch, i, scalars):
    message = '(epoch: %d, iters: %d) ' % (epoch, i)
    for k, v in scalars.items():
        message += '%s: %.3f ' % (k, v)

    print(message)

def visualize_pointcloud(points, color=None, out_file=None, show=False, set_limit=True, set_view=True):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    
    viridis  = sns.color_palette("gist_rainbow", as_cmap=True) # cm.get_cmap('nipy_spectral') # gist_rainbow
    correspondance = np.linspace(0, 1, points.shape[0])

    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    if color is not None:
        # import pdb; pdb.set_trace()
        # for i in range(points.shape[0]):
        #     ax.scatter(points[i, 2], points[i, 0], points[i, 1], color=color[i]/255)
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=color/255.)
    else:   
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=correspondance, cmap=viridis)

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    if set_limit:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        # plt.axis('off')
        plt.grid(b=None)
        plt.savefig(out_file, dpi=120, transparent=False)

    plt.close(fig)

