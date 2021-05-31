import numpy as np

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        sigma, clip = 0.015, 0.05
        noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
        noise = noise.astype(np.float32)

        data_out[None] = points + noise
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        colors = data['colors']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :] 
        data_out['colors'] = colors[indices, :] # colors

        return data_out

class SubsamplePartNetPointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        labels = data['labels']
        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :] 
        data_out['labels'] = labels[indices] # labels

        return data_out