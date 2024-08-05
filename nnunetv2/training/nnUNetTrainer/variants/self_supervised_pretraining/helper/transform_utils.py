import numpy as np
import torch
from scipy.special import comb


def crop_data(data, crop_size, center_crop=True):
    _, data_h, data_w, data_d = data.shape

    if center_crop:
        x_start = (data_h - crop_size[0]) // 2
        y_start = (data_w - crop_size[1]) // 2
        z_start = (data_d - crop_size[2]) // 2
    else:
        x_start = np.random.randint(0, data_h - crop_size[0])
        y_start = np.random.randint(0, data_w - crop_size[1])
        z_start = np.random.randint(0, data_d - crop_size[2])

    x_end = x_start + crop_size[0]
    y_end = y_start + crop_size[1]
    z_end = z_start + crop_size[2]

    return data[:, x_start:x_end, y_start:y_end, z_start:z_end]


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def view_data(*data):
    data = np.stack(data)

    from batchviewer import view_batch
    # we can only view 4D data => reduce

    # if data.ndim == 5 (batch, channels, w, h, d) => (batch, w, h, d)
    if data.ndim == 5:
        # randomly_select a channel
        num_channels = data.shape[1]
        channel_idx = np.random.randint(0, num_channels)
        data = data[:, channel_idx]

    # if data.ndim == 6 (batch, augmentation_views, channels, w, h, d, t) => (augmentation_views, w, h, d)
    if data.ndim == 6:
        data = data[0, :, 0]

    view_batch(data, width=300, height=300)
