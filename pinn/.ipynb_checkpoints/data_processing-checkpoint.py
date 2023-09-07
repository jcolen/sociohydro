from torch.utils.data import IterableDataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
import numpy as np


def smooth_with_fill(arr, sigma=2):
    msk = np.isnan(arr)
    mask = np.where(~msk)

    interp = NearestNDInterpolator(np.transpose(mask), arr[mask])
    arr = interp(*np.indices(arr.shape))
    arr = gaussian_filter(arr, sigma=sigma)
    arr[msk] = np.nan
    return arr
