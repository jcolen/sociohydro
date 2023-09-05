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


class SociohydrodynamicsDataset(IterableDataset):
    def __init__(self, 
                 filename='data/cook_IL.hdf5', 
                 window_size=(32,24),
                 batch_size=16,
                 test_bbox=[20, 43, 60, 23]):
        self.filename = filename
        self.window_size = window_size
        self.batch_size = batch_size

        with h5py.File(filename, 'r') as d:
            x_grid = d["x_grid"][:]
            y_grid = d["y_grid"][:]
            w_grid = d["w_grid_array_masked"][:]
            b_grid = d["b_grid_array_masked"][:]
            for i in range(5):
                w_grid[..., i] = smooth_with_fill(w_grid[..., i])
                b_grid[..., i] = smooth_with_fill(b_grid[..., i])

        self.x = x_grid[0]
        self.y = y_grid[:, 0]
        self.t = np.array([1980, 1990, 2000, 2010, 2020])

        self.w_func = interp1d(self.t, w_grid, axis=2)
        self.b_func = interp1d(self.t, b_grid, axis=2)

        y0, x0, h, w = test_bbox
        w_test = w_grid[y0:y0+h, x0:x0+w].transpose(2, 0, 1)
        b_test = b_grid[y0:y0+h, x0:x0+w].transpose(2, 0, 1)
        self.test_batch = {
            't': torch.from_numpy(self.t).float(),
            'w': torch.from_numpy(w_test).float(),
            'b': torch.from_numpy(b_test).float(),
        }


    def __next__(self):
        '''
        Yield a sample interpolated from the data
        '''
        t0 = np.random.random()
        t1 =  t0 + (1 - t0) * np.random.random()

        t = np.array([t0, t1])
        t = t * np.ptp(self.t) + self.t[0]

        w = self.w_func(t)
        b = self.b_func(t)

        w_windows = sliding_window_view(w, self.window_size, axis=(0, 1))
        b_windows = sliding_window_view(b, self.window_size, axis=(0, 1))

        valid = np.logical_and(~np.isnan(w_windows), ~np.isnan(b_windows))
        valid = np.all(valid, axis=(-1, -2, -3))
        w_windows = w_windows[valid]
        b_windows = b_windows[valid]

        idx = np.random.choice(w_windows.shape[0], self.batch_size, replace=False)
        w = w_windows[idx]
        b = b_windows[idx]

        return {
            't': torch.from_numpy(t).float(),
            'w': torch.from_numpy(w).float(),
            'b': torch.from_numpy(b).float(),
        }

    def __iter__(self):
        return self

'''
Implementation of numpy's sliding window view
'''
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
from numpy.lib.stride_tricks import as_strided

def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.

    .. versionadded:: 1.20.0

    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.

    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.

    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:

    - `scipy.signal.fftconvolve`

    - filtering functions in `scipy.ndimage`

    - moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.

    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.

    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    This also works in more dimensions, e.g.

    >>> i, j = np.ogrid[:3, :4]
    >>> x = 10*i + j
    >>> x.shape
    (3, 4)
    >>> x
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> shape = (2,2)
    >>> v = sliding_window_view(x, shape)
    >>> v.shape
    (2, 3, 2, 2)
    >>> v
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])

    The axis can be specified explicitly:

    >>> v = sliding_window_view(x, 3, 0)
    >>> v.shape
    (1, 4, 3)
    >>> v
    array([[[ 0, 10, 20],
            [ 1, 11, 21],
            [ 2, 12, 22],
            [ 3, 13, 23]]])

    The same axis can be used several times. In that case, every use reduces
    the corresponding original dimension:

    >>> v = sliding_window_view(x, (2, 3), (1, 1))
    >>> v.shape
    (3, 1, 2, 3)
    >>> v
    array([[[[ 0,  1,  2],
             [ 1,  2,  3]]],
           [[[10, 11, 12],
             [11, 12, 13]]],
           [[[20, 21, 22],
             [21, 22, 23]]]])

    Combining with stepped slicing (`::step`), this can be used to take sliding
    views which skip elements:

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 5)[:, ::2]
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6]])

    or views which move by multiple elements

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 3)[::2, :]
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6]])

    A common application of `sliding_window_view` is the calculation of running
    statistics. The simplest example is the
    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:

    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])

    Note that a sliding window approach is often **not** optimal (see Notes).
    """
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)
