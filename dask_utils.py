import dask.array as da
import numpy as np
import dipy_multiview as dm
from image_array import ImageArray

def scale_down_dask_array(a, b=3):
    for dim in range(3):
        relevant_size = a.chunks[dim][0]
        if relevant_size % b: raise (
            Exception('scaling down only implemented for binning factors fitting into the chunk size'))

    def dask_scale_down_chunk(x, b=4):
        out_shape = (np.array(x.shape) / b).astype(np.int64)
        res = dm.transform_stack_sitk(ImageArray(x), None, out_spacing=[b, b, b], out_shape=out_shape,
                                      out_origin=[0, 0, 0])
        return np.array(res)

    res = da.map_blocks(dask_scale_down_chunk, a, dtype=np.float32,
                        chunks=tuple([a.chunks[dim][0] / b for dim in range(3)]), **{'b': b})

    return res


def scale_up_dask_array(a, b=3):
    if not np.isclose(b, int(b)):
        raise (Exception('scaling up of dask arrays only implemented for integer scalings'))
    else:
        b = int(b)

    def dask_scale_up_chunk(x, b=4):
        out_shape = (np.array(x.shape) * b).astype(np.int64)
        res = dm.transform_stack_sitk(ImageArray(x), None, out_spacing=[1. / b] * 3, out_shape=out_shape,
                                      out_origin=[0, 0, 0])
        return np.array(res)

    print(tuple([a.chunks[dim][0] * b for dim in range(3)]))
    res = da.map_blocks(dask_scale_up_chunk, a, dtype=np.float32,
                        chunks=tuple([a.chunks[dim][0] * b for dim in range(3)]), **{'b': b})

    return res