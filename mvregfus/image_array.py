__author__ = 'malbert'

import numpy as np

# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
class ImageArray(np.ndarray):

    def __new__(cls, input_array, spacing=[1.,1.,1.], origin=[0.,0.,0.], rotation=0., meta = None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if meta is None:
            obj.spacing = np.array(spacing).astype(np.float64)
            obj.origin = np.array(origin).astype(np.float64)
            obj.rotation = float(rotation)
        else:
            obj.spacing  = meta['spacing']
            obj.origin   = meta['origin']
            obj.rotation = meta['rotation']
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.spacing = getattr(obj, 'spacing', np.array([1.,1.,1.]).astype(np.float64))
        self.origin = getattr(obj, 'origin', np.array([0.,0.,0.]).astype(np.float64))
        self.rotation = getattr(obj, 'rotation', 0.)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(ImageArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.spacing,self.origin,self.rotation)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.spacing,self.origin,self.rotation, = state[-3:]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(ImageArray, self).__setstate__(state[0:-3])

    def get_meta_dict(self):
        meta = dict()
        meta['spacing'] = self.spacing[:]
        meta['origin'] = self.origin[:]
        meta['rotation'] = float(self.rotation)
        return meta

    def get_info(self):
        meta = dict()
        meta['spacing'] = np.array(self.spacing[:])
        meta['origin'] = np.array(self.origin)
        meta['rotation'] = float(self.rotation)
        meta['size'] = np.array(self.shape)
        return meta

# # http://distributed.dask.org/en/latest/serialization.html
# try:
#     # from distributed.protocol import register_generic
#     # register_generic(ImageArray)
#
#     from distributed.protocol import dask_serialize, dask_deserialize, serialize, deserialize
#
#     @dask_serialize.register(ImageArray)
#     def serialize(imar):
#         meta = imar.get_meta_dict()
#         ar = np.array(imar)
#         header, frames = serialize([meta,ar])
#         return header, frames
#
#     @dask_deserialize.register(ImageArray)
#     def deserialize(header, frames):
#         [meta,ar] = serialize(header,frames)
#         return ImageArray(ar,meta=meta)
#
#     print('INFO: successfully registered image array for dask distributed serialization')
#
# except:
#     # http://distributed.dask.org/en/latest/serialization.html
#     print('WARNING: could not register image array for dask distributed serialization')

# from distributed.protocol import dask_serialize, dask_deserialize, serialize, deserialize
# from typing import List, Dict, Tuple
#
# @dask_serialize.register(ImageArray)
# def imar_serialize(imar: ImageArray) -> Tuple[Dict, List[bytes]]:
#     meta = imar.get_meta_dict()
#     meta = {k: v if k == 'rotation' else list(v) for k, v in meta.items()}
#     ar = np.array(imar)
#     header, frames = serialize([meta, ar])
#     return header, frames
#
# @dask_deserialize.register(ImageArray)
# def imar_deserialize(header: Dict, frames: List[bytes]) -> ImageArray:
#     [meta, ar] = deserialize(header, frames)
#     return ImageArray(ar, meta=meta)
#     # return ar
