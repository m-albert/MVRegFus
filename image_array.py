__author__ = 'malbert'

import numpy as np

# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
class ImageArray(np.ndarray):

    def __new__(cls, input_array, spacing=[1.,1.,1.], origin=[0.,0.,0.], rotation=0.):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.spacing = np.array(spacing).astype(np.float64)
        obj.origin = np.array(origin).astype(np.float64)
        obj.rotation = float(rotation)
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