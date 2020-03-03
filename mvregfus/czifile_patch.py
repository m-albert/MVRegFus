import warnings

import numpy as np

from mvregfus import czifile


def asarray_random_access(self, view=None, ch=None, ill=None, resize=True, order=1):
    """Return image data from file(s) as numpy array.

    modified by malbert

    # to use, monkey patch czifile.py as so:
    czifile.CziFile.asarray_random_access = asarray_random_access

    # then load stack like this:
    im = czifile.CziFile(filepath).asarray_random_access(view, ch, ill).squeeze()

    - if any of <view, ch, ill> is None, all possible stacks are extracted in this dimension

    # copied docstring of original czifile.CziFile.asarray function:

    Parameters
    ----------
    bgr2rgb : bool
        If True, exchange red and blue samples if applicable.
    resize : bool
        If True (default), resize sub/supersampled subblock data.
    order : int
        The order of spline interpolation used to resize sub/supersampled
        subblock data. Default is 1 (bilinear).

    resize : bool
        If True (default), resize sub/supersampled subblock data.
    order : int
        The order of spline interpolation used to resize sub/supersampled
        subblock data. Default is 0 (nearest neighbor).
    out : numpy.ndarray, str, or file-like object; optional
        Buffer where image data will be saved.
        If numpy.ndarray, a writable array of compatible dtype and shape.
        If str or open file, the file name or file object used to
        create a memory-map to an array stored in a binary file on disk.
    max_workers : int
        Maximum number of threads to read and decode subblock data.
        By default up to half the CPU cores are used.


    """


    image = []
    for directory_entry in self.filtered_subblock_directory:
        plane_is_wanted = True
        for dim in directory_entry.dimension_entries:

            if dim.dimension == 'V':
                if view is not None and not dim.start == view:
                    plane_is_wanted = False
                    break

            if dim.dimension == 'C':
                if ch is not None and not dim.start == ch:
                    plane_is_wanted = False
                    break

            if dim.dimension == 'I':
                if ill is not None and not dim.start == ill:
                    plane_is_wanted = False
                    break

        if not plane_is_wanted: continue

        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=resize, order=order)

        try:
            image.append(tile)
        except ValueError as e:
            warnings.warn(str(e))

    return np.array(image).squeeze()

# monkey patch czifile.py
czifile.CziFile.asarray_random_access = asarray_random_access

if __name__ == "__main__":

    # example

    filepath = "/myfile.czi"

    view = 0
    ch   = 0
    ill  = 0

    stack = czifile.CziFile(filepath).asarray_random_access(view, ch).squeeze()
    stack = stack.astype(np.uint16)  # czifile can also load in other dtypes

