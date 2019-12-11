"""
file taken and modified from https://github.com/tlambert03/imarispy/blob/master/imarispy
(added functions from utils.py for easy single file import)
"""

import numpy as np
import re
import h5py
from .util import h5str, make_thumbnail, subsample_data
import logging
logger = logging.getLogger(__name__)

def np_to_ims(array, fname='myfile.ims',
              subsamp=((1, 1, 1), (1, 2, 2)),
              chunks=((16, 128, 128), (64, 64, 64)),
              compression='gzip',
              thumbsize=256,
              dx=0.1, dz=0.25):

    assert len(subsamp) == len(chunks)
    assert all([len(i) == 3 for i in subsamp]), 'Only deal with 3D chunks'
    assert all([len(i) == len(x) for i, x in zip(subsamp, chunks)])
    assert compression in (None, 'gzip', 'lzf', 'szip'), 'Unknown compression type'
    if not fname.endswith('.ims'):
        fname = fname + '.ims'

    # force 5D
    if not array.ndim == 5:
        array = array.reshape(tuple([1] * (5 - array.ndim)) + array.shape)
    nt, nc, nz, ny, nx = array.shape
    nr = len(subsamp)

    GROUPS = [
        'DataSetInfo',
        'Thumbnail',
        'DataSetTimes',
        'DataSetInfo/Imaris',
        'DataSetInfo/Image',
        'DataSetInfo/TimeInfo'
    ]

    ATTRS = [
        ('/', ('ImarisDataSet', 'ImarisDataSet')),
        ('/', ('ImarisVersion', '5.5.0')),
        ('/', ('DataSetInfoDirectoryName', 'DataSetInfo')),
        ('/', ('ThumbnailDirectoryName', 'Thumbnail')),
        ('/', ('DataSetDirectoryName', 'DataSet')),
        ('DataSetInfo/Imaris', ('Version', '8.0')),
        ('DataSetInfo/Imaris', ('ThumbnailMode', 'thumbnailMIP')),
        ('DataSetInfo/Imaris', ('ThumbnailSize', thumbsize)),
        ('DataSetInfo/Image', ('X', nx)),
        ('DataSetInfo/Image', ('Y', ny)),
        ('DataSetInfo/Image', ('Z', nz)),
        ('DataSetInfo/Image', ('NumberOfChannels', nc)),
        ('DataSetInfo/Image', ('Noc', nc)),
        ('DataSetInfo/Image', ('Unit', 'um')),
        ('DataSetInfo/Image', ('Description', 'description not specified')),
        ('DataSetInfo/Image', ('MicroscopeModality', '',)),
        ('DataSetInfo/Image', ('RecordingDate', '2018-05-24 20:36:07.000')),
        ('DataSetInfo/Image', ('Name', 'name not specified')),
        ('DataSetInfo/Image', ('ExtMin0', '0')),
        ('DataSetInfo/Image', ('ExtMin1', '0')),
        ('DataSetInfo/Image', ('ExtMin2', '0')),
        ('DataSetInfo/Image', ('ExtMax0', nx * dx)),
        ('DataSetInfo/Image', ('ExtMax1', ny * dx)),
        ('DataSetInfo/Image', ('ExtMax2', nz * dz)),
        ('DataSetInfo/Image', ('LensPower', '63x')),
        ('DataSetInfo/TimeInfo', ('DatasetTimePoints', nt)),
        ('DataSetInfo/TimeInfo', ('FileTimePoints', nt)),
    ]

    COLORS = ('0 1 0', '1 0 1', '1 1 0', '0 0 1')
    for c in range(nc):
        grp = 'DataSetInfo/Channel %s' % c
        GROUPS.append(grp)
        ATTRS.append((grp, ('ColorOpacity', 1)))
        ATTRS.append((grp, ('ColorMode', 'BaseColor')))
        ATTRS.append((grp, ('Color', COLORS[c % len(COLORS)])))
        ATTRS.append((grp, ('GammaCorrection', 1)))
        ATTRS.append((grp, ('ColorRange', '0 255')))
        ATTRS.append((grp, ('Name', 'Channel %s' % c)))
        # ATTRS.append(grp, ('LSMEmissionWavelength', 0))
        # ATTRS.append(grp, ('LSMExcitationWavelength', ''))
        # ATTRS.append(grp, ('Description', '(description not specified)'))

    # TODO: create accurate timestamps
    for t in range(nt):
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        strr = '2018-05-24 {:02d}:{:02d}:{:02d}.000'.format(h, m, s)
        ATTRS.append(('DataSetInfo/TimeInfo', ('TimePoint{}'.format(t + 1), strr)))

    with h5py.File(fname, 'a') as hf:
        for grp in GROUPS:
            hf.create_group(grp)

        for grp, (key, value) in ATTRS:
            hf[grp].attrs.create(key, h5str(value))

        try:
            thumb = make_thumbnail(array[0], thumbsize)
            hf.create_dataset('Thumbnail/Data', data=thumb, dtype='u1')
        except Exception:
            logger.warn('Failed to generate Imaris thumbnail')

        # add data
        fmt = '/DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}/'
        for t in range(nt):
            for c in range(nc):
                data = np.squeeze(array[t, c])
                for r in range(nr):
                    if any([i > 1 for i in subsamp[r]]):
                        data = subsample_data(data, subsamp[r])
                    hist, edges = np.histogram(data, 256)
                    grp = hf.create_group(fmt.format(r=r, t=t, c=c))
                    print("Writing: %s" % grp)
                    grp.create_dataset('Histogram', data=hist.astype(np.uint64))
                    grp.attrs.create('HistogramMin', h5str(edges[0]))
                    grp.attrs.create('HistogramMax', h5str(edges[-1]))
                    grp.create_dataset('Data', data=data,
                                       chunks=tuple(min(*n) for n in zip(chunks[r], data.shape)),
                                       compression=compression)
                    grp.attrs.create('ImageSizeX', h5str(data.shape[2]))
                    grp.attrs.create('ImageSizeY', h5str(data.shape[1]))
                    grp.attrs.create('ImageSizeZ', h5str(data.shape[0]))

    return fname


def unmap_bdv_from_imaris(hf):
    for i in hf:
        if re.match(r'^t\d{5}$', i) or re.match(r'^s\d{2}$', i):
            del hf[i]
    return

def make_thumbnail(array, size=256):
    """ array should be 4D array """
    # TODO: don't just crop to the upper left corner
    mip = np.squeeze(array).max(1)[:3, :size, :size].astype(np.float)
    for i in range(mip.shape[0]):
        mip[i] -= np.min(mip[i])
        mip[i] *= 255 / np.max(mip[i])
    mip = np.pad(mip, ((0, 3 - mip.shape[0]),
                       (0, size - mip.shape[1]),
                       (0, size - mip.shape[2])
                       ), 'constant', constant_values=0)
    mip = np.pad(mip, ((0, 1), (0, 0), (0, 0)), 'constant',
                 constant_values=255).astype('|u1')
    return np.squeeze(mip.T.reshape(1, size, size * 4)).astype('|u1')


def h5str(s, coding='ASCII', dtype='S1'):
    return np.frombuffer(str(s).encode(coding), dtype=dtype)


def subsample_data(data, subsamp):
    return data[0::int(subsamp[0]), 0::int(subsamp[1]), 0::int(subsamp[2])]