__author__ = 'malbert'

import os,tempfile,pdb,sys
import numpy as np
import czifile
import SimpleITK as sitk
from scipy import ndimage
from image_array import ImageArray
import mv_utils
# import pyximport
# pyximport.install(setup_args={'include_dirs':'.'})
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   AffineTransform2D,TranslationTransform3D,
                                   # ShearTransform3D,
                                   RigidTransform3D,AffineTransform3D)

import io_utils
io_decorator = io_utils.io_decorator_local

import warnings
def get_number_of_zplanes(self, view=None, ch=None, ill=None, resize=True, order=1):
    """Return image data from file(s) as numpy array.

    modified by malbert
    example:
    - to extract view 0, channel 0 from multiview, channel file
    get axes from self.filtered_subblock_directory[-1].axes

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

import warnings
def asarray_random_access(self, view=None, ch=None, ill=None, resize=True, order=1):
    """Return image data from file(s) as numpy array.

    modified by malbert
    example:
    - to extract view 0, channel 0 from multiview, channel file
    get axes from self.filtered_subblock_directory[-1].axes

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

# import warnings
# def asarray_view_ch(self, view, ch, resize=True, order=1):
#     """Return image data from file(s) as numpy array.
#
#     modified by malbert
#     example:
#     - to extract view 0, channel 0 from multiview, channel file
#     get axes from self.filtered_subblock_directory[-1].axes
#
#     Parameters
#     ----------
#     bgr2rgb : bool
#         If True, exchange red and blue samples if applicable.
#     resize : bool
#         If True (default), resize sub/supersampled subblock data.
#     order : int
#         The order of spline interpolation used to resize sub/supersampled
#         subblock data. Default is 1 (bilinear).
#
#     resize : bool
#         If True (default), resize sub/supersampled subblock data.
#     order : int
#         The order of spline interpolation used to resize sub/supersampled
#         subblock data. Default is 0 (nearest neighbor).
#     out : numpy.ndarray, str, or file-like object; optional
#         Buffer where image data will be saved.
#         If numpy.ndarray, a writable array of compatible dtype and shape.
#         If str or open file, the file name or file object used to
#         create a memory-map to an array stored in a binary file on disk.
#     max_workers : int
#         Maximum number of threads to read and decode subblock data.
#         By default up to half the CPU cores are used.
#
#
#     """
#
#     nonZeroDims = []
#     for idim in range(len(self.shape)):
#         if self.shape[idim]>1: nonZeroDims.append(idim)
#
#     image = []
#
#     ndims = len(self.start)
#     for directory_entry in self.filtered_subblock_directory:
#
#         index_start = [directory_entry.start[i] - self.start[i] for i in range(ndims)]
#         if index_start[0] != view or index_start[5] != ch: continue
#
#         # print(index_start[0],directory_entry.start)
#         subblock = directory_entry.data_segment()
#         tile = subblock.data(resize=resize, order=order)
#         # index = [slice(i-j, i-j+k) for i, j, k in
#         #          zip(directory_entry.start, self.start, tile.shape)]
#
#         try:
#             image.append(tile)
#         except ValueError as e:
#             warnings.warn(str(e))
#
#     return np.array(image)
#
# # monkey patch czifile.py
# czifile.CziFile.asarray_view_ch = asarray_view_ch

@io_decorator
def readStackFromMultiviewMultiChannelCzi(filepath,view=0,ch=0,
                                          background_level=200,
                                          infoDict=None,
                                          do_clean_pixels=True,
                                          do_smooth=True,
                                          extract_rotation=True,
                                          do_despeckle=False,
                                          raw_input_binning=None,
                                          ill=None,
                                          ):
    print('reading %s view %s ch %s ill %s' %(filepath,view,ch,ill))
    # return ImageArray(np.ones((10,10,10)))
    if infoDict is None:
        infoDict = getStackInfoFromCZI(filepath)

    # stack = czifile.CziFile(filepath).asarray_view_ch(view,ch).squeeze()

    # # fuse illuminations
    # illuminations = infoDict['originalShape'][1]
    # if illuminations > 1:
    #     if ill is None:
    #         print('fusing %s illuminations using simple mean' %illuminations)
    #         # stack = np.mean([stack[i:stack.shape[0]:illuminations] for i in range(illuminations)],0).astype(np.uint16)
    #         zshape = int(stack.shape[0]/illuminations)
    #         for z in range(zshape):
    #             if not z%50: print('fusing z plane: %s' %z)
    #             stack[z] = np.mean(stack[z*illuminations:z*illuminations+illuminations],0).astype(np.uint16)
    #         stack = stack[:zshape]
    #         # print('choosing only illumination 1')
    #         # stack = np.array(stack[1:stack.shape[0]:illuminations]).astype(np.uint16)
    #     else:
    #         print('picking illumination %s' %ill)
    #         stack = np.array(stack[ill:stack.shape[0]:illuminations]).astype(np.uint16)

    # fuse illuminations
    illuminations = infoDict['originalShape'][1]
    if illuminations > 1 and ill is None:

        stack = czifile.CziFile(filepath).asarray_random_access(view, ch).squeeze()
        stack = stack.astype(np.uint16)  # czifile can also load in other dtypes

        print('fusing %s illuminations using simple mean' %illuminations)
        # stack = np.mean([stack[i:stack.shape[0]:illuminations] for i in range(illuminations)],0).astype(np.uint16)
        zshape = int(stack.shape[0]/illuminations)
        for z in range(zshape):
            if not z%50: print('fusing z plane: %s' %z)
            stack[z] = np.mean(stack[z*illuminations:z*illuminations+illuminations],0).astype(np.uint16)
        stack = stack[:zshape]
        # print('choosing only illumination 1')
        # stack = np.array(stack[1:stack.shape[0]:illuminations]).astype(np.uint16)
    else:
        stack = czifile.CziFile(filepath).asarray_random_access(view=view, ch=ch, ill=ill).squeeze()
        stack = stack.astype(np.uint16)  # czifile can also load in other dtypes

    if raw_input_binning is not None:
        print('WARNING: binning down raw input by xyz factors %s' %raw_input_binning)
        print('old shape: %s %s %s' %stack.shape)
        stack = np.array(bin_stack(ImageArray(stack),raw_input_binning))
        print('new shape: %s %s %s' %stack.shape)

    if do_despeckle: # try to supress vesicles
        print('warning: despeckling images')
        stack = despeckle(stack)
    if do_clean_pixels:
        stack = clean_pixels(stack)
        print('warning: clean pixels')
    if do_smooth:
        stack = ndimage.gaussian_filter(stack,sigma=(0,2,2.)).astype(np.uint16)
        print('warning: smoothing pixels (kxy=2!)')

    # for big run, used cleaning and gaussian. deactivated 20180404 for lucien

    # print('warning: no clean at input!')
    stack = (stack - background_level) * (stack > background_level)
    if extract_rotation:
        rotation = infoDict['positions'][view][3]
    else:
        rotation = 0

    if raw_input_binning is None:
        spacing = infoDict['spacing'][::-1]
    else:
        spacing = (infoDict['spacing'] * np.array(raw_input_binning))[::-1]
    stack = ImageArray(stack,spacing=spacing,origin=infoDict['origins'][view][::-1],rotation=rotation)

    return stack


def despeckle(im):
    # try to supress signal coming from bright, small vesicles

    ims = sitk.GetImageFromArray(im)
    ims = sitk.Cast(ims,sitk.sitkFloat32)
    radius = 2
    rf = sitk.Rank(ims,(5**3-7)/5**3,radius = [radius]*3) # -r here is equivalent to -(r+1) in np
#     return sitk.GetArrayFromImage(rf)
#     rf = sitk.Rank(ims,(8)/radius**3,radius = [radius]*3)
    res = sitk.Minimum(ims,rf)

    diff = ims - res
    k = np.zeros((3,3,3))
    a = 1
    k[1,1] = a
    k[:,1,1] = a
    k[1,:,1] = a
    k[1,1,1] = 5
    k = sitk.GetImageFromArray(k)
    k = sitk.Cast(k,sitk.sitkFloat32)

    diff = sitk.Convolution(diff,k)
    diff = sitk.Cast(diff,sitk.sitkFloat32)
    sub = (ims - diff)*sitk.Cast(ims>diff,sitk.sitkFloat32)
    sub_gauss = sitk.RecursiveGaussian(sub,2)
    res = sitk.Maximum(sub,sub_gauss) # get rid of very low values
    res = sitk.Minimum(ims,res) # get rid of glow introduced by gauss
    res = sitk.GetArrayFromImage(res)
    res = res.astype(np.uint16)
    return res

# ok, working
from scipy.fftpack import dctn,idctn
# from scipy.fftpack import dct,idct
def clean(im,cut=50):

    # axes = [im.shape[i] for i in [-2,-1]]
    axes = [-2,-1]

    # compute forward
    d = dctn(im,norm='ortho',axes=axes)
    # d = dct(dct(im,axis=-1,norm='ortho'),axis=-2,norm='ortho')

    # cut frequencies
    # typical stripe pattern on Z1 ranges from 1-10px
    slices = [slice(0,im.shape[dim]) for dim in range(im.ndim)]
    slices[-2] = slice(0,1)
    slices[-1] = slice(cut,im.shape[-1])

    d[tuple(slices)] = 0

    # cut frequencies
    # typical stripe pattern on Z1 ranges from 1-10px
    slices = [slice(0,im.shape[dim]) for dim in range(im.ndim)]
    slices[-1] = slice(0,1)
    slices[-2] = slice(cut,im.shape[-2])

    d[tuple(slices)] = 0

    # compute backward
    im = idctn(d,norm='ortho',axes=axes)
    # im = idct(idct(d,norm='ortho',axis=-2),norm='ortho',axis=-1)
    return im

def clean_pixels(im):
    """
    remove stripe artefacts in Z1 images
    where do they come from? camera problem or cable?
    :param im:
    :return:
    """
    print('cleaning pixels')

    weights = np.ones((1,11,11)).astype(np.float)
    weights[0,5,5] = 0
    weights /= np.sum(weights)

    sim = sitk.GetImageFromArray(im)
    weights = sitk.GetImageFromArray(weights)

    sim = sitk.Cast(sim,sitk.sitkFloat32)
    weights = sitk.Cast(weights,sitk.sitkFloat32)

    estimate = sitk.Convolution(sim,weights)
    estimate = sitk.Cast(estimate,sitk.sitkFloat32)

    pixel_offset = sitk.MedianProjection(sim - estimate,2)
    del sim
    pixel_offset = sitk.GetArrayFromImage(pixel_offset)

    im = im - pixel_offset
    im = (im*(im>0)).astype(np.uint16)

    return im

@io_decorator
def bin_stack(im,bin_factors=np.array([1,1,1])):
    if np.allclose(bin_factors, [1, 1, 1]): return im
    bin_factors = np.array(bin_factors)
    origin = im.origin
    spacing = im.spacing
    rotation = im.rotation
    binned_spacing = spacing * bin_factors[::-1]
    # binned_origin = origin + (spacing*bin_factors[::-1])/2
    # binned_origin = origin
    binned_origin = origin + (binned_spacing-spacing)/2.
    # print('watch out with binning origin!')

    im = sitk.GetImageFromArray(im)
    im = sitk.BinShrink(im,[int(i) for i in bin_factors])
    im = sitk.GetArrayFromImage(im)
    # im = (im - background_level) * (view > background_level)
    im = ImageArray(im,spacing=binned_spacing,origin=binned_origin,rotation=rotation)
    return im

def getStackFromMultiview(result,ichannel,iview):
    return result[ichannel][iview]

def getStackInfoFromCZI(pathToImage, xy_spacing=None):
    """

    :rtype : object
    """

    print('getting stack info')

    infoDict = dict()

    imageFile = czifile.CziFile(pathToImage)
    originalShape = imageFile.shape
    print(imageFile.shape)
    metadata = imageFile.metadata
    imageFile.close()

    if type(metadata) == str:
        try:
            from lxml import etree
        except ImportError:
            from xml.etree import cElementTree as etree
        metadata = etree.fromstring(metadata)

    # old version
    # nViews = metadata.xpath("//SizeV")
    # multiView = True
    # if len(nViews):
    #     nViews = int(metadata.xpath("//SizeV")[0].text)
    # else:
    #     nViews = 1
    #     multiView = False

    # hopefully more general
    nViews = metadata.findall(".//MultiView")
    multiView = True
    if len(nViews):
        nViews = len(metadata.findall(".//MultiView/View"))
    else:
        nViews = 1
        multiView = False

    nX = int(metadata.findall(".//SizeX")[0].text)
    nY = int(metadata.findall(".//SizeY")[0].text)

    spacing = np.array([float(i.text) for i in metadata.findall(".//Scaling")[0].findall(".//Value")]) * np.power(10, 6)
    spacing = spacing.astype(np.float64)

    if xy_spacing is not None:
        spacing[:2] = xy_spacing

    if multiView:

        def count_planes_of_view_in_czifile(self, view):

            """
            get number of zplanes of a given view independently of number of channels and illuminations
            """

            curr_ch = 0
            curr_ill = 0
            i = 0
            for directory_entry in self.filtered_subblock_directory:
                plane_is_wanted = True
                ch_or_ill_changed = False
                for dim in directory_entry.dimension_entries:

                    if dim.dimension == 'V':
                        if view is not None and not dim.start == view:
                            plane_is_wanted = False
                            break

                    if dim.dimension == 'C':
                        if curr_ch != dim.start:
                            ch_or_ill_changed = True
                            break

                    if dim.dimension == 'I':
                        if curr_ill != dim.start:
                            ch_or_ill_changed = True
                            break

                if plane_is_wanted and not ch_or_ill_changed: i += 1

            return i

        axisOfRotation = np.array([float(i) for i in metadata.findall(".//AxisOfRotation")[0].text.split(' ')])
        axisOfRotation = np.where(axisOfRotation)[0][0]
        centerOfRotation = np.array([-float(i) for i in metadata.findall(".//CenterPosition")[0].text.split(' ')])

        rPositions, xPositions, yPositions, zPositions = [], [], [], []
        nZs = []
        for i in range(nViews):
            baseNode = metadata.findall(".//View[@V='%s']" % i)
            if len(baseNode) == 2:
                baseNode = baseNode[1]
            else:
                baseNode = baseNode[0]
            xPositions.append(float(baseNode.findall(".//PositionX")[0].text))
            yPositions.append(float(baseNode.findall(".//PositionY")[0].text))
            zPositions.append(float(baseNode.findall(".//PositionZ")[0].text))
            rPositions.append(float(baseNode.findall(".//Offset")[0].text) / 180. * np.pi)
            nZs.append(count_planes_of_view_in_czifile(imageFile,i))

        sizes = np.array([[nX, nY, nZs[i]] for i in range(nViews)])
        positions = np.array([xPositions, yPositions, zPositions, rPositions]).swapaxes(0, 1)
        origins = np.array([positions[i][:3] - np.array([sizes[i][0] / 2, sizes[i][1] / 2, 0]) * spacing for i in
                            range(len(positions))])

        # infoDict['angles'] = np.array(rPositions)
        infoDict['origins'] = origins
        infoDict['positions'] = positions
        infoDict['centerOfRotation'] = centerOfRotation
        infoDict['axisOfRotation'] = axisOfRotation
        infoDict['sizes'] = sizes
    else:
        nZ = int(metadata.findall(".//SizeZ")[0].text)
        size = np.array([nX, nY, nZ])

        # position = metadata.findall('.//Positions')[3].findall('Position')[0].values()
        position = metadata.findall('.//Positions')[3].findall('Position')[0].attrib.values()
        position = np.array([float(i) for i in position])
        origin = np.array(position[:3] - np.array([size[0] / 2, size[1] / 2, 0]) * spacing)

        infoDict['sizes'] = np.array([size])
        infoDict['positions'] = np.array([position])
        infoDict['origins'] = np.array([origin])

    infoDict['spacing'] = spacing
    infoDict['originalShape'] = np.array(originalShape)

    try:
        infoDict['dT'] = float(int(metadata.findall('//TimeSpan/Value')[0].text) / 1000)
    except:
        pass

    return infoDict

def clahe(image,kernel_size,clip_limit=0.02,pad=0,ds=4):
    import claheNd
    print('compute clahe with kernel size %s' %kernel_size)
    # pad = int(pad_ratio * np.min(image.shape))
    # skimage discussion says that the artefact occurs if image shape is not multiple of kernel size
    # image = np.pad(image,pad,mode='edge')
    # pad = int(np.min(image.shape)*float(pad_factor))
    if pad > 0:
        image = np.pad(image,pad,mode='constant',constant_values=0)
        image[image==0] = np.random.randint(0,15,np.sum(image==0))

    if kernel_size == 0:
        # return normalise(image,ds)
        return image

    original_shape = np.array(image.shape)
    modulus_shape = original_shape % kernel_size
    better_shape = original_shape - modulus_shape + (modulus_shape > 0) * kernel_size

    newim = np.zeros(tuple(better_shape),dtype=image.dtype)
    newim[tuple([slice(0,original_shape[i]) for i in range(image.ndim)])] = image

    result = claheNd.equalize_adapthist(newim,kernel_size=kernel_size,clip_limit=clip_limit,nbins=2**13)
    # result = result[pad:-pad,pad:-pad,pad:-pad]
    result = result[tuple([slice(0,original_shape[i]) for i in range(image.ndim)])]

    return result

def get_chromatic_correction_parameters(im0,im1):
    # first stack is reference

    # projs = [[]*3 for stack in stacks]
    # for istack,stack in enumerate(stacks):
    #     for dim in range(3):
    #         projs[istack][dim] = np.max(stack,dim)

    # get 2d affine parameters of max proj 0
    # params = [[1,0,0,0,1,0,0,0,1,0,0,0]]
    static = np.max(im0,0)

    static = clahe(static,[10,10],clip_limit=0.02)

    # static_grid2world = np.diag([1] + list(im0.spacing[1:]))
    static_grid2world = np.diag(list(im0.spacing[1:])+[1])
    # ress = [static]

    # for istack in range(1,len(stacks)):
    moving = np.max(im1,0)
    moving = clahe(moving,[10,10],clip_limit=0.02)

    # moving_grid2world = np.diag([1] + list(im1.spacing[1:]))
    moving_grid2world = np.diag(list(im1.spacing[1:])+[1])

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = list(np.array([1000000000]*7))
    sigmas = [6,5,4,3.0,1.0,1.0, 0.0]
    factors = [6,5,4,3,2,1,1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=3,
                                options = {'gtol': 1e-8}
                                )

    # transform = TranslationTransform3D()
    params0 = None
    # starting_affine = np.eye(4)
    # starting_affine[:3,:3] = t0[:9].reshape((3,3))
    # starting_affine[:3,3] = t0[9:]
    starting_affine = None

    # transform = AffineTransform2D()
    transform = TranslationTransform2D()
    translation = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transform = AffineTransform2D()
    affine = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)

    M = np.eye(3)
    M[1:3,1:3] = affine.affine[:2,:2]
    param = np.zeros(12)
    param[:9] = M.flatten()
    param[10:] = affine.affine[:2,2]
    # transform to center of image coordinate system
    # c' = c + Ax0
    # the transform applied before resampling a new image would then be
    # c = c' - Axn
    # center = np.array(im0.shape)/2. * im0.spacing
    # # param[10:] = affine.affine[:2,2] + np.dot(affine.affine[:2,:2],center)
    # param[9:] = param[9:] - np.dot(param[9:],center)

    # transform params to middle of stack!
    # raise(Exception('todo'))

    # params.append(param)
    # params.append(affine.affine)
    # ress.append(affine.transform(moving))

    # return params,np.array(ress)
    return param

@io_decorator
def get_chromatic_correction_parameters_center(im0,im1):
    # first stack is reference

    # projs = [[]*3 for stack in stacks]
    # for istack,stack in enumerate(stacks):
    #     for dim in range(3):
    #         projs[istack][dim] = np.max(stack,dim)

    # get 2d affine parameters of max proj 0
    # params = [[1,0,0,0,1,0,0,0,1,0,0,0]]
    static = np.max(im0,0)

    static = clahe(static,[10,10],clip_limit=0.02)

    # static_grid2world = np.diag([1] + list(im0.spacing[1:]))
    static_grid2world = np.diag(list(im0.spacing[1:])+[1])
    static_grid2world[:2,2] = im0.spacing[1:]*np.array(im0.shape[1:])/2.
    # ress = [static]

    # for istack in range(1,len(stacks)):
    moving = np.max(im1,0)
    moving = clahe(moving,[10,10],clip_limit=0.02)

    # moving_grid2world = np.diag([1] + list(im1.spacing[1:]))
    moving_grid2world = np.diag(list(im1.spacing[1:])+[1])
    moving_grid2world[:2,2] = im1.spacing[1:]*np.array(im1.shape[1:])/2.

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = list(np.array([1000000000]*7))
    sigmas = [6,5,4,3.0,1.0,1.0, 0.0]
    factors = [6,5,4,3,2,1,1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=3,
                                options = {'gtol': 1e-8}
                                )

    # transform = TranslationTransform3D()
    params0 = None
    # starting_affine = np.eye(4)
    # starting_affine[:3,:3] = t0[:9].reshape((3,3))
    # starting_affine[:3,3] = t0[9:]
    starting_affine = None

    # transform = AffineTransform2D()
    transform = TranslationTransform2D()
    translation = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transform = AffineTransform2D()
    affine = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)

    M = np.eye(3)
    M[1:3,1:3] = affine.affine[:2,:2]
    param = np.zeros(12)
    param[:9] = M.flatten()
    param[10:] = affine.affine[:2,2]

    # center = np.array([0]+list(im1.spacing[1:]*np.array(im1.shape[1:])/2.))
    # param[9:] = param[9:] + np.dot(param[9:],center)

    # transform params to middle of stack!
    # raise(Exception('todo'))

    # params.append(param)
    # params.append(affine.affine)
    # ress.append(affine.transform(moving))

    # return params,np.array(ress)
    return param

def createInitialTransformFile(spacing,params,template,outPath):
    spacingString = '\n\n(Spacing %s %s %s)\n' %tuple(spacing)
    paramsString = '\n\n(TransformParameters %s %s %s %s %s %s %s %s %s %s %s %s)\n\n' %tuple(params)
    template = paramsString + spacingString + template
    outFile = open(outPath,'w')
    outFile.write(template)
    outFile.close()
    return

def createParameterFile(spacing,initialTransformFile,template,outPath):
    spacingString = '\n\n(Spacing %s %s %s)\n\n' %tuple(spacing)
    initString = '\n\n(InitialTransformParametersFileName \"%s\")\n\n' %initialTransformFile
    template = initString +spacingString+ template
    outFile = open(outPath,'w')
    outFile.write(template)
    outFile.close()
    return


def register_linear_elastix_seq(fixed,moving,t0=None,degree=2,elastix_dir=None):

    """
    assumes clahe images
    degree explanation:
    0: only translation
    1: trans + rot
    2: trans + rot + affine
    """

    elx_initial_transform_template_string = """
(Transform "AffineTransform")
(NumberOfParameters 12)

(HowToCombineTransforms "Compose")

(InitialTransformParametersFileName "NoInitialTransform")

// Image specific
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "short")
(MovingInternalImagePixelType "short")
//(UseDirectionCosines "false")

(CenterOfRotationPoint 0 0 0)
"""

    fixed_origin = fixed.origin[::-1]
    fixed_spacing = fixed.spacing[::-1]
    moving_origin = moving.origin[::-1]
    moving_spacing = moving.spacing[::-1]

    if not np.any(np.array(fixed.shape)-np.array(moving.shape)):
        if not np.any(fixed-moving):
            if not np.any(fixed_origin-moving_origin):
                if not np.any(fixed_spacing-moving_spacing):
                    return np.array([1,0,0,0,1,0,0,0,1,0,0,0])

    import subprocess,shutil
    if 'win' in sys.platform[:3]:
        elastix_bin = os.path.join(elastix_dir,'elastix.exe')
    elif 'lin' in sys.platform[:3]:
        elastix_bin = os.path.join(elastix_dir,'bin/elastix')
        os.environ['LD_LIBRARY_PATH'] = os.path.join(elastix_dir,'lib')
    elif 'dar' in sys.platform[:3]:
        elastix_bin = os.path.join(elastix_dir,'bin/elastix')
        os.environ['DYLD_LIBRARY_PATH'] = os.path.join(elastix_dir,'lib')

    # temp_dir = tempfile.mkdtemp(prefix = '/data/malbert/tmp/tmp_')
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    outdir = temp_dir
    # param_path_similarity = os.path.join(temp_dir,'elx_params_similarity.txt')
    param_strings = [params_translation,params_rotation,params_affine][:degree+1]
    # param_strings = [params_translation,params_rotation]
    # param_strings = [params_translation]#,params_rotation,params_affine]
    param_paths = [os.path.join(temp_dir,'elx_params_%s.txt' %i) for i in range(len(param_strings))]
    # param_path_affine = os.path.join(temp_dir,'elx_params_affine.txt')

    # choose scaling factors so that starting image has 10 pixel width
    # highest_factor = int(np.min(fixed.shape)/10)
    # factors = [highest_factor]*3 + [np.max([1,highest_factor/2.])]*3 + [np.max([1,highest_factor/4.])]*3  + [np.max([1,highest_factor/8.])]*3

    # choose scaling factors so that highest factor is associated to an image with >10 pixels min shape
    factors = []
    image_pyramid_line = "\n(ImagePyramidSchedule"
    number_of_iterations_line = "\n(MaximumNumberOfIterations"

    for i in range(1+int(np.trunc(np.log2(np.min(list(moving.shape)+list(fixed.shape))/20.))))[::-1]:
        f = int(2**i)
        factors.append(f)
        image_pyramid_line +=" %s %s %s" %(f,f,f)
        number_of_iterations_line +=" %s" %1000

    image_pyramid_line += ")"
    number_of_iterations_line += ")"
    image_pyramid_line +="\n(NumberOfResolutions %s)" %len(factors)

    for i in range(len(param_strings))[:1]:
        param_strings[i] = param_strings[i] + image_pyramid_line + number_of_iterations_line
    # mod_params_similarity = params_similarity + image_pyramid_line
    # mod_params_affine = params_rotation + image_pyramid_line + number_of_iterations_line

    ####################################
    # choose scaling factors so that highest factor is associated to an image with >10 pixels min shape
    factors = []
    image_pyramid_line = "\n(ImagePyramidSchedule"
    number_of_iterations_line = "\n(MaximumNumberOfIterations"

    for i in range(1+int(np.trunc(np.log2(np.min(list(moving.shape)+list(fixed.shape))/20.))))[::-1]:
        f = int(2**i)
        factors.append(f)
        image_pyramid_line +=" %s %s %s" %(f,f,f)
        number_of_iterations_line +=" %s" %1000

    image_pyramid_line += ")"
    number_of_iterations_line += ")"
    image_pyramid_line +="\n(NumberOfResolutions %s)" %len(factors)

    for i in range(len(param_strings))[1:2]:
        param_strings[i] = param_strings[i] + image_pyramid_line + number_of_iterations_line
    # mod_params_similarity = params_similarity + image_pyramid_line
    # mod_params_affine = params_rotation + image_pyramid_line + number_of_iterations_line
    ####################################

    ####################################
    # choose scaling factors so that highest factor is associated to an image with >10 pixels min shape
    factors = []
    image_pyramid_line = "\n(ImagePyramidSchedule"
    number_of_iterations_line = "\n(MaximumNumberOfIterations"

    for i in range(1+int(np.trunc(np.log2(np.min(list(moving.shape)+list(fixed.shape))/20.))))[::-1][-1:]:
        f = int(2**i)
        factors.append(f)
        image_pyramid_line +=" %s %s %s" %(f,f,f)
        number_of_iterations_line +=" %s" %1000

    image_pyramid_line += ")"
    number_of_iterations_line += ")"
    image_pyramid_line +="\n(NumberOfResolutions %s)" %len(factors)

    for i in range(len(param_strings))[2:]:
        param_strings[i] = param_strings[i] + image_pyramid_line + number_of_iterations_line
    # mod_params_similarity = params_similarity + image_pyramid_line
    # mod_params_affine = params_rotation + image_pyramid_line + number_of_iterations_line
    ####################################

    if t0 is not None:
        t0 = np.array(t0)
        t0_inv = np.array(params_invert_coordinates(t0))
        elx_initial_transform_path = os.path.join(temp_dir,'elx_initial_transform.txt')
        createInitialTransformFile(np.array(fixed_spacing), t0_inv, elx_initial_transform_template_string, elx_initial_transform_path)

    createParameterFile(np.array(fixed_spacing),elx_initial_transform_path, param_strings[0], param_paths[0])
    for i in range(1,len(param_strings)):
        open(param_paths[i],'w').write(param_strings[i])
    # open(param_path_similarity,'w').write(mod_params_similarity)
    # open(param_path_affine,'w').write(mod_params_affine)

    fixed_path = os.path.join(temp_dir,'fixed.mhd')
    moving_path = os.path.join(temp_dir,'moving.mhd')
    fixedsitk = sitk.GetImageFromArray(fixed)
    fixedsitk.SetSpacing(fixed_spacing)
    fixedsitk.SetOrigin(fixed_origin)

    movingsitk = sitk.GetImageFromArray(moving)
    movingsitk.SetSpacing(moving_spacing)
    movingsitk.SetOrigin(moving_origin)

    # set fixed mask
    fixed_mask_path = os.path.join(temp_dir,'fixed_mask.mhd')
    # fixed_clahe = clahe(fixed,40,clip_limit=0.02)
    fixed_clahe = fixed
    fixed_mask = get_mask_using_otsu(fixed_clahe)
    # print('warning: simple mask in elastix call')
    # fixed_mask = (np.array(fixed_clahe) > np.mean(fixed_clahe)).astype(np.uint16)

    fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask)
    fixed_mask_sitk.SetSpacing(fixed_spacing)
    fixed_mask_sitk.SetOrigin(fixed_origin)
    sitk.WriteImage(fixed_mask_sitk,fixed_mask_path)

    sitk.WriteImage(fixedsitk,fixed_path)
    sitk.WriteImage(movingsitk,moving_path)

    # FNULL = open(os.devnull, 'w')
    # cmd = '%s -f %s -m %s -p %s -p %s -out %s -threads 1' %(elastix_bin,fixed_path,moving_path,param_path_similarity,param_path_affine,outdir)
    # cmd = '%s -f %s -m %s -t0 %s' %(elastix_bin,fixed_path,moving_path,elx_initial_transform_path)
    cmd = '%s -f %s -m %s -t0 %s' %(elastix_bin,fixed_path,moving_path,elx_initial_transform_path)
    for i in range(len(param_strings)):
        cmd += ' -p %s' %param_paths[i]
    cmd += ' -fMask %s' %fixed_mask_path
    cmd += ' -out %s' %outdir
    # cmd += ' -threads 1' %outdir

    cmd = cmd.split(' ')
    # subprocess.Popen(cmd,stdout=FNULL).wait()
    subprocess.Popen(cmd).wait()

    final_params = t0
    for i in range(len(param_strings)):
        final_params = matrix_to_params(get_affine_parameters_from_elastix_output(os.path.join(temp_dir,'TransformParameters.%s.txt' %i),t0=final_params))

    return final_params

@io_decorator
def register_linear_elastix(fixed,moving,degree=2,elastix_dir=None):

    """
    estimate t0 and crop images to intersection in y
    :param fixed:
    :param moving:
    :return:
    """

    lower_y0 = fixed.origin[1]
    upper_y0 = fixed.origin[1] + fixed.shape[1]*fixed.spacing[1]

    lower_y1 = moving.origin[1]
    upper_y1 = moving.origin[1] + moving.shape[1]*moving.spacing[1]

    lower_overlap = np.max([lower_y0,lower_y1])
    upper_overlap = np.min([upper_y0,upper_y1])

    yl0 = int((lower_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yu0 = int((upper_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yl1 = int((lower_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])
    yu1 = int((upper_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])

    # images can have different overlaps because of rounding to integer

    origin_overlap0 = np.zeros(3)
    origin_overlap1 = np.zeros(3)

    origin_overlap0[:] = fixed.origin
    origin_overlap1[:] = moving.origin

    origin_overlap0[1] = lower_y0 + yl0 * fixed.spacing[1]
    origin_overlap1[1] = lower_y1 + yl1 * moving.spacing[1]

    # static = ImageArray(fixed[:,yl0:yu0,:],spacing=fixed.spacing,origin=origin_overlap0)
    # mov = ImageArray(moving[:,yl1:yu1,:],spacing=moving.spacing,origin=origin_overlap1)

    c0 = clahe(fixed,10,clip_limit=0.02)
    c1 = clahe(moving,10,clip_limit=0.02)

    # print('warning: not performing clahe')
    # c0 = fixed
    # c1 = moving

    static = ImageArray(c0[:,yl0:yu0,:],spacing=fixed.spacing,origin=origin_overlap0)
    mov = ImageArray(c1[:,yl1:yu1,:],spacing=moving.spacing,origin=origin_overlap1)

    t00 = mv_utils.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    center_static = np.array(static.shape)/2.*static.spacing + static.origin
    center_mov = np.array(mov.shape)/2.*mov.spacing + mov.origin
    t00offset = center_mov - np.dot(t00[:3,:3],center_static)
    t00[:3,3] = t00offset
    t00 = matrix_to_params(t00)

    # reg_spacing = np.array([fixed.spacing[0]*4]*3)
    print('WARNING: 20180614: changed fft registration spacing')
    reg_iso_spacing = np.min([np.array(im.spacing)*np.array(im.shape)/160. for im in [static,mov]])
    reg_iso_spacing = np.max([[reg_iso_spacing]+list(static.spacing)+list(mov.spacing)])
    reg_spacing = np.array([reg_iso_spacing]*3)

    stack_properties = calc_stack_properties_from_views_and_params([static,mov],[matrix_to_params(np.eye(4)),t00],
                                                                   spacing=reg_spacing,mode='union')

    static_t = transform_stack_sitk(static,
                                    matrix_to_params(np.eye(4)),
                                    out_shape=stack_properties['size'],
                                    out_origin=stack_properties['origin'],
                                    out_spacing=stack_properties['spacing']
                                    )

    mov_t = transform_stack_sitk(mov,
                                    t00,
                                    out_shape=stack_properties['size'],
                                    out_origin=stack_properties['origin'],
                                    out_spacing=stack_properties['spacing']
                                    )

    im0 = static_t
    im1 = mov_t

    offset = translation3d(im1,im0)

    # offset = np.array([-offset[2],0,offset[0]]) * reg_spacing
    # offset = np.array([offset[0],0,offset[2]]) * reg_spacing
    print('WARNING: add complete FFT offset (also y component), 20181109')
    offset = np.array([offset[0],offset[1],offset[2]]) * reg_spacing

    t0 = np.copy(t00)
    t0[9:] += np.dot(t0[:9].reshape((3,3)),offset)
    # return t0
    parameters = register_linear_elastix_seq(static,mov,t0,degree=degree,elastix_dir=elastix_dir)
    return parameters


from numpy.fft import fftn, ifftn, fftshift
def translation3d(im0, im1):
    """Return translation vector to register images."""

    # fill zeros with noise
    im0_m = np.copy(im0)
    im1_m = np.copy(im1)

    print('WARNING: ADDING NOISE IN FFT REGISTRATION (added 20181109)')
    print('stack size in FFT translation registration: %s' %list(im0.shape))
    # n0 = [im0_m[im0_m>0].min(),np.percentile(im0_m[im0_m>0],5)]
    n0 = [0.05,0.15] # typical border values resulting from applying clahe to Z1 background
    im0_m[im0==0] = np.random.random(np.sum(im0==0))*(n0[1]-n0[0])+n0[0]

    # n1 = [im1_m[im1_m>0].min(),np.percentile(im1_m[im1_m>0],5)]
    n1 = [0.05,0.15] # typical border values resulting from applying clahe to Z1 background
    im1_m[im1==0] = np.random.random(np.sum(im1==0))*(n1[1]-n1[0])+n1[0]

    shape = im0.shape
    # f0 = fftn(im0)
    # f1 = fftn(im1)
    f0 = fftn(im0_m)
    f1 = fftn(im1_m)
    ir = abs(ifftn((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))

    print('WARNING: FILTERING IN FFT REGISTRATION (added 20181109)')
    ir_gauss = ndimage.gaussian_filter(ir,1)

    # t0, t1, t2 = np.unravel_index(np.argmax(ir), shape)
    t0, t1, t2 = np.unravel_index(np.argmax(ir_gauss), shape)

    # if t0 > shape[0] // 2:
    #     t0 -= shape[0]
    # if t1 > shape[1] // 2:
    #     t1 -= shape[1]
    # if t2 > shape[2] // 2:
    #     t2 -= shape[2]

    if t0 > shape[0] // 2: t0 -= shape[0]
    if t1 > shape[1] // 2: t1 -= shape[1]
    if t2 > shape[2] // 2: t2 -= shape[2]

    return [t0, t1, t2]

# import transformations
def get_affine_parameters_from_elastix_output(filepath_or_params,t0=None):


    if type(filepath_or_params) == str:

        raw_out_params = open(filepath_or_params).read()

        elx_out_params = raw_out_params.split('\n')[2][:-1].split(' ')[1:]
        elx_out_params = np.array([float(i) for i in elx_out_params])

        # if len(elx_out_params) in [6, 7, 12]:
        if len(elx_out_params) in [6, 12]:
            outCenterOfRotation = raw_out_params.split('\n')[19][:-1].split(' ')[1:]
            outCenterOfRotation = np.array([float(i) for i in outCenterOfRotation])

    else:
        elx_out_params = filepath_or_params

        # when input is given as parameters, set center of rotation zero.
        # affinelogstacktransform doesn't use it
        # neither EulerStackTransform
        outCenterOfRotation = np.zeros(3)


    if len(elx_out_params)==6:
        # tmp = transformations.euler_matrix(elx_out_params[0],elx_out_params[1],elx_out_params[2])
        tmp = mv_utils.euler_matrix(elx_out_params[0],elx_out_params[1],elx_out_params[2])
        elx_affine_params = np.zeros(12)
        elx_affine_params[:9] = tmp[:3,:3].flatten()
        elx_affine_params[-3:] = np.array([elx_out_params[3],elx_out_params[4],elx_out_params[5]])

        # translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)

    if len(elx_out_params)==12: # affine case
        elx_affine_params = elx_out_params

    # elif len(elx_out_params)==7: # similarity transform
    #     angles = transformations.euler_from_quaternion([np.sqrt(1-np.sum([np.power(elx_out_params[i],2) for i in range(3)])),
    #                                                     elx_out_params[0],elx_out_params[1],elx_out_params[2]])
    #     tmp = transformations.compose_matrix(angles=angles)
    #     elx_affine_params = np.zeros(12)
    #     elx_affine_params[:9] = tmp[:3,:3].flatten()*elx_out_params[6]
    #     elx_affine_params[-3:] = np.array([elx_out_params[3],elx_out_params[4],elx_out_params[5]])

        # translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)

    elif len(elx_out_params)==3: # translation transform

        elx_affine_params = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])
        elx_affine_params[9:] = elx_out_params

    if len(elx_out_params) in [6,12]:
    # if len(elx_out_params) in [6,7,12]:

        # outCenterOfRotation = np.dot(params_to_matrix(params_invert_coordinates(t0)),np.array(list(outCenterOfRotation)+[1]))[:3]

        translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)


    # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)
    # elx_affine_params_numpy = np.concatenate([elx_affine_params[:9][::-1,::-1],translation[::-1]],0)

    inv_elx_affine_params = params_invert_coordinates(elx_affine_params)
    final_params = params_to_matrix(inv_elx_affine_params)

    if t0 is not None:
        final_params = np.dot(final_params,params_to_matrix(t0))

    return final_params

def register_linear(static,moving,t0=None):

    static_origin = static.origin
    static_spacing = static.spacing
    moving_origin = moving.origin
    moving_spacing = moving.spacing

    if not np.any(np.array(static.shape)-np.array(moving.shape)):
        if not np.any(static-moving):
            return np.array([1,0,0,0,1,0,0,0,1,0,0,0]),[0]

    if t0 is None:
        t0 = np.array([1,0,0,0,1,0,0,0,1,0,0,0])

    # static = ImageArray(clahe(static,[10,10,10],clip_limit=0.02),spacing=static.spacing,origin=static.origin)
    # moving = ImageArray(clahe(moving,[10,10,10],clip_limit=0.02),spacing=moving.spacing,origin=moving.origin)

    # static = static / float(static.max())
    # moving = moving / float(moving.max())

    # static = static.astype(np.float64)/float(static.max())
    # moving = moving.astype(np.float64)/float(moving.max())

    # static = static*(static > 100)#.astype(np.float)
    # moving = moving*(moving > 100)#.astype(np.float)

    # static[static == 0] = np.random.randint(0,10,(np.sum(static == 0)))
    # moving[moving == 0] = np.random.randint(0,10,(np.sum(moving == 0)))

    # import pdb

    static_grid2world = np.eye(4)
    moving_grid2world = np.eye(4)

    np.fill_diagonal(static_grid2world,list(static_spacing)+[1])
    np.fill_diagonal(moving_grid2world,list(moving_spacing)+[1])

    static_grid2world[:3,3] = static_origin
    moving_grid2world[:3,3] = moving_origin

    # moving = clahe(moving,[10,10],clip_limit=0.02)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    # level_iters = list(np.array([10000, 1000, 100])*1000)
    # level_iters = list(np.array([1000000000]*7))
    # sigmas = [6,5,4,3.0,1.0,1.0, 0.0]
    # factors = [6,5,4,3,2,1,1]

    # transform = TranslationTransform3D()
    params0 = None
    starting_affine = np.eye(4)
    starting_affine[:3,:3] = t0[:9].reshape((3,3))
    starting_affine[:3,3] = t0[9:]
    # starting_affine = None

    level_iters = list(np.array([1000000000]*5))
    sigmas = [3,1,0.0,0.0,0.0]
    factors = [4,2,1.5,1.2,1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                ss_sigma_factor = 0.1, # watch out!
                                verbosity=3,
                                options = {'gtol': 1e-8},
                                # method = 'Newton-CG',
                                )

    print('watch out, using ss_sigma_factor and sigmas are being ignored!')

    # transform = AffineTransform2D()
    transform = TranslationTransform3D()
    translation = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                ss_sigma_factor = 0.1, # watch out!
                                verbosity=3,
                                options = {'gtol': 1e-8},
                                # method = 'Newton-CG',
                                )

    transform = RigidTransform3D()
    rotation = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)

    # transform = ShearTransform3D()
    # shearing = affreg.optimize(static, moving, transform, params0,
    #                         static_grid2world, moving_grid2world,
    #                         starting_affine=translation.affine)

    # metric = MutualInformationMetric(nbins, sampling_prop)
    # level_iters = list(np.array([10000, 1000, 100]))
    # sigmas = [3,3,3,3,3.0,1.0,1.0, 0.0]
    # factors = [10,6,5,4,3,2,1,1]
    # affreg = AffineRegistration(metric=metric,
    #                             level_iters=level_iters,
    #                             sigmas=sigmas,
    #                             factors=factors,
    #                             verbosity=3,
    #                             options = {'gtol': 1e-8},
    #                             # method = 'Nelder-Mead'
    #                             # method = 'CG',
    #                             # method = 'Newton-CG' # different result somehow
    #                             # method = 'trust-ncg' # complains
    #                             # method = 'dogleg', #complains
    #                             # method = 'SLSQP',
    #                             )
    #

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                ss_sigma_factor = 0.1, # watch out!
                                verbosity=3,
                                options = {'gtol': 1e-8},
                                # method = 'Newton-CG',
                                )

    transform = AffineTransform3D()
    affine = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=rotation.affine)

    out_params = affine.affine

    params = np.zeros(12)
    params[:9] = out_params[:3,:3].flatten()
    params[9:] = out_params[:3,3]

    # print(metric.metric_evolution)

    params_evolution = np.array([np.append(metric.params_evolution[i][:3,:3].flatten(),metric.params_evolution[i][:3,3]) for i in range(len(metric.params_evolution))])

    # return params,[0]#metric.metric_evolution
    return params,[metric.metric_evolution,params_evolution]

def transform_stack_sitk(stack,p=None,out_shape=None,out_spacing=None,out_origin=None,interp='bspline',stack_properties=None):

    # print('WARNING: USING BSPLINE INTERPOLATION AS DEFAULT')
    if p is None:
        p = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])

    p = np.array(p)

    p = params_invert_coordinates(p)

    if stack_properties is not None:
        out_shape = stack_properties['size']
        out_origin = stack_properties['origin']
        out_spacing = stack_properties['spacing']

    else:
        if out_shape is None:
            out_shape = stack.shape

        if out_origin is None:
            out_origin = stack.origin

        if out_spacing is None:
            out_spacing = stack.spacing
    #
    # sstack = sitk.GetImageFromArray(stack)
    # sstack.SetSpacing(stack.spacing[::-1])
    # sstack.SetOrigin(stack.origin[::-1])

    trivial = True
    if not np.allclose(p,matrix_to_params(np.eye(4))):
        trivial = False
    if not np.allclose(out_shape, stack.shape):
        trivial = False
    if not np.allclose(out_origin, stack.origin):
        trivial = False
    if not np.allclose(out_spacing, stack.spacing):
        trivial = False

    if trivial: return stack

    sstack = image_to_sitk(stack)
    sstack = transformStack(p,sstack,
                            outShape=out_shape[::-1],
                            outSpacing=out_spacing[::-1],
                            outOrigin=out_origin[::-1],interp = interp)

    sstack = sitk.GetArrayFromImage(sstack)
    sstack = ImageArray(sstack,origin = out_origin, spacing=out_spacing)

    return sstack

# def apply_chromatic_correction_parameters(stack,p):
#
#     pprime = np.array(p)
#     # center = np.zeros(3)
#     # center[1:] = np.array(stack.shape[1:])/2.
#     # center = np.array(stack.shape)/2. * stack.spacing
#     #
#     # # transform params into origin 0 frame
#     # M = pprime[:9].reshape((3,3))
#     # cprime = pprime[9:]
#     # c = cprime + np.dot(M,center)
#     #
#     # params = np.zeros(12)
#     # params[:9] = p[:9]
#     # params[9:] = c
#
#     # s = ImageArray(stack,origin=[0,0,0],spacing=[1,1,1])
#     s = ImageArray(stack,origin=[0,0,0],spacing=stack.spacing)
#
#     # s = transform_stack_sitk(s,params)
#     s = transform_stack_sitk(s,p)
#
#     s.origin = stack.origin
#     s.spacing = stack.spacing
#
#     return s

@io_decorator
def apply_chromatic_correction_parameters_center(stack,p):

    # hack to be able to io_decorate chrom corr params
    # prolem is that we don't want to write to disk all the intermediate results
    if io_utils.is_io_path(p):
        p = io_utils.process_input_element(p)
    # pprime = np.array(p)
    # center = np.zeros(3)
    # center[1:] = np.array(stack.shape[1:])/2.
    # center = np.array(stack.shape)/2. * stack.spacing
    #
    # # transform params into origin 0 frame
    # M = pprime[:9].reshape((3,3))
    # cprime = pprime[9:]
    # c = cprime + np.dot(M,center)
    #
    # params = np.zeros(12)
    # params[:9] = p[:9]
    # params[9:] = c

    # s = ImageArray(stack,origin=[0,0,0],spacing=[1,1,1])
    # s = ImageArray(stack,origin=[0,0,0],spacing=stack.spacing)
    s = ImageArray(stack,origin=stack.spacing*np.array(stack.shape)/2.,spacing=stack.spacing)

    # s = transform_stack_sitk(s,params)
    s = transform_stack_sitk(s,p)

    s.origin = stack.origin
    s.spacing = stack.spacing

    return s

def calc_initial_parameters(pairs,infoDict,view_stacks):

    # origins = infoDict['origins']
    positions = infoDict['positions']
    # positions = np.array([i.origin for i in view_stacks])
    # spacing = infoDict['spacing']
    spacing = view_stacks[0].spacing
    # centerOfRotation = infoDict['centerOfRotation']
    # centerOfRotation = np.zeros(3)
    # axisOfRotation = infoDict['axisOfRotation']
    sizes = infoDict['sizes']

    print('calculating transformations')
    parameters = []
    # y_ranges = []
    for iview0,iview1 in pairs:

        if iview0 == iview1:
            parameters.append([1., 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
            continue

        matrix = mv_utils.euler_matrix(0,+ positions[iview0][3] - positions[iview1][3],0)
        # matrix = mv_utils.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
        tmpParams = np.append(matrix[:3, :3].flatten(), matrix[:3, 3])


        lower_y0 = view_stacks[iview0].origin[1]
        upper_y0 = view_stacks[iview0].origin[1] + view_stacks[iview0].shape[1]*view_stacks[iview0].spacing[1]

        lower_y1 = view_stacks[iview1].origin[1]
        upper_y1 = view_stacks[iview1].origin[1] + view_stacks[iview1].shape[1]*view_stacks[iview1].spacing[1]

        # yl0 = int((np.max([lower_y0,lower_y1])-lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
        # yu0 = int((np.min([upper_y0,upper_y1])-lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
        # yl1 = int((np.max([lower_y0,lower_y1])-lower_y1) / (upper_y1-lower_y1) * moving.shape[1])
        # yu1 = int((np.min([upper_y0,upper_y1])-lower_y1) / (upper_y1-lower_y1) * moving.shape[1])

        lower_overlap = np.max([lower_y0,lower_y1])
        upper_overlap = np.min([upper_y0,upper_y1])

        yl0 = int((lower_overlap - lower_y0) / (upper_y0-lower_y0) * view_stacks[iview0].shape[1])
        yu0 = int((upper_overlap - lower_y0) / (upper_y0-lower_y0) * view_stacks[iview0].shape[1])
        yl1 = int((lower_overlap - lower_y1) / (upper_y1-lower_y1) * view_stacks[iview1].shape[1])
        yu1 = int((upper_overlap - lower_y1) / (upper_y1-lower_y1) * view_stacks[iview1].shape[1])

        origin_overlap0 = np.zeros(3)
        origin_overlap1 = np.zeros(3)

        origin_overlap0[:] = view_stacks[iview0].origin
        origin_overlap1[:] = view_stacks[iview1].origin

        origin_overlap0[1] = lower_y0 + yl0 * view_stacks[iview0].spacing[1]
        origin_overlap1[1] = lower_y1 + yl1 * view_stacks[iview1].spacing[1]
        # lower_y0 = positions[iview0][1]#-sizes[iview0][1]/2.*spacing[1]
        # upper_y0 = positions[iview0][1]+sizes[iview0][1]*spacing[1]
        #
        # lower_y1 = positions[iview1][1]#-sizes[iview1][1]/2.*spacing[1]
        # upper_y1 = positions[iview1][1]+sizes[iview1][1]*spacing[1]
        #
        # # y_overlap = np.min([upper_y0,upper_y1]) - np.max([lower_y0,lower_y1])
        #
        # yl0 = int((np.max([lower_y0,lower_y1])-lower_y0) / (upper_y0-lower_y0) * view_stacks[iview0].shape[1])
        # yu0 = int((np.min([upper_y0,upper_y1])-lower_y0) / (upper_y0-lower_y0) * view_stacks[iview0].shape[1])
        # yl1 = int((np.max([lower_y0,lower_y1])-lower_y1) / (upper_y1-lower_y1) * view_stacks[iview1].shape[1])
        # yu1 = int((np.min([upper_y0,upper_y1])-lower_y1) / (upper_y1-lower_y1) * view_stacks[iview1].shape[1])

        c0 = clahe(view_stacks[iview0],40,clip_limit=0.02)[:,yl0:yu0,:]
        c1 = clahe(view_stacks[iview1],40,clip_limit=0.02)[:,yl1:yu1,:]

        m0 = get_mask_using_otsu(c0)
        m1 = get_mask_using_otsu(c1)

        # mean0 = ndimage.center_of_mass(np.array(view_stacks[iview0][:,yl0:yu0,:])) * view_stacks[iview0].spacing + view_stacks[iview0].origin
        # mean1 = ndimage.center_of_mass(np.array(view_stacks[iview1][:,yl1:yu1,:])) * view_stacks[iview1].spacing + view_stacks[iview1].origin

        mean0 = ndimage.center_of_mass(m0) * view_stacks[iview0].spacing + origin_overlap0
        mean1 = ndimage.center_of_mass(m1) * view_stacks[iview1].spacing + origin_overlap1


        print(mean0,mean1)

        offset = mean1 - np.dot(matrix[:3,:3],mean0)
        # offset[1] = (positions[iview1][1] - positions[iview0][1]) / spacing[1] * view_stacks[iview0].spacing[1]
        tmpParams[9:] = offset
        tmpParams[10] = 0
        # print('watch out, using y offset = 0')

        parameters.append(tmpParams)

    return np.array(parameters)

def calc_t0(fixed,moving):

    if not np.any(np.array(fixed.shape)-np.array(moving.shape)):
        if not np.any(fixed-moving):
            if not np.any(fixed.origin-moving.origin):
                if not np.any(fixed.spacing-moving.spacing):
                    return np.array([1,0,0,0,1,0,0,0,1,0,0,0])

    matrix = mv_utils.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    # matrix = mv_utils.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
    tmpParams = np.append(matrix[:3, :3].flatten(), matrix[:3, 3])


    lower_y0 = fixed.origin[1]
    upper_y0 = fixed.origin[1] + fixed.shape[1]*fixed.spacing[1]

    lower_y1 = moving.origin[1]
    upper_y1 = moving.origin[1] + moving.shape[1]*moving.spacing[1]

    lower_overlap = np.max([lower_y0,lower_y1])
    upper_overlap = np.min([upper_y0,upper_y1])

    yl0 = int((lower_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yu0 = int((upper_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yl1 = int((lower_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])
    yu1 = int((upper_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])

    origin_overlap0 = np.zeros(3)
    origin_overlap1 = np.zeros(3)

    origin_overlap0[:] = fixed.origin
    origin_overlap1[:] = moving.origin

    origin_overlap0[1] = lower_y0 + yl0 * fixed.spacing[1]
    origin_overlap1[1] = lower_y1 + yl1 * moving.spacing[1]

    c0 = clahe(fixed,40,clip_limit=0.02)[:,yl0:yu0,:]
    c1 = clahe(moving,40,clip_limit=0.02)[:,yl1:yu1,:]

    m0 = get_mask_using_otsu(c0)
    m1 = get_mask_using_otsu(c1)

    mean0 = ndimage.center_of_mass(m0) * fixed.spacing + origin_overlap0
    mean1 = ndimage.center_of_mass(m1) * moving.spacing + origin_overlap1


    print(mean0,mean1)

    offset = mean1 - np.dot(matrix[:3,:3],mean0)
    tmpParams[9:] = offset
    tmpParams[10] = 0

    parameters = np.array(tmpParams)

    return parameters

@io_decorator
def concatenate_view_and_time_params(time_params,view_params):
    # time_paramss order: early timepoints first (last entry is current time alignment against previous timepoint)
    # m_time_paramss = [params_to_matrix(time_params) for time_params in time_paramss]
    m_time_params = params_to_matrix(time_params)
    m_view_params = params_to_matrix(view_params)

    # # concatenate time params
    # m_final_time_params = m_time_paramss[0]
    # for m_time_params in m_time_paramss[1:]:
    #     m_final_time_params = np.dot(m_time_params,m_final_time_params)

    m_final_params = np.dot(m_view_params,m_time_params)
    # m_final_params = np.dot(m_view_params,m_final_time_params)
    final_params = matrix_to_params(m_final_params)
    return final_params

def matrix_to_params(M):
    if M.shape[0] == 4:
        params = np.zeros(12)
        params[:9] = M[:3,:3].flatten()
        params[9:] = M[:3,3]
    if M.shape[0] == 3:
        params = np.zeros(6)
        params[:4] = M[:-1,:-1].flatten()
        params[4:] = M[:-1,-1]
    return params

def params_to_matrix(params):
    params = np.array(params)
    if len(params) == 12:
        M = np.eye(4)
        M[:-1,:-1] = params[:9].reshape((3,3))
        M[:-1,-1] = params[9:]
    else:
        M = np.eye(3)
        M[:-1,:-1] = params[:4].reshape((2,2))
        M[:-1,-1] = params[4:]

    return M

def params_invert_coordinates(params):
    M = params_to_matrix(params)
    M[:-1,:-1] = M[:-1,:-1][::-1,::-1]
    M[:-1,-1] = M[:-1,-1][::-1]
    return matrix_to_params(M)

def invert_params(params):
    M = params_to_matrix(params)
    M[:-1,:-1] = np.linalg.inv(M[:-1,:-1])
    M[:-1,-1] = -np.dot(M[:-1,:-1], M[:-1,-1])
    return matrix_to_params(M)

def get_mask_using_otsu(im):
    from skimage import filters
    thresh = filters.threshold_otsu(im)
    thresh = filters.threshold_otsu(im[im<thresh]) # line added 20190702
    seg = im > thresh
    seg = ndimage.binary_erosion(seg,iterations=1)
    seg = ndimage.binary_dilation(seg,iterations=5)
    return seg.astype(np.uint16)

@io_decorator
def get_params_from_pairs(ref_view,pairs,params,time_alignment_params=None):
    """
    time_alignment_params: single params from longitudinal registration to be concatenated with view params
    """

    import networkx
    g = networkx.DiGraph()
    for ipair,pair in enumerate(pairs):
        # g.add_edge(pair[0],pair[1],{'p': params[ipair]})
        g.add_edge(pair[0],pair[1], p = params[ipair]) # after update 201809 networkx seems to have changed
        g.add_edge(pair[1],pair[0], p = invert_params(params[ipair])) # after update 201809 networkx seems to have changed

    all_views = np.unique(np.array(pairs).flatten())
    # views_to_transform = np.sort(np.array(list(set(all_views).difference(set([ref_view])))))

    final_params = []
    for iview,view in enumerate(all_views):
        if view == ref_view:
            # final_params.append(matrix_to_params(np.eye(4)))
            final_view_params = matrix_to_params(np.eye(4))

        else:
            paths = networkx.all_shortest_paths(g,ref_view,view)
            paths_params = []
            for ipath,path in enumerate(paths):
                path_pairs = [[path[i],path[i+1]] for i in range(len(path)-1)]
                print(path_pairs)
                path_params = np.eye(4)
                for edge in path_pairs:
                    tmp_params = params_to_matrix(g.get_edge_data(edge[0],edge[1])['p'])
                    path_params = np.dot(tmp_params,path_params)
                    print(path_params)
                paths_params.append(matrix_to_params(path_params))

            final_view_params = np.mean(paths_params,0)

        # concatenate with time alignment if given
        if time_alignment_params is not None:
            final_view_params = concatenate_view_and_time_params(time_alignment_params,final_view_params)

        final_params.append(final_view_params)

    return np.array(final_params)

def register_groupwise_cmd(ims, params0, ref_view = 0, elastix_dir='/scratch/malbert/dependencies_linux/elastix_linux64_v4.8'):

    """
    - perform groupwise registration (to refine and globally optimise after pair registration)
    - first euler, then affine works well
    - works only with elastix v4.9
    - principle: use varianceoverlastdimension metric together with affinelogstacktransform
    - relate parameters back to reference view

    :param ims:
    :param params0:
    :param elastix_dir:
    :return:
    """

    elx_param_string0 = \
    """
MaximumNumberOfIterations ('1000',)
NumberOfResolutions ('4',)
NumberOfSpatialSamples ('8192',)
Transform ('EulerStackTransform',)

AutomaticParameterEstimation ('true',)
CheckNumberOfSamples ('true',)
DefaultPixelValue ('0.0',)
FinalBSplineInterpolationOrder ('3',)
FinalGridSpacingInPhysicalUnits ('8',)
FixedImagePyramid ('FixedSmoothingImagePyramid',)
FixedInternalImagePixelType ('float',)
GridSpacingSchedule ('2.80322', '1.9881', '1.41', '1')
ImageSampler ('RandomCoordinate',)
Interpolator ('ReducedDimensionBSplineInterpolator',)
MaximumNumberOfSamplingAttempts ('8',)
Metric ('VarianceOverLastDimensionMetric',)
MovingImagePyramid ('MovingSmoothingImagePyramid',)
MovingInternalImagePixelType ('float',)
NewSamplesEveryIteration ('true',)
NumberOfSamplesForExactGradient ('4096',)
Optimizer ('AdaptiveStochasticGradientDescent',)
Registration ('MultiResolutionRegistration',)
ResampleInterpolator ('FinalReducedDimensionBSplineInterpolator',)
Resampler ('DefaultResampler',)
ResultImageFormat ('nii',)
WriteIterationInfo ('false',)
WriteResultImage ('true',)
    """

    elx_param_string1 = \
        """
MaximumNumberOfIterations ('1000',)
NumberOfResolutions ('1',)
NumberOfSpatialSamples ('8192',)
Transform ('AffineLogStackTransform',)
        
AutomaticParameterEstimation ('true',)
CheckNumberOfSamples ('true',)
DefaultPixelValue ('0.0',)
FinalBSplineInterpolationOrder ('3',)
FinalGridSpacingInPhysicalUnits ('8',)
FixedImagePyramid ('FixedSmoothingImagePyramid',)
FixedInternalImagePixelType ('float',)
GridSpacingSchedule ('2.80322', '1.9881', '1.41', '1')
ImageSampler ('RandomCoordinate',)
Interpolator ('ReducedDimensionBSplineInterpolator',)
MaximumNumberOfSamplingAttempts ('8',)
Metric ('VarianceOverLastDimensionMetric',)
MovingImagePyramid ('MovingSmoothingImagePyramid',)
MovingInternalImagePixelType ('float',)
NewSamplesEveryIteration ('true',)
NumberOfSamplesForExactGradient ('4096',)
Optimizer ('AdaptiveStochasticGradientDescent',)
Registration ('MultiResolutionRegistration',)
ResampleInterpolator ('FinalReducedDimensionBSplineInterpolator',)
Resampler ('DefaultResampler',)
ResultImageFormat ('nii',)
WriteIterationInfo ('false',)
WriteResultImage ('true',)
        """

    # if len(ims) < 4: raise(Exception('groupwise registration needs at least four images'))

    elastix_bin = os.path.join(elastix_dir,'bin/elastix')
    os.environ['LD_LIBRARY_PATH'] = os.path.join(elastix_dir,'lib')

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    param_strings = [elx_param_string0, elx_param_string1]
    param_paths = [os.path.join(temp_dir, 'elx_params_%s.txt' % i) for i in range(len(param_strings))]

    # if params0 is not None:
    #     t0 = np.array(t0)
    #     t0_inv = np.array(params_invert_coordinates(t0))
    #     elx_initial_transform_path = os.path.join(temp_dir,'elx_initial_transform.txt')
    #     createInitialTransformFile(np.array(fixed_spacing), t0_inv, elx_initial_transform_template_string, elx_initial_transform_path)

    # createParameterFile(np.array([1,1,1]),elx_initial_transform_path, param_strings[0], param_paths[0])
    for i in range(0,len(param_strings)):
        open(param_paths[i],'w').write(param_strings[i])

    elx_initial_transform_path = 0

    vectorOfImages = sitk.VectorOfImage()
    for iim,im in enumerate(ims):
        vectorOfImages.push_back(image_to_sitk(im))
    fixed_path = os.path.join(temp_dir,'fixed.mhd')
    vectorOfImages = sitk.JoinSeries(vectorOfImages)
    sitk.WriteImage(vectorOfImages,fixed_path)

    cmd = '%s -f %s -m %s' %(elastix_bin,fixed_path,fixed_path)#,elx_initial_transform_path)
    if params0 is not None:
        cmd += '-t0 %s' % elx_initial_transform_path
    for i in range(len(param_strings)):
        cmd += ' -p %s' %param_paths[i]
    # cmd += ' -fMask %s' %fixed_mask_path
    cmd += ' -out %s' %temp_dir
    # cmd += ' -threads 1' %outdir
    print(temp_dir)
    # print(temp_dir)
    #
    cmd = cmd.split(' ')
    # subprocess.Popen(cmd,stdout=FNULL).wait()
    import subprocess
    subprocess.Popen(cmd).wait()

    # final_params = t0
    # for i in range(len(param_strings)):
    #     final_params = matrix_to_params(get_affine_parameters_from_elastix_output(os.path.join(temp_dir,'TransformParameters.%s.txt' %i),t0=final_params))

    return cmd

# from scipy.linalg import expm,logm
# import transformations
# @io_decorator
# def register_groupwise(orig_ims, params0, ref_view_index=0, iso_reg_spacing_relative_to_input_z=1, volume_mode='sample',elastix_dir='/scratch/malbert/dependencies_linux/elastix_linux64_v4.8'):
#     """
#     - perform groupwise registration (to refine and globally optimise after pair registration)
#     - first euler, then affine works well
#     - works only with elastix v4.9
#     - principle: use varianceoverlastdimension metric together with affinelogstacktransform
#     - relate parameters back to reference view
#
#     :param ims:
#     :param params0:
#     :param elastix_dir:
#     :return:
#     """
#
#     reg_spacing = np.array([orig_ims[0].spacing[0]*iso_reg_spacing_relative_to_input_z]*3).astype(np.float32)
#     stack_properties = calc_stack_properties_from_views_and_params(orig_ims,params0,reg_spacing,volume_mode)
#
#     ims = []
#     for iview in range(len(orig_ims)):
#         im = transform_stack_sitk(orig_ims[iview], params0[iview],
#                                      out_origin=stack_properties['origin'],
#                                      out_shape=stack_properties['size'],
#                                      out_spacing=stack_properties['spacing'])
#         ims.append(im)
#
#     vectorOfImages = sitk.VectorOfImage()
#     for iim,im in enumerate(ims):
#         vectorOfImages.push_back(image_to_sitk(im))
#
#     image = sitk.JoinSeries(vectorOfImages)
#
#     log_affine_params = []
#     for i in range(len(orig_ims)):
#         tmpparams = params_to_matrix(params0[i])
#         tmpparams = logm(tmpparams)
#         tmpparams = matrix_to_params(tmpparams)
#         log_affine_params += list(tmpparams)
#
#     log_affine_params_string = ' '.join([float(i) for i in log_affine_params])
#
#     import subprocess, shutil
#
#     elastix_bin = os.path.join(elastix_dir, 'bin/elastix')
#     os.environ['LD_LIBRARY_PATH'] = os.path.join(elastix_dir, 'lib')
#
#     # temp_dir = tempfile.mkdtemp(prefix = '/data/malbert/tmp/tmp_')
#     temp_dir_obj = tempfile.TemporaryDirectory()
#     temp_dir = temp_dir_obj.name
#
#     init_transform_path = os.path.join(temp_dir,'initial_affine_log.txt')
#     image_path = os.path.join(temp_dir,'im4d.mhd')
#     params_paths = [os.path.join(temp_dir,'p_euler.mhd'),os.path.join(temp_dir,'p_bspline.mhd')]
#
#     sitk.WriteImage(image,image_path)
#
# initialTransform = '\n(TransformParameters %s)' %log_affine_params_string +"""
# (Transform "AffineLogStackTransform")
# (NumberOfParameters %s)
#
# (HowToCombineTransforms "Compose")
#
# (InitialTransformParametersFileName "NoInitialTransform")
#
# // Image specific
# (FixedImageDimension 3)
# (MovingImageDimension 3)
# (FixedInternalImagePixelType "short")
# (MovingInternalImagePixelType "short")
# //(UseDirectionCosines "false")
#
# (CenterOfRotationPoint 0 0 0)
# """ %len(log_affine_params)
#
# groupwiseParameterMap1 =
# """
# (InitialTransformParametersFileName %s)
# (AutomaticParameterEstimation "true")
# (CheckNumberOfSamples "true")
# (DefaultPixelValue 0.0)
# (FinalBSplineInterpolationOrder 3)
# (FinalGridSpacingInPhysicalUnits 8)
# (FixedImagePyramid "FixedSmoothingImagePyramid")
# (FixedInternalImagePixelType "float")
# (GridSpacingSchedule 2.80322 1.9881 1.41 1)
# (ImageSampler "RandomCoordinate")
# (Interpolator "ReducedDimensionBSplineInterpolator")
# (MaximumNumberOfIterations 500)
# (MaximumNumberOfSamplingAttempts 8)
# (Metric "VarianceOverLastDimensionMetric")
# (MovingImagePyramid "MovingSmoothingImagePyramid")
# (MovingInternalImagePixelType "float")
# (NewSamplesEveryIteration "true")
# (NumberOfResolutions 6)
# (NumberOfSamplesForExactGradient 4096)
# (NumberOfSpatialSamples 16384)
# (Optimizer "AdaptiveStochasticGradientDescent")
# (Registration "MultiResolutionRegistration")
# (ResampleInterpolator "FinalReducedDimensionBSplineInterpolator")
# (Resampler "DefaultResampler")
# (ResultImageFormat "nii")
# (Transform "EulerStackTransform")
# (WriteIterationInfo "false")
# (WriteResultImage "true")
# """ %s(init_transform_path)
#
# groupwiseParameterMap2 =
# """
# (AutomaticParameterEstimation "true")
# (CheckNumberOfSamples "true")
# (DefaultPixelValue 0.0)
# (FinalBSplineInterpolationOrder 3)
# (FinalGridSpacingInPhysicalUnits 150)
# (FixedImagePyramid "FixedSmoothingImagePyramid")
# (FixedInternalImagePixelType "float")
# (GridSpacingSchedule 2.80322 1.9881 1.41 1)
# (ImageSampler "RandomCoordinate")
# (Interpolator "ReducedDimensionBSplineInterpolator")
# (MaximumNumberOfIterations 500)
# (MaximumNumberOfSamplingAttempts 8)
# (Metric "VarianceOverLastDimensionMetric")
# (MovingImagePyramid "MovingSmoothingImagePyramid")
# (MovingInternalImagePixelType "float")
# (NewSamplesEveryIteration "true")
# (NumberOfResolutions 3)
# (NumberOfSamplesForExactGradient 4096)
# (NumberOfSpatialSamples 16384)
# (Optimizer "AdaptiveStochasticGradientDescent")
# (Registration "MultiResolutionRegistration")
# (ResampleInterpolator "FinalReducedDimensionBSplineInterpolator")
# (Resampler "DefaultResampler")
# (ResultImageFormat "nii")
# (Transform "BSplineStackTransform")
# (WriteIterationInfo "false")
# (WriteResultImage "true")
# """
#     for i in range(3):
#         string = [initialTransform,groupwiseParameterMap1,groupwiseParameterMap2][i]
#         filepath = ([init_transform_path] + params_paths)[i]
#         outFile = open(filepath, 'w')
#         outFile.write(string)
#         outFile.close()
#
#     elastixImageFilter = sitk.ElastixImageFilter()
#     elastixImageFilter.SetFixedImage(image)
#     elastixImageFilter.SetMovingImage(image)
#     elastixImageFilter.SetParameterMap([groupwiseParameterMap1,groupwiseParameterMap2])
#     elastixImageFilter.Execute()
#
#     p1 = elastixImageFilter.GetTransformParameterMap()[0]
#     p2 = elastixImageFilter.GetTransformParameterMap()[1]
#
#     raw_params = []
#     for p in [p1,p2]:
#         for line in p.iteritems():
#             if line[0] == 'TransformParameters':
#                 tmp = np.array([float(i) for i in line[1]])
#                 tmp = tmp.reshape((len(ims),-1))
#                 raw_params.append(tmp)
#                 break
#
#     def apply_center_of_rotation(p,cr):
#         translation = p[-3:] - np.dot(p[:9].reshape((3, 3)), cr) + cr
#         resp = np.concatenate([p[:9], translation], 0)
#         return resp
#
#     # center of rotation
#     cr = (ims[0].origin + ims[0].spacing * np.array(ims[0].shape) / 2)[::-1]
#
#     final_params = []
#     for iim in range(len(ims)):
#
#         # process euler transform
#         raw_p = raw_params[0][iim]
#         tmp = transformations.euler_matrix(raw_p[0],raw_p[1],raw_p[2])
#         affine_params = np.zeros(12)
#         affine_params[:9] = tmp[:3,:3].flatten()
#         affine_params[-3:] = np.array([raw_p[3],raw_p[4],raw_p[5]])
#         # euler doesn't seem to need center of rotation application
#         # (or center of rotation is 0)
#         # affine_params = apply_center_of_rotation(affine_params,cr)
#         t0 = params_to_matrix(affine_params)
#
#         # process affine transform
#         raw_p = raw_params[1][iim]
#         affine_params = params_to_matrix(raw_p)
#         affine_params = expm(affine_params)
#         affine_params = matrix_to_params(affine_params)
#         # affine params needs center of rotation application (physical center of the image)
#         affine_params = apply_center_of_rotation(affine_params,cr)
#         t1 = params_to_matrix(affine_params)
#
#         t10 = matrix_to_params(np.dot(t1,t0))
#         t10 = params_invert_coordinates(t10)
#         final_params.append(t10)
#
#     # relate all final parameters to ref_view
#     final_ref_params = []
#     for iim in range(len(ims)):
#         t0 = invert_params(final_params[ref_view_index])
#         t1 = final_params[iim]
#         tmp = np.dot(params_to_matrix(t1),params_to_matrix(t0))
#         tmp = matrix_to_params(tmp)
#         final_ref_params.append(tmp)
#
#     # concatenate with input parameters
#     final_ref_params_concat = []
#     for iim in range(len(ims)):
#         t0 = final_ref_params[iim]
#         t1 = params0[iim]
#         tmp = np.dot(params_to_matrix(t1),params_to_matrix(t0))
#         tmp = matrix_to_params(tmp)
#         final_ref_params_concat.append(tmp)
#
#     return np.array(final_ref_params_concat)

@io_decorator
def register_groupwise_bspline(orig_ims, params0, ref_view_index=0, iso_reg_spacing_relative_to_input_z=1,
                           volume_mode='sample'):
    """
    - perform groupwise registration (to refine and globally optimise after pair registration)
    - first euler, then affine works well
    - works only with elastix v4.9
    - principle: use varianceoverlastdimension metric together with affinelogstacktransform
    - relate parameters back to reference view

    :param ims:
    :param params0:
    :param elastix_dir:
    :return:
    """

    reg_spacing = np.array([orig_ims[0].spacing[0] * iso_reg_spacing_relative_to_input_z] * 3).astype(np.float32)
    stack_properties = calc_stack_properties_from_views_and_params(orig_ims, params0, reg_spacing, volume_mode)

    ims = []
    for iview in range(len(orig_ims)):
        im = transform_stack_sitk(orig_ims[iview], params0[iview],
                                  out_origin=stack_properties['origin'],
                                  out_shape=stack_properties['size'],
                                  out_spacing=stack_properties['spacing'])
        ims.append(im)

    vectorOfImages = sitk.VectorOfImage()
    for iim, im in enumerate(ims):
        vectorOfImages.push_back(image_to_sitk(im))

    image = sitk.JoinSeries(vectorOfImages)

    # Register
    groupwiseParameterMap1 = sitk.GetDefaultParameterMap('groupwise')
    groupwiseParameterMap1['FixedInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['MovingInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['NumberOfResolutions'] = ['2']  # default is 4
    groupwiseParameterMap1['MaximumNumberOfIterations'] = ['250']  # default is 250
    # groupwiseParameterMap1['NumberOfResolutions'] = ['6']  # default is 4
    # groupwiseParameterMap1['MaximumNumberOfIterations'] = ['5000']  # default is 250
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    groupwiseParameterMap1['NumberOfSpatialSamples'] = ['8192']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['2048']  # default is 2048

    groupwiseParameterMap1['Transform'] = ['EulerStackTransform']  # default is 2048
    # groupwiseParameterMap1['Scales'] = [' '.join(['10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    groupwiseParameterMap2 = sitk.GetDefaultParameterMap('groupwise')
    groupwiseParameterMap2['FixedInternalImagePixelType'] = ['float']
    groupwiseParameterMap2['MovingInternalImagePixelType'] = ['float']
    # groupwiseParameterMap2['NumberOfResolutions'] = ['3']  # default is 4
    groupwiseParameterMap2['NumberOfResolutions'] = ['3']  # default is 4
    groupwiseParameterMap2['MaximumNumberOfIterations'] = ['250']  # default is 250
    groupwiseParameterMap2['NumberOfSpatialSamples'] = ['8192']  # default is 2048
    # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    groupwiseParameterMap2['Transform'] = ['BSplineStackTransform']  # default is 2048
    groupwiseParameterMap2['FinalGridSpacingInPhysicalUnits'] = ['300']  # default is 2048
    groupwiseParameterMap2['GridSpacingSchedule'] = ['2','1.5','1']  # default is 2048
    # groupwiseParameterMap2['Scales'] = [' '.join(['10000 10000 10000 10000 10000 10000 10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)
    elastixImageFilter.SetParameterMap([groupwiseParameterMap1, groupwiseParameterMap2])
    # elastixImageFilter.SetParameterMap([groupwiseParameterMap1])#, groupwiseParameterMap2])

    # elastixImageFilter.LogToConsoleOn()
    # elastixImageFilter.LogToFileOn()

    elastixImageFilter.Execute()

    # p1 = elastixImageFilter.GetTransformParameterMap()[0]
    # p2 = elastixImageFilter.GetTransformParameterMap()[1]

    ps = elastixImageFilter.GetTransformParameterMap()

    # turn into normal dicts for pickling
    dictps = []
    for p in ps:
        tmpdict = dict()
        for k,v in p.items():
            tmpdict[k] = v
        dictps.append(tmpdict)

    transformed_img = elastixImageFilter.GetResultImage()
    extract_filter = sitk.ExtractImageFilter()
    size = list(transformed_img.GetSize())
    size[3] = 0 # set t to 0 to collapse this dimension
    extract_filter.SetSize(size)
    imgs = []
    for i in range(len(vectorOfImages)):
        extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
        img = extract_filter.Execute(transformed_img)
        imgs.append(sitk_to_image(img))

    return dictps,imgs

# def transform_stacks_simpleelastix(ims,pmaps,stack_properties,t0s=None):
#     """
#     transform stacks using list of stacktransforms
#     :param stack:
#     :param pmaps:
#     :param out_shape:
#     :param out_spacing:
#     :param out_origin:
#     :return:
#     """
#
#     vectorOfImages = sitk.VectorOfImage()
#     for iim, im in enumerate(ims):
#         # vectorOfImages.push_back(image_to_sitk(im))
#         vectorOfImages.push_back(image_to_sitk(ims[0]))
#
#     image = sitk.JoinSeries(vectorOfImages)
#
#     cr = (ims[0].origin + ims[0].spacing * np.array(ims[0].shape) / 2)[::-1]
#
#     if t0s is not None:
#         pmaps = [params_to_stackpmap(t0s)] + list(pmaps)
#
#     for pmap in pmaps:
#         if 'CenterOfRotation' is not in pmap.keys():
#             pmap['CenterOfRotation'] = tuple([str(i) for i in cr])
#         pmap['Size']        = tuple([str(i) for i in list(stack_properties['size']   [::-1])+[1]])
#         pmap['Origin']      = tuple([str(i) for i in list(stack_properties['origin'] [::-1])+[0]])
#         pmap['Spacing']     = tuple([str(i) for i in list(stack_properties['spacing'][::-1])+[1]])
#         pmap['StackOrigin'] = tuple([str(i) for i in list(stack_properties['origin'] [::-1])+[0]])
#         pmap['StackSpacing']= tuple([str(i) for i in list(stack_properties['spacing'][::-1])+[1]])
#         pmap['NumberOfSubTransforms']= ("%s"%len(ims),)
#         # pmap = sitk.ParameterMap(pmap)
#
#
#     imaget = sitk.Transformix(image,pmaps)
#
#     size = list(imaget.GetSize())
#     size[3] = 0 # set t to 0 to collapse this dimension
#     extract_filter.SetSize(size)
#     imgs = []
#     for i in range(len(vectorOfImages)):
#         extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
#         img = extract_filter.Execute(imaget)
#         imgs.append(sitk_to_image(img))
#
#     return imgs

def transform_stack_simpleelastix(im,pmaps,stack_properties,t0=None):
    """
    transform stacks using list of stacktransforms
    :param stack:
    :param pmaps:
    :param out_shape:
    :param out_spacing:
    :param out_origin:
    :return:
    """

    # vectorOfImages = sitk.VectorOfImage()
    # for iim, im in enumerate(ims):
    #     # vectorOfImages.push_back(image_to_sitk(im))
    #     vectorOfImages.push_back(image_to_sitk(ims[0]))
    #
    # image = sitk.JoinSeries(vectorOfImages)

    # cr = (im.origin + im.spacing * np.array(im.shape) / 2)[::-1]

    if t0 is not None:
        pmaps = [params_to_pmap(t0)] + pmaps

    for pmap in pmaps:
        # if not 'CenterOfRotation' in pmap.keys():
        #     pmap['CenterOfRotation'] = tuple([str(i) for i in cr])
        # pmap['Size']        = tuple([str(i) for i in list(stack_properties['size']   [::-1])+[1]])
        # pmap['Origin']      = tuple([str(i) for i in list(stack_properties['origin'] [::-1])+[0]])
        # pmap['Spacing']     = tuple([str(i) for i in list(stack_properties['spacing'][::-1])+[1]])
        # pmap['StackOrigin'] = tuple([str(i) for i in list(stack_properties['origin'] [::-1])+[0]])
        # pmap['StackSpacing']= tuple([str(i) for i in list(stack_properties['spacing'][::-1])+[1]])
        # pmap['NumberOfSubTransforms']= ("%s"%len(ims),)
        # pmap = sitk.ParameterMap(pmap)
        pmap['Size']        = tuple([str(i) for i in list(stack_properties['size']   [::-1])])
        pmap['Origin']      = tuple([str(i) for i in list(stack_properties['origin'] [::-1])])
        pmap['Spacing']     = tuple([str(i) for i in list(stack_properties['spacing'][::-1])])

        pmap['UseDirectionCosines']     = ("true",)
        pmap['UseBinaryFormatForTransformationParameters']     = ("false",)
        pmap['ResampleInterpolator']     = ("FinalBSplineInterpolator",)

    image = image_to_sitk(im)
    # order of pmaps for transformix seems to be:
    # first transformation closest to target space, then those close to moving space
    # example: bspline, affine
    # example: final_params, final_params0
    imaget = sitk.Transformix(image,pmaps[::-1])
    imaget = sitk_to_image(imaget)

    # size = list(imaget.GetSize())
    # size[3] = 0 # set t to 0 to collapse this dimension
    # extract_filter.SetSize(size)
    # imgs = []
    # for i in range(len(vectorOfImages)):
    #     extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
    #     img = extract_filter.Execute(imaget)
    #     imgs.append(sitk_to_image(img))

    return imaget

from scipy.linalg import expm
def stackpmaps_to_pmaps(stackpmaps,n_subtransforms=4):

    n_subtransforms = int(n_subtransforms)

    def apply_center_of_rotation(p, cr):
        translation = p[-3:] - np.dot(p[:9].reshape((3, 3)), cr) + cr
        resp = np.concatenate([p[:9], translation], 0)
        return resp

    pmaps = []
    for st in range(n_subtransforms):
        stpmaps = []
        for sm in range(len(stackpmaps)):
            n_params = int(len(stackpmaps[sm]['TransformParameters']) / n_subtransforms)
            pmap = dict()
            pmap['FinalBSplineInterpolationOrder'] = ("1",)
            pmap['FixedImageDimension'] = ("3",)
            pmap['MovingImageDimension'] = ("3",)
            pmap['FixedInternalImagePixelType'] = ("float",)
            pmap['MovingInternalImagePixelType'] = ("float",)
            pmap['ResultImagePixelType'] = ("float",)
            pmap['HowToCombineTransforms'] = ("Compose",)
            pmap['Index'] = ("0", "0", "0")
            pmap['Direction'] = tuple([str(i) for i in matrix_to_params(np.eye(4))[:9]])
            pmap['InitialTransformParametersFileName'] = ("NoInitialTransform",)
            pmap['DefaultPixelValue'] = ("0",)
            pmap['Resampler'] = ("DefaultResampler",)
            # pmap['ResampleInterpolator'] = ("FinalReducedDimensionBSplineInterpolator",)
            pmap['CompressResultImage'] = ("false",)

            size = np.array([float(i) for i in stackpmaps[sm]['Size'][:3]])
            origin = np.array([float(i) for i in stackpmaps[sm]['Origin'][:3]])
            spacing = np.array([float(i) for i in stackpmaps[sm]['Spacing'][:3]])

            if stackpmaps[sm]['Transform'][0] == 'EulerStackTransform':
                pmap['Transform'] = ("EulerTransform",)
                pmap['TransformParameters'] = stackpmaps[sm]['TransformParameters'][st * n_params:(st + 1) * n_params]
            elif stackpmaps[sm]['Transform'][0] == 'AffineLogStackTransform':
                pmap['Transform'] = ("AffineTransform",)
                tmpparams = stackpmaps[sm]['TransformParameters'][st * n_params:(st + 1) * n_params]
                tmpparams = matrix_to_params(expm(params_to_matrix(tmpparams)))
                # affine params needs center of rotation application (physical center of the image)
                cr = (origin + spacing * size / 2)
                tmpparams = apply_center_of_rotation(tmpparams, cr)
                pmap['TransformParameters'] = tmpparams
            elif stackpmaps[sm]['Transform'][0] == 'BSplineStackTransform':
                pmap['Transform'] = ("BSplineTransform",)
                pmap['TransformParameters'] = stackpmaps[sm]['TransformParameters'][st * n_params:(st + 1) * n_params]
                pmap['BSplineTransformSplineOrder'] = ("3",)
                copy_keys = ['GridDirection','GridIndex','GridOrigin','GridSize','GridSpacing']
                for copy_key in copy_keys:
                    pmap[copy_key] = stackpmaps[sm][copy_key]
            # BSplineTransformSplineOrder('3', )
            # CompressResultImage('false', )
            # DefaultPixelValue('0.0', )
            # Direction('1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1')
            # FixedImageDimension('4', )
            # FixedInternalImagePixelType('float', )
            # GridDirection('1', '0', '0', '0', '1', '0', '0', '0', '1')
            # GridIndex('0', '0', '0')
            # GridOrigin('-511.5', '-630.5', '-649')
            # GridSize('4', '6', '5')
            # GridSpacing('500', '500', '500')
            # HowToCombineTransforms('Compose', )
            # Index('0', '0', '0', '0')
            # InitialTransformParametersFileName('NoInitialTransform', )
            # MovingImageDimension('4', )
            # MovingInternalImagePixelType('float', )
            # NumberOfParameters('1440', )
            # NumberOfSubTransforms('4', )
            # Origin('0', '0', '0', '0')
            # ResampleInterpolator('FinalReducedDimensionBSplineInterpolator', )
            # Resampler('DefaultResampler', )
            # ResultImageFormat('nii', )
            # ResultImagePixelType('float', )
            # Size('160', '414', '235', '4')
            # Spacing('3', '3', '3', '1')
            # StackOrigin('0', )
            # StackSpacing('1', )
            # Transform('BSplineStackTransform', )

            else:
                print(stackpmaps[sm]['Transform'])
                raise(Exception())

            pmap['NumberOfParameters'] = ("%s" % len(pmap['TransformParameters']),)
            # pmap['CenterOfRotation'] = tuple([str(i) for i in origin])
            # pmap['CenterOfRotation'] = tuple([str(i) for i in [0, 0, 0]])
            pmap['CenterOfRotationPoint'] = tuple([str(i) for i in [0, 0, 0]])


            stpmaps.append(pmap)
        pmaps.append(stpmaps)

    return pmaps

# from scipy.linalg import logm
# def params_to_stackpmap(params):
#
#     logparams = []
#     for p in params:
#         logparams += list(matrix_to_params(logm(params_to_matrix(p))))
#
#     pmap = dict()
#     pmap['FixedImageDimension'] = ("4",)
#     pmap['MovingImageDimension'] = ("4",)
#     pmap['FixedInternalImagePixelType'] = ("float",)
#     pmap['MovingInternalImagePixelType'] = ("float",)
#     pmap['ResultImagePixelType'] = ("float",)
#     pmap['HowToCombineTransforms'] = ("Compose",)
#     pmap['Index'] = ("0","0","0","0")
#     pmap['InitialTransformParametersFileName'] = ("NoInitialTransform",)
#     pmap['DefaultPixelValue'] = ("0",)
#     pmap['Resampler'] = ("DefaultResampler",)
#     pmap['ResampleInterpolator'] = ("FinalReducedDimensionBSplineInterpolator",)
#     pmap['CompressResultImage'] = ("false",)
#
#     pmap['Transform'] = ("AffineLogStackTransform",)
#     pmap['NumberOfParameters'] = ("%s"%len(logparams),)
#     pmap['TransformParameters'] = tuple([str(i) for i in logparams])
#     pmap['CenterOfRotation'] = tuple([str(i) for i in [0,0,0]])
#
#     return pmap

from scipy.linalg import logm
def params_to_pmap(params):

    # params[-3:] = np.dot(params[:9].reshape((3,3)),

    params = params_invert_coordinates(params)

    pmap = dict()
    # pmap['ResampleInterpolator'] = ("BSplineResampleInterpolatorFloat",)
    pmap['FinalBSplineInterpolationOrder'] = ("1",)
    pmap['Index'] = ("0", "0", "0")
    pmap['Direction'] = tuple([str(i) for i in matrix_to_params(np.eye(4))[:9]])
    pmap['FixedImageDimension'] = ("3",)
    pmap['MovingImageDimension'] = ("3",)
    pmap['FixedInternalImagePixelType'] = ("float",)
    pmap['MovingInternalImagePixelType'] = ("float",)
    pmap['ResultImagePixelType'] = ("float",)
    pmap['HowToCombineTransforms'] = ("Compose",)
    # pmap['Index'] = ("0", "0", "0", "0")
    pmap['InitialTransformParametersFileName'] = ("NoInitialTransform",)
    pmap['DefaultPixelValue'] = ("0",)
    pmap['Resampler'] = ("DefaultResampler",)
    # pmap['ResampleInterpolator'] = ("FinalReducedDimensionBSplineInterpolator",)
    pmap['CompressResultImage'] = ("false",)

    pmap['Transform'] = ("AffineTransform",)
    pmap['NumberOfParameters'] = ("%s" % len(params),)
    pmap['TransformParameters'] = tuple([str(i) for i in params])
    # pmap['CenterOfRotation'] = tuple([str(i) for i in [0, 0, 0]])
    pmap['CenterOfRotationPoint'] = tuple([str(i) for i in [0, 0, 0]])

    return pmap

# # compute mean image
# transformed_img = elastixImageFilter.GetResultImage()
# extract_filter = sitk.ExtractImageFilter()
# size = list(transformed_img.GetSize())
# size[3] = 0 # set t to 0 to collapse this dimension
# extract_filter.SetSize(size)
# imgs = []
# for i in range(len(vectorOfImages)):
#     extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
#     img = extract_filter.Execute(transformed_img)
#     imgs.append(sitk.GetArrayFromImage(img))
#

from scipy.linalg import expm, logm
# import transformations
@io_decorator
def register_groupwise(orig_ims, params0, ref_view_index=0, iso_reg_spacing_relative_to_input_z=1,
                           volume_mode='sample'):
    """
    - perform groupwise registration (to refine and globally optimise after pair registration)
    - first euler, then affine works well
    - works only with elastix v4.9
    - principle: use varianceoverlastdimension metric together with affinelogstacktransform
    - relate parameters back to reference view

    :param ims:
    :param params0:
    :param elastix_dir:
    :return:
    """

    reg_spacing = np.array([orig_ims[0].spacing[0] * iso_reg_spacing_relative_to_input_z] * 3).astype(np.float32)
    stack_properties = calc_stack_properties_from_views_and_params(orig_ims, params0, reg_spacing, volume_mode)

    ims = []
    for iview in range(len(orig_ims)):
        im = transform_stack_sitk(orig_ims[iview], params0[iview],
                                  out_origin=stack_properties['origin'],
                                  out_shape=stack_properties['size'],
                                  out_spacing=stack_properties['spacing'])

        imc = clahe(im, 10, clip_limit=0.02).astype(np.float32)
        imc = ImageArray(imc,origin=im.origin,spacing=im.spacing)
        ims.append(imc)

        # ims.append(im)

    vectorOfImages = sitk.VectorOfImage()
    for iim, im in enumerate(ims):
        vectorOfImages.push_back(image_to_sitk(im))

    image = sitk.JoinSeries(vectorOfImages)

    # Register
    groupwiseParameterMap1 = sitk.GetDefaultParameterMap('groupwise')
    groupwiseParameterMap1['FixedInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['MovingInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['NumberOfResolutions'] = ['6']  # default is 4
    # groupwiseParameterMap1['MaximumNumberOfIterations'] = ['5000']  # default is 250
    groupwiseParameterMap1['MaximumNumberOfIterations'] = ['500']  # default is 250
    groupwiseParameterMap1['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['8192']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['2048']  # default is 2048

    groupwiseParameterMap1['Transform'] = ['EulerStackTransform']  # default is 2048

    groupwiseParameterMap1['Optimizer'] = ['QuasiNewtonLBFGS']  # default is 2048
    groupwiseParameterMap1['NumberOfSpatialSamples'] = ['%s' % 100 ** 3]  # default is 2048
    groupwiseParameterMap1['GradientMagnitudeTolerance'] = ['1e-8']  # default is 2048
    # groupwiseParameterMap1['ImageSampler'] = ['Full']  # default is 2048
    groupwiseParameterMap1['Interpolator'] = ['BSplineInterpolatorFloat']

    # (Interpolator "BSplineInterpolatorFloat")
    # groupwiseParameterMap1['Scales'] = [' '.join(['10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    # groupwiseParameterMap2 = sitk.GetDefaultParameterMap('groupwise')
    # groupwiseParameterMap2['FixedInternalImagePixelType'] = ['float']
    # groupwiseParameterMap2['MovingInternalImagePixelType'] = ['float']
    # groupwiseParameterMap2['NumberOfResolutions'] = ['3']  # default is 4
    # groupwiseParameterMap2['MaximumNumberOfIterations'] = ['500']  # default is 250
    # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    # groupwiseParameterMap2['Transform'] = ['AffineLogStackTransform']  # default is 2048
    # # groupwiseParameterMap2['Scales'] = [' '.join(['10000 10000 10000 10000 10000 10000 10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)
    elastixImageFilter.SetParameterMap([groupwiseParameterMap1])#, groupwiseParameterMap2])
    elastixImageFilter.Execute()

    p1 = elastixImageFilter.GetTransformParameterMap()[0]
    # p2 = elastixImageFilter.GetTransformParameterMap()[1]

    raw_params = []
    for p in [p1]:
        for line in p.iteritems():
            if line[0] == 'TransformParameters':
                tmp = np.array([float(i) for i in line[1]])
                tmp = tmp.reshape((len(ims), -1))
                raw_params.append(tmp)
                break

    def apply_center_of_rotation(p, cr):
        translation = p[-3:] - np.dot(p[:9].reshape((3, 3)), cr) + cr
        resp = np.concatenate([p[:9], translation], 0)
        return resp

    # center of rotation
    cr = (ims[0].origin + ims[0].spacing * np.array(ims[0].shape) / 2)[::-1]

    final_params = []
    for iim in range(len(ims)):
        # process euler transform
        raw_p = raw_params[0][iim]
        tmp = mv_utils.euler_matrix(raw_p[0], raw_p[1], raw_p[2])
        affine_params = np.zeros(12)
        affine_params[:9] = tmp[:3, :3].flatten()
        affine_params[-3:] = np.array([raw_p[3], raw_p[4], raw_p[5]])
        # euler doesn't seem to need center of rotation application
        # (or center of rotation is 0)
        # affine_params = apply_center_of_rotation(affine_params,cr)
        t0 = params_to_matrix(affine_params)

        # process affine transform
        # raw_p = raw_params[1][iim]
        # affine_params = params_to_matrix(raw_p)
        # affine_params = expm(affine_params)
        # affine_params = matrix_to_params(affine_params)
        # # affine params needs center of rotation application (physical center of the image)
        # affine_params = apply_center_of_rotation(affine_params, cr)
        # t1 = params_to_matrix(affine_params)
        #
        # t10 = matrix_to_params(np.dot(t1, t0))
        # t10 = params_invert_coordinates(t10)
        t0 = matrix_to_params(t0)
        t0 = params_invert_coordinates(t0)
        final_params.append(t0)

    # relate all final parameters to ref_view
    final_ref_params = []
    for iim in range(len(ims)):
        t0 = invert_params(final_params[ref_view_index])
        t1 = final_params[iim]
        tmp = np.dot(params_to_matrix(t1), params_to_matrix(t0))
        tmp = matrix_to_params(tmp)
        final_ref_params.append(tmp)

    # concatenate with input parameters
    final_ref_params_concat = []
    for iim in range(len(ims)):
        t0 = final_ref_params[iim]
        t1 = params0[iim]
        tmp = np.dot(params_to_matrix(t1), params_to_matrix(t0))
        tmp = matrix_to_params(tmp)
        final_ref_params_concat.append(tmp)

    return np.array(final_ref_params_concat)

from scipy.linalg import expm, logm
# import transformations
@io_decorator
def register_groupwise_euler_and_affine(orig_ims, params0, ref_view_index=0, iso_reg_spacing_relative_to_input_z=1,
                           volume_mode='sample'):
    """
    - perform groupwise registration (to refine and globally optimise after pair registration)
    - first euler, then affine works well
    - works only with elastix v4.9
    - principle: use varianceoverlastdimension metric together with affinelogstacktransform
    - relate parameters back to reference view

    :param ims:
    :param params0:
    :param elastix_dir:
    :return:
    """

    reg_spacing = np.array([orig_ims[0].spacing[0] * iso_reg_spacing_relative_to_input_z] * 3).astype(np.float32)
    stack_properties = calc_stack_properties_from_views_and_params(orig_ims, params0, reg_spacing, volume_mode)

    ims = []
    for iview in range(len(orig_ims)):

        im = transform_stack_sitk(orig_ims[iview], params0[iview],
                                  out_origin=stack_properties['origin'],
                                  out_shape=stack_properties['size'],
                                  out_spacing=stack_properties['spacing'])

        imc = clahe(im, 10, clip_limit=0.02).astype(np.float32)
        imc = ImageArray(imc,origin=im.origin,spacing=im.spacing)
        ims.append(imc)

        # ims.append(im)

    vectorOfImages = sitk.VectorOfImage()
    for iim, im in enumerate(ims):
        vectorOfImages.push_back(image_to_sitk(im))

    image = sitk.JoinSeries(vectorOfImages)

    # Register
    groupwiseParameterMap1 = sitk.GetDefaultParameterMap('groupwise')
    groupwiseParameterMap1['FixedInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['MovingInternalImagePixelType'] = ['float']
    groupwiseParameterMap1['NumberOfResolutions'] = ['6']  # default is 4
    # groupwiseParameterMap1['MaximumNumberOfIterations'] = ['5000']  # default is 250
    groupwiseParameterMap1['MaximumNumberOfIterations'] = ['500']  # default is 250
    groupwiseParameterMap1['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['8192']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['2048']  # default is 2048

    groupwiseParameterMap1['Transform'] = ['EulerStackTransform']  # default is 2048


    # groupwiseParameterMap1['Optimizer'] = ['QuasiNewtonLBFGS']  # default is 2048
    # groupwiseParameterMap1['NumberOfSpatialSamples'] = ['%s' %100**3]  # default is 2048
    # groupwiseParameterMap1['GradientMagnitudeTolerance'] = ['1e-8']  # default is 2048
    # # groupwiseParameterMap1['ImageSampler'] = ['Full']  # default is 2048
    # groupwiseParameterMap1['Interpolator'] = ['BSplineInterpolatorFloat']
    # groupwiseParameterMap1['Scales'] = [' '.join(['10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    groupwiseParameterMap2 = sitk.GetDefaultParameterMap('groupwise')
    groupwiseParameterMap2['FixedInternalImagePixelType'] = ['float']
    groupwiseParameterMap2['MovingInternalImagePixelType'] = ['float']
    groupwiseParameterMap2['NumberOfResolutions'] = ['3']  # default is 4
    groupwiseParameterMap2['MaximumNumberOfIterations'] = ['500']  # default is 250
    groupwiseParameterMap2['NumberOfSpatialSamples'] = ['16384']  # default is 2048
    # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['32768']  # default is 2048
    groupwiseParameterMap2['Transform'] = ['AffineLogStackTransform']  # default is 2048

    # groupwiseParameterMap2['Optimizer'] = ['QuasiNewtonLBFGS']  # default is 2048
    # groupwiseParameterMap2['NumberOfSpatialSamples'] = ['%s' %100**3]  # default is 2048
    # groupwiseParameterMap2['GradientMagnitudeTolerance'] = ['1e-8']  # default is 2048
    # # groupwiseParameterMap2['ImageSampler'] = ['Full']  # default is 2048
    # groupwiseParameterMap2['Interpolator'] = ['BSplineInterpolatorFloat']
    # groupwiseParameterMap2['Scales'] = [' '.join(['10000 10000 10000 10000 10000 10000 10000 10000 10000 1 1 1']*len(ims))]  # default is 2048

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)
    elastixImageFilter.SetParameterMap([groupwiseParameterMap1, groupwiseParameterMap2])
    elastixImageFilter.Execute()

    p1 = elastixImageFilter.GetTransformParameterMap()[0]
    p2 = elastixImageFilter.GetTransformParameterMap()[1]

    raw_params = []
    for p in [p1, p2]:
        for line in p.iteritems():
            if line[0] == 'TransformParameters':
                tmp = np.array([float(i) for i in line[1]])
                tmp = tmp.reshape((len(ims), -1))
                raw_params.append(tmp)
                break

    def apply_center_of_rotation(p, cr):
        translation = p[-3:] - np.dot(p[:9].reshape((3, 3)), cr) + cr
        resp = np.concatenate([p[:9], translation], 0)
        return resp

    # center of rotation
    cr = (ims[0].origin + ims[0].spacing * np.array(ims[0].shape) / 2)[::-1]

    final_params = []
    for iim in range(len(ims)):
        # process euler transform
        raw_p = raw_params[0][iim]
        tmp = mv_utils.euler_matrix(raw_p[0], raw_p[1], raw_p[2])
        affine_params = np.zeros(12)
        affine_params[:9] = tmp[:3, :3].flatten()
        affine_params[-3:] = np.array([raw_p[3], raw_p[4], raw_p[5]])
        # euler doesn't seem to need center of rotation application
        # (or center of rotation is 0)
        # affine_params = apply_center_of_rotation(affine_params,cr)
        t0 = params_to_matrix(affine_params)

        # process affine transform
        raw_p = raw_params[1][iim]
        affine_params = params_to_matrix(raw_p)
        affine_params = expm(affine_params)
        affine_params = matrix_to_params(affine_params)
        # affine params needs center of rotation application (physical center of the image)
        affine_params = apply_center_of_rotation(affine_params, cr)
        t1 = params_to_matrix(affine_params)

        t10 = matrix_to_params(np.dot(t1, t0))
        t10 = params_invert_coordinates(t10)
        final_params.append(t10)

    # relate all final parameters to ref_view
    final_ref_params = []
    for iim in range(len(ims)):
        t0 = invert_params(final_params[ref_view_index])
        t1 = final_params[iim]
        tmp = np.dot(params_to_matrix(t1), params_to_matrix(t0))
        tmp = matrix_to_params(tmp)
        final_ref_params.append(tmp)

    # concatenate with input parameters
    final_ref_params_concat = []
    for iim in range(len(ims)):
        t0 = final_ref_params[iim]
        t1 = params0[iim]
        tmp = np.dot(params_to_matrix(t1), params_to_matrix(t0))
        tmp = matrix_to_params(tmp)
        final_ref_params_concat.append(tmp)

    return np.array(final_ref_params_concat)

    # # compute mean image
    # transformed_img = elastixImageFilter.GetResultImage()
    # extract_filter = sitk.ExtractImageFilter()
    # size = list(transformed_img.GetSize())
    # size[3] = 0 # set t to 0 to collapse this dimension
    # extract_filter.SetSize(size)
    # imgs = []
    # for i in range(len(vectorOfImages)):
    #     extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
    #     img = extract_filter.Execute(transformed_img)
    #     imgs.append(sitk.GetArrayFromImage(img))
    #
    #
    # img_mean = np.mean(imgs, axis=0)
    # img_fused = sitk.GetImageFromArray(img_mean)
    # sitk.WriteImage(img_fused,'/data/malbert/regtest/mv_transf_view_mean.imagear.mhd')
    #
    # for i in range(len(vectorOfImages)):
    #     sitk.WriteImage(sitk.GetImageFromArray(imgs[i]), '/data/malbert/regtest/mv_transf_view_after_groupwise_%02d.imagear.mhd' %i)
    #
    # params = 0
    # return params

# @io_decorator
# def get_final_params_global(ref_view,pairs,params,time_alignment_params=None):
#     """
#     time_alignment_params: single params from longitudinal registration to be concatenated with view params
#     """
#
#     import networkx
#     g = networkx.DiGraph()
#     for ipair,pair in enumerate(pairs):
#         # g.add_edge(pair[0],pair[1],{'p': params[ipair]})
#         g.add_edge(pair[0],pair[1], p = params[ipair]) # after update 201809 networkx seems to have changed
#
#     all_views = np.unique(np.array(pairs).flatten())
#     # views_to_transform = np.sort(np.array(list(set(all_views).difference(set([ref_view])))))
#
#     # calculate params mapping coords in common space to those in the views
#     params_common_space_to_view = dict()
#     for iview0,view0 in enumerate(all_views):
#
#         params_to_other_views = []
#         for iview1, view1 in enumerate(all_views):
#
#             if view0 == view1: params_to_other_views.append(matrix_to_params(np.eye(4)))
#             else:
#                 paths = networkx.all_shortest_paths(g,view0,view1)
#                 paths_params = []
#                 for ipath,path in enumerate(paths):
#                     path_pairs = [[path[i],path[i+1]] for i in range(len(path)-1)]
#                     print(path_pairs)
#                     path_params = np.eye(4)
#                     for edge in path_pairs:
#                         tmp_params = params_to_matrix(g.get_edge_data(edge[0],edge[1])['p'])
#                         path_params = np.dot(tmp_params,path_params)
#                         print(path_params)
#                     paths_params.append(matrix_to_params(path_params))
#
#                 # params_pairwise['%03d_%03d' %(view0,view1)] = np.mean(paths_params,0)
#                 params_to_other_views.append(np.mean(paths_params,0))
#
#         mean_to_other_views = np.mean(params_to_other_views,0)
#
#         if view0 == ref_view:
#             params_ref_to_common = mean_to_other_views
#             print(params_ref_to_common)
#
#         mean_to_other_views_inv = invert_params(mean_to_other_views)
#         params_common_space_to_view[view0] = mean_to_other_views_inv
#
#     # compose params such that params map coords in ref_view to coords in the views
#
#     final_params = []
#     for iview, view in enumerate(all_views):
#         if view == ref_view: final_view_params = matrix_to_params(np.eye(4))
#         else:
#             final_view_params = np.dot(params_to_matrix(params_ref_to_common), params_to_matrix(params_common_space_to_view[view]))
#             final_view_params = matrix_to_params(final_view_params)
#
#         # concatenate with time alignment if given
#         if time_alignment_params is not None:
#             final_view_params = concatenate_view_and_time_params(time_alignment_params,final_view_params)
#
#         final_params.append(final_view_params)
#
#     return np.array(final_params)

# @io_decorator
# def get_final_params_global(ref_view,pairs,params,time_alignment_params=None):
#     """
#     time_alignment_params: single params from longitudinal registration to be concatenated with view params
#     """
#
#     import networkx
#     g = networkx.DiGraph()
#     for ipair,pair in enumerate(pairs):
#         g.add_edge(pair[0],pair[1], p = params[ipair]) # after update 201809 networkx seems to have changed
#         g.add_edge(pair[1],pair[0], p = invert_params(params[ipair])) # after update 201809 networkx seems to have changed
#
#     all_views = np.unique(np.array(pairs).flatten())
#     # views_to_transform = np.sort(np.array(list(set(all_views).difference(set([ref_view])))))
#
#     # calculate params mapping coords in common space to those in the views
#     params_common_space_to_view = dict()
#     params_pairwise = dict()
#     for iview0,view0 in enumerate(all_views):
#         for iview1, view1 in enumerate(all_views):
#
#             if view0 == view1: pairwise = matrix_to_params(np.eye(4))
#             else:
#                 paths = networkx.all_shortest_paths(g,view0,view1)
#                 paths_params = []
#                 for ipath,path in enumerate(paths):
#                     path_pairs = [[path[i],path[i+1]] for i in range(len(path)-1)]
#                     print(path_pairs)
#                     path_params = np.eye(4)
#                     for edge in path_pairs:
#                         tmp_params = params_to_matrix(g.get_edge_data(edge[0],edge[1])['p'])
#                         path_params = np.dot(tmp_params,path_params)
#                         print(path_params)
#                     paths_params.append(matrix_to_params(path_params))
#
#                 # params_pairwise['%03d_%03d' %(view0,view1)] = np.mean(paths_params,0)
#                 pairwise = np.mean(paths_params,0)
#
#             params_pairwise[(view0,view1)] = pairwise
#
#     # calc params to ref as mean over every intermediate
#     final_params = []
#     for iview, view in enumerate(all_views):
#         intermediates = []
#         for iview_interm, view_interm in enumerate(all_views):
#             ref_to_intermediate  = params_to_matrix(params_pairwise[(ref_view   ,view_interm)])
#             intermediate_to_view = params_to_matrix(params_pairwise[(view_interm,view       )])
#             ref_to_view = matrix_to_params(np.dot(intermediate_to_view, ref_to_intermediate))
#             intermediates.append(ref_to_view)
#         final_view_params = np.mean(intermediates,0)
#
#         # concatenate with time alignment if given
#         if time_alignment_params is not None:
#             final_view_params = concatenate_view_and_time_params(time_alignment_params,final_view_params)
#
#         final_params.append(final_view_params)
#
#     return np.array(final_params)

# @io_decorator
# def get_params_pairwise(ref_view,pairs,params,time_alignment_params=None):
#     """
#     time_alignment_params: single params from longitudinal registration to be concatenated with view params
#     """
#
#     import networkx
#     g = networkx.DiGraph()
#     for ipair,pair in enumerate(pairs):
#         g.add_edge(pair[0],pair[1], p = params[ipair]) # after update 201809 networkx seems to have changed
#         g.add_edge(pair[1],pair[0], p = invert_params(params[ipair])) # after update 201809 networkx seems to have changed
#
#     all_views = np.unique(np.array(pairs).flatten())
#     # views_to_transform = np.sort(np.array(list(set(all_views).difference(set([ref_view])))))
#
#     # calculate params mapping coords in common space to those in the views
#     params_common_space_to_view = dict()
#     params_pairwise = dict()
#     for iview0,view0 in enumerate(all_views):
#         for iview1, view1 in enumerate(all_views):
#
#             if view0 == view1: pairwise = matrix_to_params(np.eye(4))
#             else:
#                 paths = networkx.all_shortest_paths(g,view0,view1)
#                 paths_params = []
#                 for ipath,path in enumerate(paths):
#                     path_pairs = [[path[i],path[i+1]] for i in range(len(path)-1)]
#                     print(path_pairs)
#                     path_params = np.eye(4)
#                     for edge in path_pairs:
#                         tmp_params = params_to_matrix(g.get_edge_data(edge[0],edge[1])['p'])
#                         path_params = np.dot(tmp_params,path_params)
#                         print(path_params)
#                     paths_params.append(matrix_to_params(path_params))
#
#                 # params_pairwise['%03d_%03d' %(view0,view1)] = np.mean(paths_params,0)
#                 pairwise = np.mean(paths_params,0)
#
#             params_pairwise[(view0,view1)] = pairwise
#     return params_pairwise

def get_union_volume(ims, params):
    """
    back project first planes in every view to get maximum volume
    """

    generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    vertices = np.zeros((len(ims)*len(generic_vertices),3))
    for iim,im in enumerate(ims):
        tmp_vertices = generic_vertices * np.array(im.shape) * im.spacing + im.origin
        inv_params = params_to_matrix(invert_params(params[iim]))
        tmp_vertices_transformed = np.dot(inv_params[:3,:3], tmp_vertices.T).T + inv_params[:3,3]
        vertices[iim*len(generic_vertices):(iim+1)*len(generic_vertices)] = tmp_vertices_transformed

    # res = dict()
    # res['lower'] = np.min(vertices,0)
    # res['upper'] = np.max(vertices,0)

    lower = np.min(vertices,0)
    upper = np.max(vertices,0)

    return lower,upper

def get_intersection_volume(ims, params):
    """
    back project first planes in every view to get maximum volume
    """

    # generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    vertices = np.zeros((len(ims),len(generic_vertices),3))
    for iim,im in enumerate(ims):
        tmp_vertices = generic_vertices * np.array(im.shape) * im.spacing + im.origin
        inv_params = params_to_matrix(invert_params(params[iim]))
        tmp_vertices_transformed = np.dot(inv_params[:3,:3], tmp_vertices.T).T + inv_params[:3,3]
        vertices[iim,:] = tmp_vertices_transformed

    # res = dict()
    # res['lower'] = np.min(vertices,0)
    # res['upper'] = np.max(vertices,0)

    lower = np.max(np.min(vertices,1),0)
    upper = np.min(np.max(vertices,1),0)

    return lower,upper

def get_sample_volume(ims, params):
    """
    back project first planes in every view to get maximum volume
    """

    # generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    generic_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0.5]])
    vertices = np.zeros((len(ims)*len(generic_vertices),3))
    for iim,im in enumerate(ims):
        tmp_vertices = generic_vertices * np.array(im.shape) * im.spacing + im.origin
        inv_params = params_to_matrix(invert_params(params[iim]))
        tmp_vertices_transformed = np.dot(inv_params[:3,:3], tmp_vertices.T).T + inv_params[:3,3]
        vertices[iim*len(generic_vertices):(iim+1)*len(generic_vertices)] = tmp_vertices_transformed

    # res = dict()
    # res['lower'] = np.min(vertices,0)
    # res['upper'] = np.max(vertices,0)

    lower = np.min(vertices,0)
    upper = np.max(vertices,0)

    return lower,upper

def calc_stack_properties_from_volume(volume,spacing):

    """
    :param volume: lower and upper edge of final volume (e.g. [edgeLow,edgeHigh] as calculated by calc_final_stack_cube)
    :param spacing: final spacing
    :return: dictionary containing size, origin and spacing of final stack
    """

    origin                      = volume[0]
    size                        = np.ceil((volume[1]-volume[0]) / spacing).astype(np.uint16)

    properties_dict = dict()
    properties_dict['size']     = size
    properties_dict['spacing']  = spacing
    properties_dict['origin']   = origin

    return properties_dict

# @io_decorator
# def get_weights_simple(
#                     views,
#                     params,
#                     stack_properties,
#                     ):
#     """
#     sigmoid on borders
#     """
#
#     w_stack_properties = stack_properties.copy()
#     minspacing = 3.
#     changed_stack_properties = False
#     if w_stack_properties['spacing'][0] < minspacing:
#         changed_stack_properties = True
#         print('using downsampled images for calculating simple weights..')
#         w_stack_properties['spacing'] = np.array([minspacing]*3)
#         w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']
#
#     ws = []
#     for iview,view in enumerate(views):
#
#         tmporigin = views[iview].origin+views[iview].spacing/2.
#         badplanes = int(0/views[iview].spacing[0]) # in microns
#         tmporigin[0]+= views[iview].spacing[0]*badplanes
#         # print('WATCH OUT! simple_weights: disregarding %s bad planes at the end of the stack' % badplanes)
#         # sig =
#
#         # x,y,z = np.mgrid[:sig.shape[0],:sig.shape[1],:sig.shape[2]]
#         # dists = np.ones(sig.shape)*np.max(sig.shape)
#         # for d in range(3):
#         #     ddists =
#         #     dists = np.min([dists,ddists])
#         reducedview = view[badplanes:-1,:-1,:-1]
#         tmp_view = ImageArray(reducedview+1,spacing=views[iview].spacing,origin=tmporigin,rotation=views[iview].rotation)
#         # tmp_view = ImageArray(view[:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
#         mask = transform_stack_sitk(tmp_view,params[iview],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'],
#                                 interp='nearest')
#
#         def sigmoid(x,borderwidth):
#             x0 = float(borderwidth)/2.
#             a = 12./borderwidth
#             return 1/(1+np.exp(-a*(x-x0)))
#
#         r = 0.1 # relative border width
#         sig = np.ones(reducedview.shape,dtype=np.float32)
#         for d in range(3):
#             borderwidth = int(r * sig.shape[d])
#             print(borderwidth)
#             slices = [slice(0, sig.shape[i]) for i in range(3)]
#             for bx in range(borderwidth):
#                 slices[d] = slice(bx, bx + 1)
#                 sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)
#
#             # don't blend best part of the image (assuming that is true for high zs)
#             if d > 0: borderwidth = int(0.01 * sig.shape[d])
#             for bx in range(borderwidth):
#                 slices[d] = slice(sig.shape[d] - bx - 1, sig.shape[d] - bx)
#                 sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)
#
#         sig = ImageArray(sig,spacing=views[iview].spacing,origin=views[iview].origin)
#         tmpvs = transform_stack_sitk(sig,params[iview],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'])
#
#
#         mask = mask > 0
#         # print('WARNING; 1 ITERATIONS FOR MASK DILATION (DCT WEIGHTS')
#         # mask = ndimage.binary_dilation(mask == 0,iterations=1)
#         ws.append(tmpvs*(mask))
#
#     if changed_stack_properties:
#         for iview in range(len(ws)):
#             ws[iview] = transform_stack_sitk(ws[iview],[1,0,0,0,1,0,0,0,1,0,0,0],
#                                    out_origin=stack_properties['origin'],
#                                    out_shape=stack_properties['size'],
#                                    out_spacing=stack_properties['spacing'])
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         # ws[iw] /= wsum
#         ws[iw] = ws[iw] / wsum
#
#     return ws

# @io_decorator
# def get_weights_simple(
#                     views,
#                     params,
#                     stack_properties,
#                     ):
#     """
#     sigmoid on borders
#     """
#
#     w_stack_properties = stack_properties.copy()
#     minspacing = 3.
#     changed_stack_properties = False
#     if w_stack_properties['spacing'][0] < minspacing:
#         changed_stack_properties = True
#         print('using downsampled images for calculating simple weights..')
#         w_stack_properties['spacing'] = np.array([minspacing]*3)
#         w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']
#
#     ws = []
#     for iview,view in enumerate(views):
#
#         tmporigin = views[iview].origin+views[iview].spacing/2.
#         badplanes = int(0/views[iview].spacing[0]) # in microns
#         tmporigin[0]+= views[iview].spacing[0]*badplanes
#         # print('WATCH OUT! simple_weights: disregarding %s bad planes at the end of the stack' % badplanes)
#         # sig =
#
#         # x,y,z = np.mgrid[:sig.shape[0],:sig.shape[1],:sig.shape[2]]
#         # dists = np.ones(sig.shape)*np.max(sig.shape)
#         # for d in range(3):
#         #     ddists =
#         #     dists = np.min([dists,ddists])
#         reducedview = view[badplanes:-1,:-1,:-1]
#         tmp_view = ImageArray(reducedview+1,spacing=views[iview].spacing,origin=tmporigin,rotation=views[iview].rotation)
#         # tmp_view = ImageArray(view[:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
#         mask = transform_stack_sitk(tmp_view,params[iview],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'],
#                                 interp='nearest')
#
#         def sigmoid(x,borderwidth):
#             x0 = float(borderwidth)/2.
#             a = 12./borderwidth
#             return 1/(1+np.exp(-a*(x-x0)))
#
#         r = 0.1 # relative border width
#         # sig = np.ones(reducedview.shape,dtype=np.float32)
#         sigN = 200
#         sig = np.ones([sigN]*3,dtype=np.float32)
#
#         for d in range(3):
#             borderwidth = int(r * sig.shape[d])
#             print(borderwidth)
#             slices = [slice(0, sig.shape[i]) for i in range(3)]
#             for bx in range(borderwidth):
#                 slices[d] = slice(bx, bx + 1)
#                 sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)
#
#             # don't blend best part of the image (assuming that is true for high zs)
#             if d > 0: borderwidth = int(0.01 * sig.shape[d])
#             for bx in range(borderwidth):
#                 slices[d] = slice(sig.shape[d] - bx - 1, sig.shape[d] - bx)
#                 sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)
#
#         # sig = ImageArray(sig,spacing=views[iview].spacing,origin=views[iview].origin)
#
#         sigspacing = (np.array(views[iview].shape)-1)/(sigN-1)*views[iview].spacing
#         sig = ImageArray(sig,spacing=sigspacing,origin=views[iview].origin)
#
#         tmpvs = transform_stack_sitk(sig,params[iview],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'])
#
#
#         mask = mask > 0
#         # print('WARNING; 1 ITERATIONS FOR MASK DILATION (DCT WEIGHTS')
#         # mask = ndimage.binary_dilation(mask == 0,iterations=1)
#         ws.append(tmpvs*(mask))
#
#     if changed_stack_properties:
#         for iview in range(len(ws)):
#             ws[iview] = transform_stack_sitk(ws[iview],[1,0,0,0,1,0,0,0,1,0,0,0],
#                                    out_origin=stack_properties['origin'],
#                                    out_shape=stack_properties['size'],
#                                    out_spacing=stack_properties['spacing'])
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         # ws[iw] /= wsum
#         ws[iw] = ws[iw] / wsum
#
#     return ws

def get_mask_in_target_space(orig_stack_props,
                               target_stack_props,
                               param,
                               ):

    # tmporigin = views[iview].origin + views[iview].spacing / 2.
    tmporigin = orig_stack_props['origin'] + orig_stack_props['spacing'] / 2.
    badplanes = int(0 / orig_stack_props['spacing'][0])  # in microns
    tmporigin[0] += orig_stack_props['spacing'][0] * badplanes
    # print('WATCH OUT! simple_weights: disregarding %s bad planes at the end of the stack' % badplanes)

    # reducedview = view[badplanes:-1, :-1, :-1]
    # reducedview_shape = np.array(orig_stack_props['size'])-1
    reducedview_shape = np.array(orig_stack_props['size'])-1
    reducedview_shape[0] -= badplanes
    reducedview = np.ones(reducedview_shape,dtype=np.uint16)
    reducedview = ImageArray(reducedview + 1, spacing=orig_stack_props['spacing'], origin=tmporigin)
    mask = transform_stack_sitk(reducedview, param,
                                out_origin=target_stack_props['origin'],
                                out_shape=target_stack_props['size'],
                                out_spacing=target_stack_props['spacing'],
                                interp='nearest')
    mask = mask > 0
    return mask

def get_stack_properties_from_view_dict(view_dict,raw_input_binning=[1,1,1]):

    raw_input_binning = np.array(raw_input_binning)

    info = getStackInfoFromCZI(view_dict['filename'])
    stack_props = dict()
    stack_props['spacing'] = info['spacing'][::-1]
    stack_props['origin'] = info['origins'][view_dict['view']][::-1]
    stack_props['size'] = info['sizes'][view_dict['view']][::-1].astype(np.int64)
    for i in range(3):
        if raw_input_binning[::-1][i] > 1:
            stack_props['size'][i] = stack_props['size'][i] // raw_input_binning[::-1][i]
            stack_props['spacing'][i] = stack_props['spacing'][i] * raw_input_binning[::-1][i]
            stack_props['origin'][i] = stack_props['origin'][i] + (stack_props['spacing'][i]-info['spacing'][::-1][i])/2.

    return stack_props


@io_decorator
def get_weights_simple(
                    orig_stack_propertiess,
                    params,
                    stack_properties,
                    ):
    """
    sigmoid on borders
    """

    # w_stack_properties = stack_properties.copy()
    # minspacing = 3.
    # changed_stack_properties = False
    # if w_stack_properties['spacing'][0] < minspacing:
    #     changed_stack_properties = True
    #     print('using downsampled images for calculating simple weights..')
    #     w_stack_properties['spacing'] = np.array([minspacing]*3)
    #     w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']

    ws = []
    # for iview,view in enumerate(views):
    for iview in range(len(params)):

        # tmporigin = views[iview].origin+views[iview].spacing/2.
        # badplanes = int(0/views[iview].spacing[0]) # in microns
        # tmporigin[0]+= views[iview].spacing[0]*badplanes
        # # print('WATCH OUT! simple_weights: disregarding %s bad planes at the end of the stack' % badplanes)
        # # sig =
        #
        # # x,y,z = np.mgrid[:sig.shape[0],:sig.shape[1],:sig.shape[2]]
        # # dists = np.ones(sig.shape)*np.max(sig.shape)
        # # for d in range(3):
        # #     ddists =
        # #     dists = np.min([dists,ddists])
        # reducedview = view[badplanes:-1,:-1,:-1]
        # tmp_view = ImageArray(reducedview+1,spacing=views[iview].spacing,origin=tmporigin,rotation=views[iview].rotation)
        # # tmp_view = ImageArray(view[:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
        # mask = transform_stack_sitk(tmp_view,params[iview],
        #                        out_origin=w_stack_properties['origin'],
        #                        out_shape=w_stack_properties['size'],
        #                        out_spacing=w_stack_properties['spacing'],
        #                         interp='nearest')

        def sigmoid(x,borderwidth):
            x0 = float(borderwidth)/2.
            a = 12./borderwidth
            return 1/(1+np.exp(-a*(x-x0)))

        r = 0.05 # relative border width
        # sig = np.ones(reducedview.shape,dtype=np.float32)
        sigN = 200
        sig = np.ones([sigN]*3,dtype=np.float32)

        for d in range(3):
            borderwidth = int(r * sig.shape[d])
            print(borderwidth)
            slices = [slice(0, sig.shape[i]) for i in range(3)]
            for bx in range(borderwidth):
                slices[d] = slice(bx, bx + 1)
                sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)

            # don't blend best part of the image (assuming that is true for high zs)
            # if d > 0: borderwidth = int(0.01 * sig.shape[d])
            if d == 0: borderwidth = int(0.01 * sig.shape[d])
            for bx in range(borderwidth):
                slices[d] = slice(sig.shape[d] - bx - 1, sig.shape[d] - bx)
                sig[tuple(slices)] = np.min([sig[tuple(slices)] * 0 + sigmoid(bx, borderwidth), sig[tuple(slices)]], 0)

        # sig = ImageArray(sig,spacing=views[iview].spacing,origin=views[iview].origin)

        # sigspacing = (np.array(views[iview].shape)-1)/(sigN-1)*views[iview].spacing
        sigspacing = (np.array(orig_stack_propertiess[iview]['size'])-1)/(sigN-1)*orig_stack_propertiess[iview]['spacing']
        sig = ImageArray(sig,spacing=sigspacing,origin=orig_stack_propertiess[iview]['origin'])

        tmpvs = transform_stack_sitk(sig,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'])

        mask = get_mask_in_target_space(orig_stack_propertiess[iview],
                                 stack_properties,
                                 params[iview]
                                 )

        # mask = mask > 0
        # print('WARNING; 1 ITERATIONS FOR MASK DILATION (DCT WEIGHTS')
        # mask = ndimage.binary_dilation(mask == 0,iterations=1)
        ws.append(tmpvs*(mask))
        # ws.append(tmpvs)

    wsum = np.sum(ws,0)
    wsum[wsum==0] = 1
    for iw,w in enumerate(ws):
        # ws[iw] /= wsum
        ws[iw] = ws[iw] / wsum

    return ws


# from scipy.fftpack import dctn,idctn
# from scipy import ndimage
# import dask.array as da
# @io_decorator
# def get_weights_dct(
#                     views,
#                     params,
#                     orig_stack_propertiess,
#                     stack_properties,
#                     size=None,
#                     max_kernel=None,
#                     gaussian_kernel=None,
#                     ):
#     """
#     DCT Shannon Entropy, as in:
#     Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
#     http://www.nature.com/articles/nbt.3708
#
#     Adaptations:
#     - consider the full bandwidth, so set r0=d0 in their equation
#     - calculate on blocks of size <size> and then interpolate to full grid
#     - run maximum filter
#     - run smoothing gaussian filter
#     - final sigmoidal blending at view transitions
#
#     :param vrs:
#     :return:
#     """
#
#     w_stack_properties = stack_properties.copy()
#     minspacing = 3.
#     changed_stack_properties = False
#     if w_stack_properties['spacing'][0] < minspacing:
#         changed_stack_properties = True
#         print('using downsampled images for calculating weights..')
#         w_stack_properties['spacing'] = np.array([minspacing]*3)
#         w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']
#
#     vs = []
#     vdils = []
#     for iview,view in enumerate(views):
#
#         tmpvs = transform_stack_sitk(view,matrix_to_params(np.eye(4)),
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'])
#
#         mask = get_mask_in_target_space(orig_stack_propertiess[iview],
#                                  w_stack_properties,
#                                  params[iview]
#                                  )
#
#         vdils.append(mask == 0)
#         vs.append(tmpvs*(mask>0))
#
#     if size is None:
#         size = np.max([4,int(50 / vs[0].spacing[0])]) # 50um
#         print('dct: choosing size %s' %size)
#     if max_kernel is None:
#         max_kernel = int(size/2.)
#         print('dct: choosing max_kernel %s' %max_kernel)
#     if gaussian_kernel is None:
#         gaussian_kernel = int(max_kernel)
#         print('dct: choosing gaussian_kernel %s' %gaussian_kernel)
#
#     print('calculating dct weights...')
#     def determine_quality(vrs):
#
#         """
#         DCT Shannon Entropy, as in:
#         Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
#         http://www.nature.com/articles/nbt.3708
#         Consider the full bandwidth, so set r0=d0 in their equation
#         :param vrs:
#         :return:
#         """
#         # print('dw...')
#
#         vrs = np.copy(vrs)
#
#         axes = [0,1,2]
#         ds = []
#         for v in vrs:
#
#             if np.sum(v==0) > np.product(v.shape) * (4/5.):
#                 ds.append([0])
#                 continue
#             elif v.min()<0.0001:
#                 v[v==0] = v[v>0].min() # or nearest neighbor
#
#             d = dctn(v,norm='ortho',axes=axes)
#             # cut = size//2
#             # d[:cut,:cut,:cut] = 0
#             ds.append(d.flatten())
#
#         # l2 norm
#         dsl2 = np.array([np.sum(np.abs(d)) for d in ds])
#         # don't divide by zero below
#         dsl2[dsl2==0] = 1
#
#         def abslog(x):
#             res = np.zeros_like(x)
#             x = np.abs(x)
#             res[x==0] = 0
#             res[x>0] = np.log2(x[x>0])
#             return res
#
#         ws = np.array([-np.sum(np.abs(d)*abslog(d/dsl2[id])) for id,d in enumerate(ds)])
#
#         # simple weights in case everything is zero
#         if not ws.max():
#             ws = np.ones(len(ws))/float(len(ws))
#
#         return ws[:,None,None,None]
#
#
#     x = da.from_array(np.array(vs), chunks=(len(vs),size,size,size))
#     # ws=x.map_blocks(determine_quality,dtype=np.float)
#     ws = x.map_blocks(determine_quality,dtype=np.float,chunks=(len(vs),1,1,1))
#
#     ws = ws.compute(scheduler = 'threads')
#     ws = np.array(ws)
#
#     ws = ImageArray(ws,
#                     spacing= np.array([size]*3)*np.array(w_stack_properties['spacing']),
#                     origin = w_stack_properties['origin'] + ((size-1)*w_stack_properties['spacing'])/2.,
#                     )
#
#     newws = []
#     for iw in range(len(ws)):
#         newws.append(transform_stack_sitk(ws[iw],
#                             [1,0,0,0,1,0,0,0,1,0,0,0],
#                             # out_shape=stack_properties['size'],
#                             # out_origin=stack_properties['origin'],
#                             # out_spacing=stack_properties['spacing'],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'],
#                             interp='linear',
#                              ))
#     ws = np.array(newws)
#
#     for iw,w in enumerate(ws):
#         print('filtering')
#         ws[iw] = ndimage.maximum_filter(ws[iw],max_kernel)
#
#     for iw,w in enumerate(ws):
#         ws[iw][vdils[iw]] = 0
#
#     wsmin = ws.min(0)
#     wsmax = ws.max(0)
#     ws = np.array([(w - wsmin)/(wsmax - wsmin + 0.01) for w in ws])
#     # ws = np.array([(w - wsmin)/(wsmax - wsmin) for w in ws])
#
#     # for iw,w in enumerate(ws):
#     #     ws[iw][vdils[iw]] = 0.00001
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         ws[iw] /= wsum
#
#     # tifffile.imshow(np.array([np.array(ts)*10,ws]).swapaxes(-3,-2),vmax=10000)
#     for iw,w in enumerate(ws):
#         print('filtering')
#         # ws[iw] = ndimage.maximum_filter(ws[iw],10)
#         # ws[iw][vdils[iw]] = 0.00001
#         ws[iw] = ndimage.gaussian_filter(ws[iw],gaussian_kernel)
#         # zeros = ndimage.binary_dilation(vs[iw] == 0)
#         # ws[iw][zeros] = 0.00001
#         # ws[iw][vdils[iw]] = 0.00001
#         ws[iw][vdils[iw]] = 0
#
#
#     # HEURISTIC to adapt weights to number of views
#     # idea: typically, 2-3 views carry good information at a given location
#     # and the rest should not contribute
#     # w**exp with exp>1 polarises the weights
#     # we want to find exp such that 90% of the quality contribution
#     # is given by the two best views
#     # this is overall and the analysis is limited to regions where the best view
#     # has at least double its baseline value 1/len(views)
#     # alternatively: best view should have 0.5
#
#     if len(ws) > 2:
#
#         print('applying heuristic to adapt weights to N=%s views' %len(ws))
#         print('criterion: weights**exp such that best two views > 0.9')
#
#         wsum = np.sum(ws,0)
#         wsum[wsum==0] = 1
#         for iw,w in enumerate(ws):
#             ws[iw] /= wsum
#
#         wf = ws[:, np.max(ws, 0) > (2 * (1 / len(ws)))]
#         # wf = wf[:,np.sum(wf,0)>0]
#         wfs = np.sort(wf, axis=0)
#
#         def energy(exp):
#             exp = exp[0]
#             tmpw = wfs ** exp
#             tmpsum = np.sum(tmpw, 0)
#             tmpw = tmpw / tmpsum
#             nsum = np.sum(tmpw[-2:], (-1)) / wfs.shape[-1]
#             energy = np.abs(np.sum(nsum) - 0.9)
#             return energy
#
#         from scipy import optimize
#         res = optimize.minimize(energy, [0.5], bounds=[[0.5, 5]], method='L-BFGS-B', options={'maxiter': 10})
#
#         exp = res.x[0]
#
#         print('found exp=%s' %exp)
#
#         ws = [ws[i]**exp for i in range(len(ws))]
#
#     ws = list(ws)
#     for iw,w in enumerate(ws):
#         ws[iw] = ImageArray(ws[iw],
#                             origin=w_stack_properties['origin'],
#                             spacing=w_stack_properties['spacing'])
#
#
#     if changed_stack_properties:
#         for iview in range(len(ws)):
#             ws[iview] = transform_stack_sitk(ws[iview],[1,0,0,0,1,0,0,0,1,0,0,0],
#                                    out_origin=stack_properties['origin'],
#                                    out_shape=stack_properties['size'],
#                                    out_spacing=stack_properties['spacing'])
#
#     # smooth edges
#     ws_simple = get_weights_simple(
#                     orig_stack_propertiess,
#                     params,
#                     stack_properties
#     )
#
#     ws = [ws[i]*ws_simple[i] for i in range(len(ws))]
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         ws[iw] /= wsum
#
#     return ws

import dask.array as da
import h5py
def fuse_blockwise(fn,
                   fns_tview,
                   params,
                   stack_properties,
                   fusion_block_overlap=None,
                   weights_func=None,
                   fusion_func=None,
                   weights_kwargs=None,
                   fusion_kwargs=None,
                   ):

    stack_properties = io_utils.process_input_element(stack_properties)
    params = io_utils.process_input_element(params)

    # in pixels
    if fusion_block_overlap is None:
        fusion_block_overlap = 0

    nviews = len(fns_tview)

    tviews = []
    for fn_tview in fns_tview:
        tmpdset = h5py.File(fn_tview)['array']
        tmpda = da.from_array(tmpdset,chunks = tmpdset.chunks)
        tviews.append(tmpda)

    # tviews_stack = da.stack(tviews,axis=0)

    import dask
    # with dask.config.set(scheduler='single-threaded'):
    with dask.config.set(scheduler='threads'):

        # tviews_stack = da.from_array(tviews, chunks = ((nviews,),)+tviews[0].chunks)
        # tviews_stack = da.from_array(tviews, chunks = (nviews,50,50,50))
        # tviews_stack = da.concatenate(tviews,axis=0)
        # tviews_stack = da.vstack(tviews)
        tviews_stack = da.stack(tviews)

        depth = int(fusion_block_overlap)
        depth_dict = {0: 0, 1: depth, 2: depth, 3: depth}
        tviews_overlap = da.overlap.overlap(tviews_stack,
                               depth=depth_dict,
                               # boundary = {0: 'periodic', 1: 'periodic', 2: 'periodic', 3: 'periodic'})
                               boundary = {1: 'periodic', 2: 'periodic', 3: 'periodic'})

        # weights_kwargs['params'] = params
        # fusion_kwargs['params'] = params
        # weights_kwargs['stack_properties'] = stack_properties
        # fusion_kwargs['stack_properties'] = stack_properties

        result_overlap = tviews_overlap.map_blocks(fuse_block, drop_axis = [0], dtype=tviews[0].dtype,
                                                   **{
                                                       'params': params,
                                                       'stack_properties': stack_properties,
                                                       'weights_func': weights_func,
                                                       'fusion_func': fusion_func,
                                                       'weights_kwargs': weights_kwargs,
                                                       'fusion_kwargs': fusion_kwargs,

                                                      })

        # dummy
        # result_overlap = tviews_overlap.map_blocks(lambda x: x[0], drop_axis=[0], dtype=tviews[0].dtype)

        trim_dict = {i:depth for i in range(3)}
        result = da.overlap.trim_internal(result_overlap, trim_dict)
        # pdb.set_trace()

        if os.path.exists(fn):
            print('WARNING: OVERWRITING %s' %fn)
            os.remove(fn)


        result.to_hdf5(fn,'array',compression='gzip')#,scheduler = "single-threaded")
        # maybe add imagearray metadata here

    return fn

def fuse_block(tviews_block,params,stack_properties,weights_func,fusion_func,weights_kwargs,fusion_kwargs):

    # return np.random.randint(0,100,tviews_block.shape[1:]).astype(tviews_block.dtype)

    max_vals = np.array([tview_block.max() for tview_block in tviews_block])

    inds = np.where(max_vals>0)[0]

    # abort in trivial case
    if len(inds) == 0:
        return tviews_block[0]
    elif len(inds) == 1:
        return tviews_block[inds[0]]

    tviews_block = tviews_block[inds]
    params = params[inds]

    print('performing fusion on %s blocks' %len(params))

    weights_kwargs['params'] = params
    fusion_kwargs['params'] = params

    tviews = [ImageArray(tview_block,spacing=stack_properties['spacing'],origin=stack_properties['origin']) for tview_block in tviews_block]

    # stack_properties = tviews[0].get_info()

    # orig_stack_propertiess = [0 for i in range(len(params))]
    # orig_stack_propertiess = []

    block_stack_properties = stack_properties.copy()
    block_stack_properties['size'] = np.array(tviews_block[0].shape)

    weights_kwargs['stack_properties'] = block_stack_properties
    fusion_kwargs['stack_properties'] = block_stack_properties

    weights = weights_func(tviews,
                              **weights_kwargs)
                              # params,
                              # orig_stack_propertiess,
                              # stack_properties,
                              # )

                              # views,
                              # params,
                              # orig_stack_propertiess,
                              # stack_properties,
                              # size=None,
                              # max_kernel=None,
                              # gaussian_kernel=None,
                              # how_many_best_views=1,
                              # cumulative_weight_best_views=0.9,

    fused = fusion_func(tviews,weights=weights,
            **fusion_kwargs)

    return fused

        # tviews,
        # params,
        # stack_properties,
        # weights = weights,
        # blur_func = blur_view_in_target_space,
        # orig_prop_list=orig_stack_propertiess,
    # )

    # views,
    # params,
    # stack_properties,
    # num_iterations = 25,
    # sz = 4,
    # sxy = 0.5,
    # tol = 5e-5,
    # weights = None,
    # regularisation = False,
    # blur_func = blur_view_in_view_space,
    # orig_prop_list = None,
    # views_in_target_space = True,

    # return fused

    # fused_block = np.random.randint(0,100,block.shape[1:]).astype(block.dtype)
    # return fused_block

def get_dct_options(spacing,size=None,max_kernel=None,gaussian_kernel=None):

    spacing = np.max([3,spacing])

    if size is None:
        size = np.max([4,int(50 / spacing)]) # 50um
        print('dct: choosing size %s' %size)
    if max_kernel is None:
        max_kernel = int(size/2.)
        print('dct: choosing max_kernel %s' %max_kernel)
    if gaussian_kernel is None:
        gaussian_kernel = int(max_kernel)
        print('dct: choosing gaussian_kernel %s' %gaussian_kernel)

    return size,max_kernel,gaussian_kernel

from scipy.fftpack import dctn,idctn
# from scipy.fftpack import dct,idct
from scipy import ndimage
import dask.array as da
@io_decorator
def get_weights_dct(
                    views,
                    params,
                    orig_stack_propertiess,
                    stack_properties,
                    size=None,
                    max_kernel=None,
                    gaussian_kernel=None,
                    how_many_best_views = 1,
                    cumulative_weight_best_views = 0.9,
                    ):
    """
    DCT Shannon Entropy, as in:
    Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
    http://www.nature.com/articles/nbt.3708

    Adaptations:
    - consider the full bandwidth, so set r0=d0 in their equation
    - calculate on blocks of size <size> and then interpolate to full grid
    - run maximum filter
    - run smoothing gaussian filter
    - final sigmoidal blending at view transitions

    :param vrs:
    :return:
    """

    w_stack_properties = stack_properties.copy()
    minspacing = 3.
    changed_stack_properties = False
    if w_stack_properties['spacing'][0] < minspacing:
        changed_stack_properties = True
        print('using downsampled images for calculating weights..')
        w_stack_properties['spacing'] = np.array([minspacing]*3)
        w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']

    vs = []
    vdils = []
    for iview,view in enumerate(views):

        tmpvs = transform_stack_sitk(view,matrix_to_params(np.eye(4)),
                               out_origin=w_stack_properties['origin'],
                               out_shape=w_stack_properties['size'],
                               out_spacing=w_stack_properties['spacing'])

        # mask = get_mask_in_target_space(orig_stack_propertiess[iview],
        #                          w_stack_properties,
        #                          params[iview]
        #                          )

        # mask = tmpvs > 0
        mask = np.ones(tmpvs.shape,dtype=np.bool)

        vdils.append(mask == 0)
        vs.append(tmpvs*(mask>0))

    size,max_kernel,gaussian_kernel = get_dct_options(w_stack_properties['spacing'][0],
                                                      size,
                                                      max_kernel,
                                                      gaussian_kernel,
                                                      )
    # if size is None:
    #     size = np.max([4,int(50 / vs[0].spacing[0])]) # 50um
    #     print('dct: choosing size %s' %size)
    # if max_kernel is None:
    #     max_kernel = int(size/2.)
    #     print('dct: choosing max_kernel %s' %max_kernel)
    # if gaussian_kernel is None:
    #     gaussian_kernel = int(max_kernel)
    #     print('dct: choosing gaussian_kernel %s' %gaussian_kernel)

    print('calculating dct weights...')
    def determine_quality(vrs):

        """
        DCT Shannon Entropy, as in:
        Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
        http://www.nature.com/articles/nbt.3708
        Consider the full bandwidth, so set r0=d0 in their equation
        :param vrs:
        :return:
        """
        # print('dw...')

        vrs = np.copy(vrs)

        axes = [0,1,2]
        ds = []
        for v in vrs:

            if np.sum(v==0) > np.product(v.shape) * (4/5.):
                ds.append([0])
                continue
            elif v.min()<0.0001:
                v[v==0] = v[v>0].min() # or nearest neighbor

            d = dctn(v,norm='ortho',axes=axes)
            # d = dct(dct(dct(v,axis=-1,norm='ortho'),axis=-2,norm='ortho'),axis=-3,norm='ortho')
            # cut = size//2
            # d[:cut,:cut,:cut] = 0
            ds.append(d.flatten())

        # l2 norm
        dsl2 = np.array([np.sum(np.abs(d)) for d in ds])
        # don't divide by zero below
        dsl2[dsl2==0] = 1

        def abslog(x):
            res = np.zeros_like(x)
            x = np.abs(x)
            res[x==0] = 0
            res[x>0] = np.log2(x[x>0])
            return res

        ws = np.array([-np.sum(np.abs(d)*abslog(d/dsl2[id])) for id,d in enumerate(ds)])

        # simple weights in case everything is zero
        if not ws.max():
            ws = np.ones(len(ws))/float(len(ws))


        # HEURISTIC to adapt weights to number of views
        # idea: typically, 2-3 views carry good information at a given location
        # and the rest should not contribute
        # w**exp with exp>1 polarises the weights
        # we want to find exp such that 90% of the quality contribution
        # is given by the two best views
        # this is overall and the analysis is limited to regions where the best view
        # has at least double its baseline value 1/len(views)
        # alternatively: best view should have 0.5

        if len(ws) > 2 and ws.min() < ws.max():

            # print('applying heuristic to adapt weights to N=%s views' % len(ws))
            # print('criterion: weights**exp such that best two views > 0.9')

            wsum = np.sum(ws, 0)
            # wsum[wsum == 0] = 1
            for iw, w in enumerate(ws):
                ws[iw] /= wsum



            # wf = ws[:, np.max(ws, 0) > (2 * (1 / len(ws)))]
            wf = ws
            # wf = wf[:,np.sum(wf,0)>0]
            wfs = np.sort(wf, axis=0)

            def energy(exp):
                exp = exp[0]
                tmpw = wfs ** exp
                tmpsum = np.sum(tmpw, 0)
                tmpw = tmpw / tmpsum

                nsum = np.sum(tmpw[-int(how_many_best_views):], (-1))# / wfs.shape[-1]
                energy = np.abs(np.sum(nsum) - cumulative_weight_best_views)

                # nsum = np.sum(tmpw[-1:], (-1))# / wfs.shape[-1]
                # energy = np.abs(np.sum(nsum) - 0.5)

                return energy

            from scipy import optimize
            res = optimize.minimize(energy, [0.5], bounds=[[0.1, 10]], method='L-BFGS-B', options={'maxiter': 10})

            exp = res.x[0]

            # print('found exp=%s' % exp)

            ws = [ws[i] ** exp for i in range(len(ws))]



            ws = np.array(ws)

        return ws[:,None,None,None]

    x = da.from_array(np.array(vs), chunks=(len(vs),size,size,size))
    # ws=x.map_blocks(determine_quality,dtype=np.float)
    ws = x.map_blocks(determine_quality,dtype=np.float,chunks=(len(vs),1,1,1))

    ws = ws.compute(scheduler = 'threads')
    ws = np.array(ws)

    ws = ImageArray(ws,
                    spacing= np.array([size]*3)*np.array(w_stack_properties['spacing']),
                    origin = w_stack_properties['origin'] + ((size-1)*w_stack_properties['spacing'])/2.,
                    )

    newws = []
    for iw in range(len(ws)):
        newws.append(transform_stack_sitk(ws[iw],
                            [1,0,0,0,1,0,0,0,1,0,0,0],
                            # out_shape=stack_properties['size'],
                            # out_origin=stack_properties['origin'],
                            # out_spacing=stack_properties['spacing'],
                               out_origin=w_stack_properties['origin'],
                               out_shape=w_stack_properties['size'],
                               out_spacing=w_stack_properties['spacing'],
                            interp='linear',
                             ))
    ws = np.array(newws)


    for iw,w in enumerate(ws):
        print('filtering')
        ws[iw] = ndimage.maximum_filter(ws[iw],max_kernel)

    for iw,w in enumerate(ws):
        ws[iw][vdils[iw]] = 0

    # wsmin = ws.min(0)
    # wsmax = ws.max(0)
    # ws = np.array([(w - wsmin)/(wsmax - wsmin + 0.01) for w in ws])

    wsum = np.sum(ws,0)
    wsum[wsum==0] = 1
    for iw,w in enumerate(ws):
        ws[iw] /= wsum

    # tifffile.imshow(np.array([np.array(ts)*10,ws]).swapaxes(-3,-2),vmax=10000)
    for iw,w in enumerate(ws):
        print('filtering')
        # ws[iw] = ndimage.maximum_filter(ws[iw],10)
        # ws[iw][vdils[iw]] = 0.00001
        ws[iw] = ndimage.gaussian_filter(ws[iw],gaussian_kernel)
        # zeros = ndimage.binary_dilation(vs[iw] == 0)
        # ws[iw][zeros] = 0.00001
        # ws[iw][vdils[iw]] = 0.00001
        ws[iw][vdils[iw]] = 0


    ws = list(ws)
    for iw,w in enumerate(ws):
        ws[iw] = ImageArray(ws[iw],
                            origin=w_stack_properties['origin'],
                            spacing=w_stack_properties['spacing'])




    if changed_stack_properties:
        for iview in range(len(ws)):
            ws[iview] = transform_stack_sitk(ws[iview],[1,0,0,0,1,0,0,0,1,0,0,0],
                                   out_origin=stack_properties['origin'],
                                   out_shape=stack_properties['size'],
                                   out_spacing=stack_properties['spacing'])

    # # smooth edges
    # ws_simple = get_weights_simple(
    #                 orig_stack_propertiess,
    #                 params,
    #                 stack_properties
    # )
    #
    # ws = [ws[i]*ws_simple[i] for i in range(len(ws))]

    wsum = np.sum(ws,0)
    wsum[wsum==0] = 1
    for iw,w in enumerate(ws):
        ws[iw] /= wsum

    return ws

# from scipy.fftpack import dctn,idctn
# from scipy import ndimage
# import dask.array as da
# @io_decorator
# def get_weights_dct(
#                     views,
#                     params,
#                     orig_stack_propertiess,
#                     stack_properties,
#                     size=None,
#                     max_kernel=None,
#                     gaussian_kernel=None,
#                     how_many_best_views = 2,
#                     cumulative_weight_best_views = 0.9,
#                     ):
#     """
#     DCT Shannon Entropy, as in:
#     Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
#     http://www.nature.com/articles/nbt.3708
#
#     Adaptations:
#     - consider the full bandwidth, so set r0=d0 in their equation
#     - calculate on blocks of size <size> and then interpolate to full grid
#     - run maximum filter
#     - run smoothing gaussian filter
#     - final sigmoidal blending at view transitions
#
#     :param vrs:
#     :return:
#     """
#
#     w_stack_properties = stack_properties.copy()
#     minspacing = 3.
#     changed_stack_properties = False
#     if w_stack_properties['spacing'][0] < minspacing:
#         changed_stack_properties = True
#         print('using downsampled images for calculating weights..')
#         w_stack_properties['spacing'] = np.array([minspacing]*3)
#         w_stack_properties['size'] = (stack_properties['spacing'][0]/w_stack_properties['spacing'][0])*stack_properties['size']
#
#     vs = []
#     vdils = []
#     for iview,view in enumerate(views):
#
#         tmpvs = transform_stack_sitk(view,matrix_to_params(np.eye(4)),
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'])
#
#         mask = get_mask_in_target_space(orig_stack_propertiess[iview],
#                                  w_stack_properties,
#                                  params[iview]
#                                  )
#
#         vdils.append(mask == 0)
#         vs.append(tmpvs*(mask>0))
#
#     if size is None:
#         size = np.max([4,int(50 / vs[0].spacing[0])]) # 50um
#         print('dct: choosing size %s' %size)
#     if max_kernel is None:
#         max_kernel = int(size/2.)
#         print('dct: choosing max_kernel %s' %max_kernel)
#     if gaussian_kernel is None:
#         gaussian_kernel = int(max_kernel)
#         print('dct: choosing gaussian_kernel %s' %gaussian_kernel)
#
#     print('calculating dct weights...')
#     def determine_quality(vrs):
#
#         """
#         DCT Shannon Entropy, as in:
#         Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
#         http://www.nature.com/articles/nbt.3708
#         Consider the full bandwidth, so set r0=d0 in their equation
#         :param vrs:
#         :return:
#         """
#         # print('dw...')
#
#         vrs = np.copy(vrs)
#
#         axes = [0,1,2]
#         ds = []
#         for v in vrs:
#
#             if np.sum(v==0) > np.product(v.shape) * (4/5.):
#                 ds.append([0])
#                 continue
#             elif v.min()<0.0001:
#                 v[v==0] = v[v>0].min() # or nearest neighbor
#
#             d = dctn(v,norm='ortho',axes=axes)
#             # cut = size//2
#             # d[:cut,:cut,:cut] = 0
#             ds.append(d.flatten())
#
#         # l2 norm
#         dsl2 = np.array([np.sum(np.abs(d)) for d in ds])
#         # don't divide by zero below
#         dsl2[dsl2==0] = 1
#
#         def abslog(x):
#             res = np.zeros_like(x)
#             x = np.abs(x)
#             res[x==0] = 0
#             res[x>0] = np.log2(x[x>0])
#             return res
#
#         ws = np.array([-np.sum(np.abs(d)*abslog(d/dsl2[id])) for id,d in enumerate(ds)])
#
#         # simple weights in case everything is zero
#         if not ws.max():
#             ws = np.ones(len(ws))/float(len(ws))
#
#
#         # HEURISTIC to adapt weights to number of views
#         # idea: typically, 2-3 views carry good information at a given location
#         # and the rest should not contribute
#         # w**exp with exp>1 polarises the weights
#         # we want to find exp such that 90% of the quality contribution
#         # is given by the two best views
#         # this is overall and the analysis is limited to regions where the best view
#         # has at least double its baseline value 1/len(views)
#         # alternatively: best view should have 0.5
#
#         if len(ws) > 2 and ws.min() < ws.max():
#
#             # print('applying heuristic to adapt weights to N=%s views' % len(ws))
#             # print('criterion: weights**exp such that best two views > 0.9')
#
#             wsum = np.sum(ws, 0)
#             # wsum[wsum == 0] = 1
#             for iw, w in enumerate(ws):
#                 ws[iw] /= wsum
#
#
#
#             # # wf = ws[:, np.max(ws, 0) > (2 * (1 / len(ws)))]
#             # wf = ws
#             # # wf = wf[:,np.sum(wf,0)>0]
#             # wfs = np.sort(wf, axis=0)
#             #
#             # def energy(exp):
#             #     exp = exp[0]
#             #     tmpw = wfs ** exp
#             #     tmpsum = np.sum(tmpw, 0)
#             #     tmpw = tmpw / tmpsum
#             #
#             #     nsum = np.sum(tmpw[-int(how_many_best_views):], (-1))# / wfs.shape[-1]
#             #     energy = np.abs(np.sum(nsum) - cumulative_weight_best_views)
#             #
#             #     # nsum = np.sum(tmpw[-1:], (-1))# / wfs.shape[-1]
#             #     # energy = np.abs(np.sum(nsum) - 0.5)
#             #
#             #     return energy
#             #
#             # from scipy import optimize
#             # res = optimize.minimize(energy, [0.5], bounds=[[0.1, 10]], method='L-BFGS-B', options={'maxiter': 10})
#             #
#             # exp = res.x[0]
#             #
#             # # print('found exp=%s' % exp)
#             #
#             # ws = [ws[i] ** exp for i in range(len(ws))]
#
#
#
#             ws = np.array(ws)
#
#         return ws[:,None,None,None]
#
#     x = da.from_array(np.array(vs), chunks=(len(vs),size,size,size))
#     # ws=x.map_blocks(determine_quality,dtype=np.float)
#     ws = x.map_blocks(determine_quality,dtype=np.float,chunks=(len(vs),1,1,1))
#
#     ws = ws.compute(scheduler = 'threads')
#     ws = np.array(ws)
#
#     ws = ImageArray(ws,
#                     spacing= np.array([size]*3)*np.array(w_stack_properties['spacing']),
#                     origin = w_stack_properties['origin'] + ((size-1)*w_stack_properties['spacing'])/2.,
#                     )
#
#     newws = []
#     for iw in range(len(ws)):
#         newws.append(transform_stack_sitk(ws[iw],
#                             [1,0,0,0,1,0,0,0,1,0,0,0],
#                             # out_shape=stack_properties['size'],
#                             # out_origin=stack_properties['origin'],
#                             # out_spacing=stack_properties['spacing'],
#                                out_origin=w_stack_properties['origin'],
#                                out_shape=w_stack_properties['size'],
#                                out_spacing=w_stack_properties['spacing'],
#                             interp='linear',
#                              ))
#     ws = np.array(newws)
#
#
#     for iw,w in enumerate(ws):
#         print('filtering')
#         ws[iw] = ndimage.maximum_filter(ws[iw],max_kernel)
#
#     for iw,w in enumerate(ws):
#         ws[iw][vdils[iw]] = 0
#
#     wsmin = ws.min(0)
#     wsmax = ws.max(0)
#     ws = np.array([(w - wsmin)/(wsmax - wsmin + 0.01) for w in ws])
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         ws[iw] /= wsum
#
#     # tifffile.imshow(np.array([np.array(ts)*10,ws]).swapaxes(-3,-2),vmax=10000)
#     for iw,w in enumerate(ws):
#         print('filtering')
#         # ws[iw] = ndimage.maximum_filter(ws[iw],10)
#         # ws[iw][vdils[iw]] = 0.00001
#         ws[iw] = ndimage.gaussian_filter(ws[iw],gaussian_kernel)
#         # zeros = ndimage.binary_dilation(vs[iw] == 0)
#         # ws[iw][zeros] = 0.00001
#         # ws[iw][vdils[iw]] = 0.00001
#         ws[iw][vdils[iw]] = 0
#
#
#     ws = list(ws)
#     for iw,w in enumerate(ws):
#         ws[iw] = ImageArray(ws[iw],
#                             origin=w_stack_properties['origin'],
#                             spacing=w_stack_properties['spacing'])
#
#
#
#
#     if changed_stack_properties:
#         for iview in range(len(ws)):
#             ws[iview] = transform_stack_sitk(ws[iview],[1,0,0,0,1,0,0,0,1,0,0,0],
#                                    out_origin=stack_properties['origin'],
#                                    out_shape=stack_properties['size'],
#                                    out_spacing=stack_properties['spacing'])
#
#     # smooth edges
#     ws_simple = get_weights_simple(
#                     orig_stack_propertiess,
#                     params,
#                     stack_properties
#     )
#
#     ws = [ws[i]*ws_simple[i] for i in range(len(ws))]
#
#     wsum = np.sum(ws,0)
#     wsum[wsum==0] = 1
#     for iw,w in enumerate(ws):
#         ws[iw] /= wsum
#
#     return ws


@io_decorator
def fuse_views_weights(views,
                       params,
                       stack_properties,
                       weights=None,
                       views_in_target_space = True,
                       ):

    # if spacing is None:
    #     spacing = np.max([view.spacing for view in views],0)

    # volume = get_union_volume(views,params)
    # stack_properties = calc_stack_properties_from_volume(volume,spacing)

    if not views_in_target_space:
        transformed = []
        for iview,view in enumerate(views):
            tmp = transform_stack_sitk(view,params[iview],
                                   out_origin=stack_properties['origin'],
                                   out_shape=stack_properties['size'],
                                   out_spacing=stack_properties['spacing'])

            transformed.append(np.array(tmp))
    else:
        transformed = views

    if weights is not None:
        f = np.zeros_like(transformed[0])
        for iw in range(len(transformed)):
                f += (weights[iw]*transformed[iw].astype(np.float)).astype(np.uint16)
    else:
        f = np.mean(transformed,0)

    f = np.clip(f,0,2**16-1)
    f = ImageArray(f.astype(np.uint16),spacing=stack_properties['spacing'],origin=stack_properties['origin'])

    return f

@io_decorator
def calc_stack_properties_from_views_and_params(views,params,spacing=None,mode='sample'):

    spacing = np.array(spacing).astype(np.float64)
    if spacing is None:
        spacing = np.max([view.spacing for view in views],0)

    if mode == 'sample':
        volume = get_sample_volume(views,params)
    elif mode == 'union':
        volume = get_union_volume(views,params)
    elif mode == 'intersection':
        volume = get_intersection_volume(views,params)

    stack_properties = calc_stack_properties_from_volume(volume,spacing)
    return stack_properties

def transform_view_and_save_chunked(fn,view,params,iview,stack_properties,chunksize=None):

    print('transforming %s' %fn)

    params = io_utils.process_input_element(params)
    stack_properties = io_utils.process_input_element(stack_properties)

    res = transform_stack_sitk(view, params[iview], stack_properties=stack_properties)

    # if chunksize_phys is not None:
    #
    #     chunksize = int(chunksize_phys/stack_properties['spacing'][0])
    #     chunksize = np.max([50,chunksize])
    #     chunksize = np.min([500,chunksize])
    #
    # else:
    #     chunksize = 100

    if chunksize is None:
        chunks = np.min([[100]*3,stack_properties['size']],0)
        chunks = tuple([int(i) for i in chunks])
    else:
        chunks = tuple([int(chunksize)]*3)


    f = h5py.File(fn,'w')
    # chunks = np.min([[chunksize]*3, res.shape], 0)
    # chunks = tuple(chunks)
    f.create_dataset("array", data=np.array(res), chunks=chunks, compression="gzip")
    f['spacing'] = np.array(stack_properties['spacing'])
    f['origin'] = np.array(stack_properties['origin'])
    f['rotation'] = 0
    f.close()

    return fn

# # transform_view_with_decorator = io_decorator(transform_stack_sitk)
# @io_decorator
# def transform_view_with_decorator(view,params,iview,stack_properties):
#
#     res = transform_stack_sitk(view,params[iview],stack_properties=stack_properties)
#
#     return res
    #
    # sview = sitk.GetImageFromArray(view)
    # sview.SetSpacing(view.spacing[::-1])
    # sview.SetOrigin(view.origin[::-1])
    # sview = transformStack(params_invert_coordinates(params[iview]),
    #                        sview,
    #                        outOrigin=stack_properties['origin'][::-1],
    #                        outShape=stack_properties['size'][::-1],
    #                        outSpacing=stack_properties['spacing'][::-1])
    #
    # return ImageArray(sitk.GetArrayFromImage(sview),spacing=stack_properties['spacing'])

@io_decorator
def fuse_dct(views,params,stack_properties):
    weights = get_weights_dct(views,params,stack_properties)
    return fuse_views_weights(views,params,stack_properties,weights=weights)

@io_decorator
def fuse_views_content(views,
                       axisOfRotation=1,
                       gaussian_kernel_size=1.,
                       window_size=5,max_proj=100):

    """
    deprecated fusion
    :param views:
    :param axisOfRotation:
    :param gaussian_kernel_size:
    :param window_size:
    :param max_proj:
    :return:
    """

    spacing = views[0].spacing
    views = np.array(views)
    nviews = len(views)

    def fuse_plane(iplane):

        print(iplane)

        plane_slice = [slice(0,views[0].shape[dim]) for dim in range(views[0].ndim)]
        plane_slice[axisOfRotation] = iplane
        plane_slice = tuple(plane_slice)
        view_plane_slice = (slice(0,len(views)),)+plane_slice
        plane = views[view_plane_slice].astype(np.float32)

        axes = [0,1]
        weights = []
        derivss = []
        for iview,view in enumerate(plane):
            derivs = []
            domain = view > 0
            ndomain = view==0
            ndomain = ndimage.binary_erosion(ndomain,iterations=1)
            ndomain = ndimage.binary_dilation(ndomain,iterations=int(np.max([gaussian_kernel_size,window_size])))
            for dim in axes:
                deriv = np.abs(ndimage.gaussian_filter1d(view,gaussian_kernel_size,axis=dim,order=1))
                deriv[ndomain] = 0.00
                # this step above induces lines!!!
                deriv[ndomain] = 0
                deriv[domain] = deriv[domain] / view[domain]
                deriv = ndimage.convolve1d(deriv,np.ones(window_size),axis=dim)
                deriv = ndimage.filters.maximum_filter1d(deriv,max_proj,axis=dim)
                derivs.append(deriv)
            derivss.append(derivs)
            weight = np.sum([np.abs(deriv)**5 for deriv in derivs],0)
            # weight = np.sum([np.abs(deriv)**10 for deriv in derivs],0)
            # print('watch out in fusion!')
            weight[ndomain] = 0.00
#             weight = ndimage.grey_dilation
            weights.append(weight)

        weights = np.array(weights)
        weightsum = np.sum(weights,0)
        domain = weightsum > 0
        # weights[:,domain] /= weightsum[domain]
        weights[:,domain] = weights[:,domain] / weightsum[domain]

        result_array[plane_slice] = np.sum([weights[iview]*plane[iview] for iview in range(nviews)],0)

        return

    # slices = []
    # for iplane in range(views[0].shape[axisOfRotation]):
    #     plane_slice = [slice(0,view[0][dim]) for dim in range(views[0].ndim)]
    #     plane_slice[axisOfRotation] = iplane
    #     slices.append((slice(0,len(views)),)+tuple(plane_slice))

#     from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool
    import ctypes,multiprocessing

    result_array_base = multiprocessing.Array(ctypes.c_uint16, int(np.product(views[0].shape)))
    result_array = np.ctypeslib.as_array(result_array_base.get_obj())
    result_array = result_array.reshape(*views[0].shape)

    pool = ThreadPool()
    pmap = pool.map
    # pmap = map
    pmap(fuse_plane, [iplane for iplane in range(views[0].shape[axisOfRotation])])

    pool.close()

    # result_array = sitk.GetImageFromArray(result_array)
    # drop origin for fused image
    return ImageArray(result_array,origin=np.zeros(3),spacing=spacing)

@io_decorator
def calc_lambda_fusion_seg(
                        views,
                        params,
                        stack_properties,
                        ):

    """
    calc seg which can serve as an input for fusion_lambda.
    Like this, seg and the fusion can be constant despite varying intensities (or channels)
    """
    # stack_properties = calc_stack_properties_from_views_and_params(views,params,spacing)

    # save views for later

    ts = []
    for i in range(len(views)):
        print('transforming view %s' %i)
        t = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                views[i]+1,params[i], # adding one because of weights (taken away again further down)
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        # make sure that only those pixels are kept which are interpolated from 100% valid interpolation pixels
        tmp_view = ImageArray(views[i][:-1,:-1,:-1]+1,spacing=views[i].spacing,origin=views[i].origin+views[i].spacing/2.,rotation=views[i].rotation)

        mask = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                tmp_view, params[i],
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                interp='nearest',
                                                )

        ts.append(t*(mask>0))
        del tmp_view
        del mask

    # ts = np.array(ts)

    tmin = np.max(ts,0)
    for t in ts: d = t>0; tmin[d] = np.min([tmin[d],t[d]],0)
    del t,d
    tmin = ndimage.gaussian_filter(tmin,2)

    min_int = np.percentile(tmin.flatten()[1:-1:100],1)
    mean_int = np.mean(tmin.flatten()[1:-1:100])
    seg_level = min_int + (mean_int - min_int)*0.4
    seg = (tmin > seg_level).astype(np.uint16)
    seg = ImageArray(seg,spacing=ts[0].spacing,rotation=ts[0].rotation,origin=ts[0].origin)

    return seg


def get_lambda_weights(
                        views,
                        params,
                        stack_properties,
                        seg = None,
                        ):

    """
    weights are calculated before transforming
    """
    # stack_properties = calc_stack_properties_from_views_and_params(views,params,spacing)

    # save views for later

    # compress arrays in memory to save RAM in case of many zeros
    import bcolz

    ts = []
    for i in range(len(views)):
        print('transforming view %s' %i)
        t = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                views[i]+1,params[i], # adding one because of weights (taken away again further down)
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        # make sure that only those pixels are kept which are interpolated from 100% valid interpolation pixels
        tmp_view = ImageArray(views[i][:-1,:-1,:-1]+1,spacing=views[i].spacing,origin=views[i].origin+views[i].spacing/2.,rotation=views[i].rotation)

        mask = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                tmp_view, params[i],
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                interp='nearest',
                                                )
        # mask = sitk.GetArrayFromImage((mask>0).astype(np.uint16))
        # mask = sitk.BinaryErode(mask)
        # barr = bcolz.carray(t*(mask>0))
        barr = (t*(mask>0))
        del t
        ts.append(barr)
        del tmp_view
        del mask

    # ts = np.array(ts)

    if seg is None:
        tmin = np.max(ts,0)
        for t in ts: d = t>0; tmin[d] = np.min([tmin[d],t[d]],0)
        del t,d
        tmin = ndimage.gaussian_filter(tmin,2)
        min_int = np.percentile(tmin.flatten()[1:-1:100],1)
        mean_int = np.mean(tmin.flatten()[1:-1:100])
        seg_level = min_int + (mean_int - min_int)*0.4
        seg = (tmin > seg_level).astype(np.uint16)
        seg = ImageArray(seg,spacing=ts[0].spacing,rotation=ts[0].rotation,origin=ts[0].origin)
    else:
        print('fusion lambda: loading seg from %s' %seg)
        io_utils.process_input_element(seg)

    weights = []
    for i in range(len(views)):
        print('calculating weight %s' %i)
        inv_params = invert_params(params[i])
        tmp_weights_spacing = np.array(stack_properties['spacing']) * 4
        tmp_out_size = (views[i].spacing * np.array(views[i].shape) / tmp_weights_spacing).astype(np.int64)
        tmp_out_origin = views[i].origin
        pad = np.ones(3)*1
        tmp_out_size = (tmp_out_size + 2*pad).astype(np.int64)
        tmp_out_origin = tmp_out_origin - pad * tmp_weights_spacing
        bt = transform_stack_sitk(seg,inv_params,
                                                out_shape=tmp_out_size,
                                                out_spacing=tmp_weights_spacing,
                                                out_origin=tmp_out_origin,
                                                )
        bt = bt.astype(np.float32)

        bt = np.cumsum(bt[::-1],axis=0)[::-1]

        # orig is 0.1, 5
        # bt = 0.1 + np.exp(-5/np.max(bt)*bt)

        bt = 0.1 + np.exp(-5/np.max(bt)*bt)
        bt = transform_stack_sitk(bt,params[i],
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        barr = bcolz.carray(bt)
        del bt
        weights.append(barr)

    del seg

    # weights = [w**2 for w in weights]
    # zero weights where there's no signal
    # weights = [weights[i]*(ts[i]>0) for i in range(len(weights))]
    for i in range(len(weights)):
        # weights[i] = weights[i][:]**2 # from bcolz carray to np.ndarray
        # weights[i] = weights[i]*(ts[i]>0)
        # weights[i] = bcolz.carray(weights[i])
        w = weights[i]
        t = ts[i]
        # weights[i] = bcolz.eval('w**2 * (t>0)')
        weights[i] = bcolz.eval('w * (t>0)')


    # tifffile.imshow(np.array([t/500.,mask>0,bt]),vmin=0,vmax=1)
    # tifffile.imshow(np.array([ts[1]/500.,weights[1]]),vmin=0,vmax=1)

    # normalise weights
    # weight_sum = np.sum(weights,0)

    weight_sum = np.zeros_like(weights[0])
    for i in range(len(weights)):
        weight_sum += weights[i][:] # from bcolz carray to np.ndarray

    domain = weight_sum > 0
    for i in range(len(weights)):
        weights[i] = weights[i][:]
        weights[i][domain] = weights[i][domain] / weight_sum[domain]
        weights[i] = bcolz.carray(weights[i])

    return weights

def blur_view_in_view_space(view,
              p,
              orig_properties,
              stack_properties,
              sz,
              sxy,
              ):

    # print('blur view..')
    p = params_invert_coordinates(p)
    inv_p = invert_params(p)
    # print('transf md %s' %ip)
    # o = transform_stack_sitk(density,
    #                      p           = inv_p,
    #                      out_shape   = orig_prop_list[ip]['size'],
    #                      out_spacing = orig_prop_list[ip]['spacing'],
    #                      out_origin  = orig_prop_list[ip]['origin'],
    #                      interp      ='linear')
    # print('transform to view..')
    o = transformStack(
                         p          = inv_p,
                         stack      = view,
                         outShape   = orig_properties['size'][::-1],
                         outSpacing = orig_properties['spacing'][::-1],
                         outOrigin  = orig_properties['origin'][::-1],
                        # interp='bspline',
                       )
    # print('not blurring!')
    # print('Warning: hard coded blur')
    if sz:
        # o = (sitk.SmoothingRecursiveGaussian(o,[0.2,0.2,2]) + 0.5*sitk.SmoothingRecursiveGaussian(o,[2,2,7]))/1.5
        o = sitk.SmoothingRecursiveGaussian(o,[sxy,sxy,sz])
    # else:
        # print('not blurring! (not sz is True)')
    # print('transform to fused..')
    o = transformStack(
                         p          = p,
                         stack      = o,
                         outShape   = stack_properties['size'][::-1],
                         outSpacing = stack_properties['spacing'][::-1],
                         outOrigin  = stack_properties['origin'][::-1],
                        # interp='bspline',
                       )
    return o


def blur_view_in_target_space(view,
              p,
              orig_properties_unused,
              stack_properties,
              sz,
              sxy,
              ):
    """

    :param view: in target space
    :param p: normal view p
    :param stack_properties:
    :param sz:
    :param sxy:
    :return:
    """

    if not sz and not sxy:
        print('not blurring because of zero sigmas')
        return view

    print('blur view..')

    # construct psf with shape containing kernel radius three times
    psf_shape = (np.array([np.max([1,np.max([sz,sxy,sxy])*3])]) / stack_properties['spacing']).astype(np.int64)
    ## make shape odd
    psf_shape = np.array([ps+[1,0][int(ps%2)] for ps in psf_shape]).astype(np.int64)
    print('psf with sigma %s has shape %s' %([sxy,sxy,sz],list(psf_shape)))
    psf_orig = np.zeros(psf_shape,dtype=np.float32)
    psf_orig[psf_shape[0]//2,psf_shape[1]//2,psf_shape[2]//2]     = 1

    ## blur
    psf_orig = ndimage.gaussian_filter(psf_orig,np.array([sz,sxy,sxy])/stack_properties['spacing'])
    ## assign metadata
    psf_orig = ImageArray(psf_orig)
    psf_orig.spacing = stack_properties['spacing']
    psf_orig.origin = -stack_properties['spacing']*(psf_shape//2)
    # psf_orig_sitk = image_to_sitk(psf_orig)

    psf_stack_properties = dict()
    psf_stack_properties['spacing'] = stack_properties['spacing']
    psf_stack_properties['origin']  = psf_orig.origin
    # psf_stack_properties['origin']  = np.zeros(3)
    psf_stack_properties['size']    = psf_shape

    # print(psf_orig.get_info())
    # print(psf_stack_properties)

    # eliminate translation component from parameters
    tmpp = np.copy(p)
    tmpp[-3:] = 0

    psf_target = transform_stack_sitk(psf_orig, tmpp,
                                out_origin=psf_stack_properties['origin'],
                                out_shape=psf_stack_properties['size'],
                                out_spacing=psf_stack_properties['spacing'],
                                interp='linear')

    # normalise
    psf_target = psf_target / np.sum(psf_target)
    psf_target = psf_target.astype(np.float32)

    psf_target = image_to_sitk(psf_target)
    # conv = sitk.Convolution(sitk.Cast(view,sitk.sitkFloat32),psf_target)
    conv = sitk.FFTConvolution(sitk.Cast(view,sitk.sitkFloat32),psf_target)

    return conv

def density_to_multiview_data(
                              density,
                              params,
                              orig_prop_list,
                              stack_properties,
                              sz,
                              sxy,
                              blur_func,
                              ):
    """
    Takes a 3D image input, returns a stack of multiview data
    adapted from https://code.google.com/archive/p/iterative-fusion/
    """

    """
    Simulate the imaging process by applying multiple blurs
    """
    out = []
    for ip,p in enumerate(params):
        # print('gauss dm %s' %ip)
        # o = sitk.SmoothingRecursiveGaussian(density,sigmas[ip])
        o = blur_func(density,p,orig_prop_list[ip],stack_properties,sz,sxy)
        o = sitk.Cast(o, sitk.sitkFloat32)
        out.append(o)
    return out

def multiview_data_to_density(
                              multiview_data,
                              params,
                              orig_prop_list,
                              stack_properties,
                              sz,
                              sxy,
                              weights,
                              blur_func,
                              ):
    """
    The transpose of the density_to_multiview_data operation we perform above.
    adapted from https://code.google.com/archive/p/iterative-fusion/

    - multiply with DCT weights here
    """

    density = multiview_data[0]*0.
    density = sitk.Cast(density,sitk.sitkFloat32)
    # outs = multiview_data[0]*0.
    # outs = sitk.Cast(outs,sitk.sitkUInt16)
    for ip,p in enumerate(params):
        # print('gauss md %s' %ip)
        o = multiview_data[ip]
        # o = sitk.SmoothingRecursiveGaussian(multiview_data[ip],sigmas[ip])

        # smooth and resample in original view
        o = blur_func(o,p,orig_prop_list[ip],stack_properties,sz,sxy)

        o = sitk.Cast(o,sitk.sitkFloat32)

        if weights is not None:
            o = o*weights[ip]

        density += o

    density = sitk.Cast(density,sitk.sitkFloat32)
    return density

@io_decorator
def get_image_from_list_of_images(ims,ind):
    return ims[ind]

@io_decorator
def fuse_LR_with_weights(
        views,
        params,
        stack_properties,
        num_iterations = 25,
        sz = 4,
        sxy = 0.5,
        tol = 5e-5,
        weights = None,
        regularisation = False,
        blur_func = blur_view_in_view_space,
        orig_prop_list = None,
        views_in_target_space = True,
):
    """
    Combine
    - LR multiview fusion
      (adapted from python code given in https://code.google.com/archive/p/iterative-fusion/
       from publication https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3986040/)
    - DCT weights

    This addresses the problems that
    1) multi-view deconvolution is highly dependent on high precision registration
    between views. However, an affine transformation is often not enough due to
    optical aberrations and results in poor overlap.
    2) due to scattering, the psf strongly varies within each view

    In the case of highly scattering samples, FFT+elastix typically results in good
    registration accuracy in regions of good image quality (those with small psfs
    and short optical paths through the sample). These regions are found using a DCT
    quality measure and weighted accordingly. Therefore, to reduce the contribution
    to multi-view LR of unwanted regions in the individual views, the weights are
    applied in each iteration during convolution with the psf.

    Adaptations and details:
    - convolve views in original space
     - recapitulates imaging process and trivially deals with view parameters
     - allows for iterative raw data reconstruction without deconvolution
     - disadvantage: slow in current implementation
    - apply DCT weights in each blurring iteration to account for strong scattering
    - simulate convolution by psf with gaussian blurring
    - TV regularisation not working yet, to be optimised (Multiview Deblurring for 3-D Images from
Light-Sheet-Based Fluorescence Microscopy, https://ieeexplore.ieee.org/document/6112225)

    Interesting case: sz,sxy=0
    - formally no deconvolution but iterative multi-view raw data reconstruction

    works well:
    - sz6 it 10, some rings
    - sz5 it 20, looks good (good compromise between sz and its)
    - sz4 it 30, good and no rings

    :param views: original views
    :param params: parameters mapping views into target space
    :param stack_properties: properties of target space
    :param num_iterations: max number of deconvolution iterations
    :param sz: sigma z
    :param sxy: sigma xy
    :param tol: convergence threshold
    :return:
    """

    # get orig properties
    # zfactor = float(1)

    if orig_prop_list is None and not views_in_target_space:
        orig_prop_list = []
        for ip in range(len(params)):
            prop_dict = dict()
            prop_dict['size'] = views[ip].shape
            prop_dict['origin'] = views[ip].origin
            prop_dict['spacing'] = views[ip].spacing
            orig_prop_list.append(prop_dict)

    # weights = weight_func(
    #                    views,
    #                    params,
    #                    stack_properties,
    #                    **(weight_func_kwargs or {})
    # )

    # if weight_func == get_weights_dct:
    #     weights = get_weights_dct(
    #                        views,
    #                        params,
    #                        stack_properties,
    #                        # size=50,
    #                        size=None,
    #                        # max_kernel=10,
    #                        max_kernel=None,
    #                        # gaussian_kernel=10)
    #                        gaussian_kernel=None)
    # else:
    #     weights = weight_func(
    #                        views,
    #                        params,
    #                        stack_properties,
    #     )

    # tmp_fused = fuse_views_weights(views, params, stack_properties, weights = weights)

    weights = list(weights)
    for iw in range(len(weights)):
        tmp = sitk.GetImageFromArray(weights[iw])
        tmp.SetSpacing(stack_properties['spacing'][::-1])
        tmp.SetOrigin(stack_properties['origin'][::-1])
        tmp = sitk.Cast(tmp,sitk.sitkFloat32)
        weights[iw] = tmp

    if not views_in_target_space:
        nviews = []
        for iview,view in enumerate(views):
            tmp = transform_stack_sitk(view,params[iview],
                                   out_origin=stack_properties['origin'],
                                   out_shape=stack_properties['size'],
                                   out_spacing=stack_properties['spacing'])

            # make sure to take only interpolations with full data
            # tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)

            # tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
            # mask = transform_stack_sitk(tmp_view,params[iview],
            #                        out_origin=stack_properties['origin'],
            #                        out_shape=stack_properties['size'],
            #                        out_spacing=stack_properties['spacing'],
            #                         interp='nearest')
            # # tmp[tmp==0] = 10
            # mask = mask > 0
            # nviews.append(tmp*(mask))

            nviews.append(tmp)
            # masks.append(mask)
        views = nviews

    # convert to sitk images
    for ip,p in enumerate(params):
        tmp = sitk.GetImageFromArray(views[ip])
        tmp = sitk.Cast(tmp,sitk.sitkFloat32)
        tmp.SetSpacing(views[ip].spacing[::-1])
        tmp.SetOrigin(views[ip].origin[::-1])
        # debug crop
        # tmp = tmp[:,500:550,:]
        views[ip] = tmp

    noisy_multiview_data = views

    """
    Time for deconvolution!!!
    """

    def calc_imsum(im):
        tmp = sitk.Abs(im)
        for d in range(3):
            tmp = sitk.SumProjection(tmp, d)  # [0]
        return tmp[0, 0, 0]
    # estimate = sitk.Image(
    #     int(stack_properties['size'][2]),
    #     int(stack_properties['size'][1]),
    #     int(stack_properties['size'][0]),
    #     sitk.sitkFloat32,
    # )
    # estimate *= 0.
    # estimate += 1.
    print('WARNING: Initialising with fused views')
    estimate = weights[0]*views[0]
    for i in range(1,len(params)):
        estimate += weights[i]*views[i]
    estimate = sitk.Cast(estimate,sitk.sitkFloat32)

    estimate.SetSpacing(stack_properties['spacing'][::-1])
    estimate.SetOrigin(stack_properties['origin'][::-1])

    # return sitk_to_image(estimate),tmp_fused

    # estimate = np.ones(views[0].shape, dtype=np.float64)
    # expected_data = np.zeros_like(noisy_multiview_data)
    # correction_factor = np.zeros_like(estimate)
    # history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
    # history[0, :, :, :] = estimate
    # for i in range(num_iterations):

    curr_imsum = calc_imsum(estimate)

    i = 0
    while 1:
        print("Iteration", i)

        """
        Construct the expected data from the estimate
        """
        # print("Constructing estimated data...")
        # print('WARNING: saving each iteration')
        # sitk.WriteImage(sitk.Cast(estimate,sitk.sitkUInt16),'/data/malbert/regtest/ills/ills_gw_reg2_fus1/iter%03d.mhd' %i)

        expected_data = density_to_multiview_data(
              estimate,
              params,
              orig_prop_list,
              stack_properties,
              sz,
              sxy,
              blur_func,
        )
        # multiview_data_to_visualization(expected_data, outfile='expected_data.tif')
        "Done constructing."
        """
        Take the ratio between the measured data and the expected data.
        Store this ratio in 'expected_data'
        """
        for ip in range(len(params)):
            expected_data[ip] += 1e-6 #Don't want to divide by 0!
        expected_data = [sitk.Cast(noisy_multiview_data[ip] / expected_data[ip],sitk.sitkFloat32) for ip in range(len(params))]

        # multiply with mask to reduce border artifacts
        for ip in range(len(params)):
            expected_data[ip] = expected_data[ip] * sitk.Cast(weights[ip]>0,sitk.sitkFloat32)
        """
        Apply the transpose of the expected data operation to the correction factor
        """
        correction_factor = multiview_data_to_density(
            expected_data,
            params,
            orig_prop_list,
            stack_properties,
            sz,
            sxy,
            weights,
            blur_func,
        )#, out=correction_factor)

        """
        Multiply the old estimate by the correction factor to get the new estimate
        """
        if regularisation:
            print('WARNING: regularising')
            correction_factor = sitk.Cast(correction_factor / get_LR_regularisation(estimate,len(params)), sitk.sitkFloat32)

        estimate = estimate * correction_factor

        estimate = estimate * sitk.Cast(estimate<2**16,sitk.sitkFloat32)
        estimate = estimate * sitk.Cast(estimate>0,    sitk.sitkFloat32)

        # if num_iterations < 1:
        new_imsum = calc_imsum(estimate)
        conv = np.abs(1-new_imsum/curr_imsum)
        print('convergence: %s' %conv)

        if conv < tol and i>=10: break
        if i >= num_iterations-1: break

        curr_imsum = new_imsum
        i += 1

        """
        Update the history
        """
    print("Done deconvolving")

    estimate = ImageArray(sitk.GetArrayFromImage(estimate).astype(np.uint16),spacing=np.array(estimate.GetSpacing())[::-1],origin=np.array(estimate.GetOrigin())[::-1])

    return estimate

# @io_decorator
# def fuse_LR_with_weights_dct(
#         views,
#         params,
#         stack_properties,
#         num_iterations = 25,
#         sz = 4,
#         sxy = 0.5,
#         tol = 5e-5,
#         weight_func = get_weights_dct,
#         regularisation = False,
#         blur_func = blur_view_in_view_space,
#         weight_func_kwargs = None,
# ):
#     """
#     Combine
#     - LR multiview fusion
#       (adapted from python code given in https://code.google.com/archive/p/iterative-fusion/
#        from publication https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3986040/)
#     - DCT weights
#
#     This addresses the problems that
#     1) multi-view deconvolution is highly dependent on high precision registration
#     between views. However, an affine transformation is often not enough due to
#     optical aberrations and results in poor overlap.
#     2) due to scattering, the psf strongly varies within each view
#
#     In the case of highly scattering samples, FFT+elastix typically results in good
#     registration accuracy in regions of good image quality (those with small psfs
#     and short optical paths through the sample). These regions are found using a DCT
#     quality measure and weighted accordingly. Therefore, to reduce the contribution
#     to multi-view LR of unwanted regions in the individual views, the weights are
#     applied in each iteration during convolution with the psf.
#
#     Adaptations and details:
#     - convolve views in original space
#      - recapitulates imaging process and trivially deals with view parameters
#      - allows for iterative raw data reconstruction without deconvolution
#      - disadvantage: slow in current implementation
#     - apply DCT weights in each blurring iteration to account for strong scattering
#     - simulate convolution by psf with gaussian blurring
#     - TV regularisation not working yet, to be optimised (Multiview Deblurring for 3-D Images from
# Light-Sheet-Based Fluorescence Microscopy, https://ieeexplore.ieee.org/document/6112225)
#
#     Interesting case: sz,sxy=0
#     - formally no deconvolution but iterative multi-view raw data reconstruction
#
#     works well:
#     - sz6 it 10, some rings
#     - sz5 it 20, looks good (good compromise between sz and its)
#     - sz4 it 30, good and no rings
#
#     :param views: original views
#     :param params: parameters mapping views into target space
#     :param stack_properties: properties of target space
#     :param num_iterations: max number of deconvolution iterations
#     :param sz: sigma z
#     :param sxy: sigma xy
#     :param tol: convergence threshold
#     :return:
#     """
#
#     # get orig properties
#     # zfactor = float(1)
#     orig_prop_list = []
#     for ip in range(len(params)):
#         prop_dict = dict()
#         prop_dict['size'] = views[ip].shape
#         prop_dict['origin'] = views[ip].origin
#         prop_dict['spacing'] = views[ip].spacing
#         orig_prop_list.append(prop_dict)
#
#     weights = weight_func(
#                        views,
#                        params,
#                        stack_properties,
#                        **(weight_func_kwargs or {})
#     )
#
#     # if weight_func == get_weights_dct:
#     #     weights = get_weights_dct(
#     #                        views,
#     #                        params,
#     #                        stack_properties,
#     #                        # size=50,
#     #                        size=None,
#     #                        # max_kernel=10,
#     #                        max_kernel=None,
#     #                        # gaussian_kernel=10)
#     #                        gaussian_kernel=None)
#     # else:
#     #     weights = weight_func(
#     #                        views,
#     #                        params,
#     #                        stack_properties,
#     #     )
#
#     # tmp_fused = fuse_views_weights(views, params, stack_properties, weights = weights)
#
#     weights = list(weights)
#     for iw in range(len(weights)):
#         tmp = sitk.GetImageFromArray(weights[iw])
#         tmp.SetSpacing(stack_properties['spacing'][::-1])
#         tmp.SetOrigin(stack_properties['origin'][::-1])
#         tmp = sitk.Cast(tmp,sitk.sitkFloat32)
#         weights[iw] = tmp
#
#     nviews = []
#     for iview,view in enumerate(views):
#         tmp = transform_stack_sitk(view,params[iview],
#                                out_origin=stack_properties['origin'],
#                                out_shape=stack_properties['size'],
#                                out_spacing=stack_properties['spacing'])
#
#         # make sure to take only interpolations with full data
#         # tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
#
#         # tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
#         # mask = transform_stack_sitk(tmp_view,params[iview],
#         #                        out_origin=stack_properties['origin'],
#         #                        out_shape=stack_properties['size'],
#         #                        out_spacing=stack_properties['spacing'],
#         #                         interp='nearest')
#         # # tmp[tmp==0] = 10
#         # mask = mask > 0
#         # nviews.append(tmp*(mask))
#
#         nviews.append(tmp)
#         # masks.append(mask)
#     views = nviews
#
#     # convert to sitk images
#     for ip,p in enumerate(params):
#         tmp = sitk.GetImageFromArray(views[ip])
#         tmp = sitk.Cast(tmp,sitk.sitkFloat32)
#         tmp.SetSpacing(views[ip].spacing[::-1])
#         tmp.SetOrigin(views[ip].origin[::-1])
#         # debug crop
#         # tmp = tmp[:,500:550,:]
#         views[ip] = tmp
#
#     noisy_multiview_data = views
#
#     """
#     Time for deconvolution!!!
#     """
#
#     def calc_imsum(im):
#         tmp = sitk.Abs(im)
#         for d in range(3):
#             tmp = sitk.SumProjection(tmp, d)  # [0]
#         return tmp[0, 0, 0]
#     # estimate = sitk.Image(
#     #     int(stack_properties['size'][2]),
#     #     int(stack_properties['size'][1]),
#     #     int(stack_properties['size'][0]),
#     #     sitk.sitkFloat32,
#     # )
#     # estimate *= 0.
#     # estimate += 1.
#     print('WARNING: Initialising with fused views')
#     estimate = weights[0]*views[0]
#     for i in range(1,len(params)):
#         estimate += weights[i]*views[i]
#     estimate = sitk.Cast(estimate,sitk.sitkFloat32)
#
#     estimate.SetSpacing(stack_properties['spacing'][::-1])
#     estimate.SetOrigin(stack_properties['origin'][::-1])
#
#     # return sitk_to_image(estimate),tmp_fused
#
#     # estimate = np.ones(views[0].shape, dtype=np.float64)
#     # expected_data = np.zeros_like(noisy_multiview_data)
#     # correction_factor = np.zeros_like(estimate)
#     # history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
#     # history[0, :, :, :] = estimate
#     # for i in range(num_iterations):
#
#     curr_imsum = calc_imsum(estimate)
#
#     i = 0
#     while 1:
#         print("Iteration", i)
#
#         """
#         Construct the expected data from the estimate
#         """
#         # print("Constructing estimated data...")
#         # print('WARNING: saving each iteration')
#         # sitk.WriteImage(sitk.Cast(estimate,sitk.sitkUInt16),'/data/malbert/regtest/ills/ills_gw_reg2_fus1/iter%03d.mhd' %i)
#
#         expected_data = density_to_multiview_data(
#               estimate,
#               params,
#               orig_prop_list,
#               stack_properties,
#               sz,
#               sxy,
#               blur_func,
#         )
#         # multiview_data_to_visualization(expected_data, outfile='expected_data.tif')
#         "Done constructing."
#         """
#         Take the ratio between the measured data and the expected data.
#         Store this ratio in 'expected_data'
#         """
#         for ip in range(len(params)):
#             expected_data[ip] += 1e-6 #Don't want to divide by 0!
#         expected_data = [sitk.Cast(noisy_multiview_data[ip] / expected_data[ip],sitk.sitkFloat32) for ip in range(len(params))]
#
#         # multiply with mask to reduce border artifacts
#         for ip in range(len(params)):
#             expected_data[ip] = expected_data[ip] * sitk.Cast(weights[ip]>0,sitk.sitkFloat32)
#         """
#         Apply the transpose of the expected data operation to the correction factor
#         """
#         correction_factor = multiview_data_to_density(
#             expected_data,
#             params,
#             orig_prop_list,
#             stack_properties,
#             sz,
#             sxy,
#             weights,
#             blur_func,
#         )#, out=correction_factor)
##
#         """
#         Multiply the old estimate by the correction factor to get the new estimate
#         """
#         if regularisation:
#             print('WARNING: regularising')
#             correction_factor = sitk.Cast(correction_factor / get_LR_regularisation(estimate,len(params)), sitk.sitkFloat32)
#
#         estimate = estimate * correction_factor
#
#         estimate = estimate * sitk.Cast(estimate<2**16,sitk.sitkFloat32)
#         estimate = estimate * sitk.Cast(estimate>0,    sitk.sitkFloat32)
#
#         # if num_iterations < 1:
#         new_imsum = calc_imsum(estimate)
#         conv = np.abs(1-new_imsum/curr_imsum)
#         print('convergence: %s' %conv)
#
#         if conv < tol and i>=10: break
#         if i >= num_iterations-1: break
#
#         curr_imsum = new_imsum
#         i += 1
#
#         """
#         Update the history
#         """
#     print("Done deconvolving")
#
#     estimate = ImageArray(sitk.GetArrayFromImage(estimate).astype(np.uint16),spacing=np.array(estimate.GetSpacing())[::-1],origin=np.array(estimate.GetOrigin())[::-1])
#
#     return estimate

def get_LR_regularisation(u,nviews,l=0.005):
    """
    TV regularisation as in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6112225
    not working well, seems to amplify noise for real data
    works okayish for synthetic data (tested on blurred box)
    :param u:
    :param nviews:
    :param l:
    :return:
    """
    # l = 0.02
    # u = sitk.SmoothingRecursiveGaussian(u,u.GetSpacing()[0]*3)
    gm  = sitk.GradientMagnitude(u,useImageSpacing=False)
    gm  = gm + sitk.Cast(gm == 0,sitk.sitkFloat32)
    # gm = gm + sitk.Cast(gm < 1e-2, sitk.sitkFloat32)
    lapl = sitk.Laplacian(u,useImageSpacing=False)
    # lapl = sitk.Clamp(lapl, sitk.sitkFloat32, -2, 2)
    rf = 1 - l/nviews * lapl/gm
    rf = sitk.Cast(rf,sitk.sitkFloat32)
    # rf = sitk.Clamp(rf, sitk.sitkFloat32, 1e-9, 1e9)

    return rf


@io_decorator
def fuse_views_lambda(
                        views,
                        params,
                        # spacing = [1.,1,1],
                        stack_properties,
                        seg = None,
                        ):

    """
    weights are calculated before transforming
    """
    # stack_properties = calc_stack_properties_from_views_and_params(views,params,spacing)

    # save views for later

    # compress arrays in memory to save RAM in case of many zeros
    import bcolz

    ts = []
    for i in range(len(views)):
        print('transforming view %s' %i)
        t = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                views[i]+1,params[i], # adding one because of weights (taken away again further down)
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        # take away border planes for producing mask for avoiding edge effects
        # views[i][0] = 0
        # views[i][-1] = 0
        # views[i][:,0] = 0
        # views[i][:,-1] = 0
        # views[i][:,:,0] = 0
        # views[i][:,:,-1] = 0

        # make sure that only those pixels are kept which are interpolated from 100% valid interpolation pixels
        tmp_view = ImageArray(views[i][:-1,:-1,:-1]+1,spacing=views[i].spacing,origin=views[i].origin+views[i].spacing/2.,rotation=views[i].rotation)

        mask = transform_stack_sitk(
                                                # vsr[i],final_params[i],
                                                tmp_view, params[i],
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                interp='nearest',
                                                )
        # mask = sitk.GetArrayFromImage((mask>0).astype(np.uint16))
        # mask = sitk.BinaryErode(mask)
        barr = bcolz.carray(t*(mask>0))
        del t
        ts.append(barr)
        del tmp_view
        del mask

    # ts = np.array(ts)

    if seg is None:
        tmin = np.max(ts,0)
        for t in ts: d = t>0; tmin[d] = np.min([tmin[d],t[d]],0)
        del t,d
        tmin = ndimage.gaussian_filter(tmin,2)
        min_int = np.percentile(tmin.flatten()[1:-1:100],1)
        mean_int = np.mean(tmin.flatten()[1:-1:100])
        seg_level = min_int + (mean_int - min_int)*0.4
        seg = (tmin > seg_level).astype(np.uint16)
        seg = ImageArray(seg,spacing=ts[0].spacing,rotation=ts[0].rotation,origin=ts[0].origin)
    else:
        print('fusion lambda: loading seg from %s' %seg)
        io_utils.process_input_element(seg)

    # # calculate additive fusion weights
    # tmax = np.max(ts,0)
    # # masking taken from klb paper, processTimepoint.m
    # tmax = ndimage.gaussian_filter(tmax,2)
    # min_int = np.percentile(tmax.flatten()[1:-1:100],1)
    # mean_int = np.mean(tmax.flatten()[1:-1:100])
    # seg_level = min_int + (mean_int - min_int)*0.4
    # seg = (tmax > seg_level).astype(np.uint16)
    # seg = ImageArray(seg,spacing=ts[0].spacing,rotation=ts[0].rotation,origin=ts[0].origin)

    weights = []
    for i in range(len(views)):
        print('calculating weight %s' %i)
        inv_params = invert_params(params[i])
        tmp_weights_spacing = np.array(stack_properties['spacing']) * 4
        tmp_out_size = (views[i].spacing * np.array(views[i].shape) / tmp_weights_spacing).astype(np.int64)
        tmp_out_origin = views[i].origin
        pad = np.ones(3)*1
        tmp_out_size = (tmp_out_size + 2*pad).astype(np.int64)
        tmp_out_origin = tmp_out_origin - pad * tmp_weights_spacing
        bt = transform_stack_sitk(seg,inv_params,
                                                out_shape=tmp_out_size,
                                                out_spacing=tmp_weights_spacing,
                                                out_origin=tmp_out_origin,
                                                )
        bt = bt.astype(np.float32)

        bt = np.cumsum(bt[::-1],axis=0)[::-1]

        # orig is 0.1, 5
        # bt = 0.1 + np.exp(-5/np.max(bt)*bt)

        bt = 0.1 + np.exp(-5/np.max(bt)*bt)
        bt = transform_stack_sitk(bt,params[i],
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        barr = bcolz.carray(bt)
        del bt
        weights.append(barr)

    del seg

    # weights = [w**2 for w in weights]

    # zero weights where there's no signal
    # weights = [weights[i]*(ts[i]>0) for i in range(len(weights))]
    for i in range(len(weights)):
        # weights[i] = weights[i][:]**2 # from bcolz carray to np.ndarray
        # weights[i] = weights[i]*(ts[i]>0)
        # weights[i] = bcolz.carray(weights[i])
        w = weights[i]
        t = ts[i]
        weights[i] = bcolz.eval('w**2 * (t>0)')

    # tifffile.imshow(np.array([t/500.,mask>0,bt]),vmin=0,vmax=1)
    # tifffile.imshow(np.array([ts[1]/500.,weights[1]]),vmin=0,vmax=1)

    # normalise weights
    # weight_sum = np.sum(weights,0)

    weight_sum = np.zeros_like(weights[0])
    for i in range(len(weights)):
        weight_sum += weights[i][:] # from bcolz carray to np.ndarray

    domain = weight_sum > 0
    for i in range(len(weights)):
        weights[i] = weights[i][:]
        weights[i][domain] = weights[i][domain] / weight_sum[domain]
        weights[i] = bcolz.carray(weights[i])

    # perform final fusion
    f = np.zeros_like(weight_sum)
    for i in range(len(weights)):
        f += weights[i][:] * (ts[i][:]-1) # taking away one that was added at the beginning
    f = f.astype(np.uint16)

    # return f,weights,seg
    return ImageArray(f,spacing=stack_properties['spacing'])

def image_to_sitk(im):
    sim = sitk.GetImageFromArray(im)
    sim.SetOrigin(im.origin[::-1])
    sim.SetSpacing(im.spacing[::-1])
    return sim

def sitk_to_image(sim):
    im = sitk.GetArrayFromImage(sim)
    im = ImageArray(im,spacing=np.array(sim.GetSpacing())[::-1],origin=np.array(sim.GetOrigin())[::-1])
    return im


@io_decorator
def fuse_dct(views,params,stack_properties):
    weights = get_weights_dct(views,params,stack_properties)
    return fuse_views_weights(views,params,stack_properties,weights=weights)

def transformStack(p,stack,outShape=None,outSpacing=None,outOrigin=None,interp='linear'):
    # can handle composite transformations (len(p)%12)
    # 20140326: added outOrigin option

    # handle arguments
    transf = sitk.Transform(3,sitk.sitkIdentity) #!!!
    if not (p is None):
        for i in range(len(p)//12):
            transf.AddTransform(sitk.Transform(3,sitk.sitkAffine))
        #p = np.array(p)
        #p = np.concatenate([p[:9].reshape((3,3))[::-1,::-1].flatten(),p[9:][::-1]])
        transf.SetParameters(np.array(p,dtype=np.float64))
    if outShape is None: shape = stack.GetSize()
    else:
        shape = np.ceil(np.array(outShape))
        shape = [int(i) for i in shape]
    if outSpacing is None: outSpacing = stack.GetSpacing()
    else: outSpacing = np.array(outSpacing)
    outSpacing = np.array(outSpacing).astype(np.float)
    if outOrigin is None: outOrigin = stack.GetOrigin()
    else: outOrigin = np.array(outOrigin)

    # don't do anything if stack nothing is to be done
    def vectors_are_same(v1,v2):
        # return np.sum(np.abs(np.array(v1).astype(np.uint8) - np.array(v2).astype(np.uint8))) == 0
        return np.allclose(v1,v2)

    p_is_identity = vectors_are_same(p,[1,0,0,0,1,0,0,0,1,0,0,0])
    out_shape_same = vectors_are_same(outShape,stack.GetSize())
    out_spacing_same = vectors_are_same(outSpacing,stack.GetSpacing())
    out_origin_same = vectors_are_same(outOrigin,stack.GetOrigin())
    if vectors_are_same([p_is_identity,out_shape_same,out_spacing_same,out_origin_same],[True for i in range(4)]):
        return stack

    numpyarray = False
    if type(stack)==np.ndarray:
        numpyarray = True
        stack = sitk.GetImageFromArray(stack)

    if interp == 'bspline':
        interpolator = sitk.sitkBSpline
    elif interp == 'linear':
        interpolator = sitk.sitkLinear
    elif interp == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    elif interp == 'gaussian':
        # addresses the problem that when downsampling, pixel information is disregarded
        # if new spacing is larger, smooth input with new 0.8*spacing
        # see https://www.insight-journal.org/browse/publication/705
        # implementation from publication does not consider spacing or is strangely implemented
        # therefore manually here
        if np.any(np.array(outSpacing) > np.array(stack.GetSpacing())):
            stack = sitk.SmoothingRecursiveGaussian(stack,0.8*np.array(outSpacing))
        interpolator = sitk.sitkLinear
    else:
        interpolator = interp

    orig_sitk_dtype = stack.GetPixelID()
    stack = sitk.Cast(stack,sitk.sitkFloat32) # avoid overflow
    stack = sitk.Resample(stack,shape,transf,interpolator,outOrigin,outSpacing)
    stack = sitk.Clamp(stack,orig_sitk_dtype) # avoid overflow
    if numpyarray:
        stack = sitk.GetArrayFromImage(stack)

    #

    return stack


"""(Transform "AffineTransform")
(NumberOfParameters 12)

(HowToCombineTransforms "Compose")

(InitialTransformParametersFileName "NoInitialTransform")

// Image specific
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "short")
(MovingInternalImagePixelType "short")
//(UseDirectionCosines "false")

(CenterOfRotationPoint 0 0 0)
"""

def get_t0(fixed,moving):

    """
    estimate t0 and crop images to intersection in y
    :param fixed:
    :param moving:
    :return:
    """

    lower_y0 = fixed.origin[1]
    upper_y0 = fixed.origin[1] + fixed.shape[1]*fixed.spacing[1]

    lower_y1 = moving.origin[1]
    upper_y1 = moving.origin[1] + moving.shape[1]*moving.spacing[1]

    lower_overlap = np.max([lower_y0,lower_y1])
    upper_overlap = np.min([upper_y0,upper_y1])

    yl0 = int((lower_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yu0 = int((upper_overlap - lower_y0) / (upper_y0-lower_y0) * fixed.shape[1])
    yl1 = int((lower_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])
    yu1 = int((upper_overlap - lower_y1) / (upper_y1-lower_y1) * moving.shape[1])

    # images can have different overlaps because of rounding to integer

    origin_overlap0 = np.zeros(3)
    origin_overlap1 = np.zeros(3)

    origin_overlap0[:] = fixed.origin
    origin_overlap1[:] = moving.origin

    origin_overlap0[1] = lower_y0 + yl0 * fixed.spacing[1]
    origin_overlap1[1] = lower_y1 + yl1 * moving.spacing[1]

    # c0 = clahe(fixed,10,clip_limit=0.02)
    # c1 = clahe(moving,10,clip_limit=0.02)

    static = ImageArray(fixed[:,yl0:yu0,:],spacing=fixed.spacing,origin=origin_overlap0)
    mov = ImageArray(moving[:,yl1:yu1,:],spacing=moving.spacing,origin=origin_overlap1)

    m0 = get_mask_using_otsu(static)
    m1 = get_mask_using_otsu(mov)

    mean0 = ndimage.center_of_mass(m0) * fixed.spacing + origin_overlap0
    mean1 = ndimage.center_of_mass(m1) * moving.spacing + origin_overlap1

    print(mean0,mean1)

    matrix = mv_utils.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    # matrix = mv_utils.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
    t0 = np.append(matrix[:3, :3].flatten(), matrix[:3, 3])

    offset = mean1 - np.dot(matrix[:3,:3],mean0)
    t0[9:] = offset
    t0[10] = 0

    return t0

params_translation = """
(SubtractMean "false")
(AutomaticParameterEstimation "false")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "false")
//(Scales 10000 10000 10000 10000 10000 10000 1 1 1)
(AutomaticTransformInitializationMethod "CenterOfGravity")
(CheckNumberOfSamples "true")
(DefaultPixelValue 0.000000)
(FinalBSplineInterpolationOrder 1.000000)
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
//(NumberOfResolutions 4.000000)
//(ImagePyramidSchedule 16 16 16  8.000000 8.000000 8.000000 4.000000 4.000000 4.000000 2.000000 2.000000 2.000000)
//(ImageSampler "RandomCoordinate")
(ImageSampler "Full")
(Interpolator "LinearInterpolator")
//(MaximumNumberOfIterations 200.000000 200.000000 200.000000 200.000000)
//(MaximumNumberOfIterations 300.000000 100.000000 50.000000 20.000000)
(MaximumNumberOfSamplingAttempts 1.000000)
//(Metric "AdvancedMattesMutualInformation")
(Metric "AdvancedNormalizedCorrelation")

(NewSamplesEveryIteration "true")
(NumberOfSamplesForExactGradient 4096.000000)
(NumberOfSpatialSamples 2048.000000)
//(Optimizer "AdaptiveStochasticGradientDescent")
(Optimizer "QuasiNewtonLBFGS")
(GradientMagnitudeTolerance 1e-8)
(Registration "MultiResolutionRegistration")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ResultImageFormat "mhd")
(Transform "TranslationTransform")
(WriteIterationInfo "false")
(WriteResultImage "false")
(RequiredRatioOfValidSamples 0.02)
"""

params_rotation = """
(SubtractMean "false")
(AutomaticParameterEstimation "false")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "false")
//(Scales 10000 10000 10000 10000 10000 10000 1 1 1)
(AutomaticTransformInitializationMethod "CenterOfGravity")
(CheckNumberOfSamples "true")
(DefaultPixelValue 0.000000)
(FinalBSplineInterpolationOrder 1.000000)
(FixedImagePyramid "FixedRecursiveImagePyramid")
//(NumberOfResolutions 4.000000)
//(ImagePyramidSchedule 16 16 16  8.000000 8.000000 8.000000 4.000000 4.000000 4.000000 2.000000 2.000000 2.000000)
//(ImageSampler "RandomCoordinate")
(ImageSampler "Full")
(Interpolator "LinearInterpolator")
//(MaximumNumberOfIterations 200.000000 200.000000 200.000000 200.000000)
//(MaximumNumberOfIterations 300.000000 100.000000 50.000000 20.000000)
(MaximumNumberOfSamplingAttempts 1.000000)
(Metric "AdvancedNormalizedCorrelation")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(NewSamplesEveryIteration "true")
(NumberOfSamplesForExactGradient 4096.000000)
(NumberOfSpatialSamples 2048.000000)
//(Optimizer "AdaptiveStochasticGradientDescent")
(Optimizer "QuasiNewtonLBFGS")
(GradientMagnitudeTolerance 1e-8)
(Registration "MultiResolutionRegistration")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ResultImageFormat "mhd")
(Transform "EulerTransform")
(WriteIterationInfo "false")
(WriteResultImage "false")
(RequiredRatioOfValidSamples 0.02)
"""

params_similarity = """
(SubtractMean "false")
(AutomaticParameterEstimation "false")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "false")
//(Scales 10000 10000 10000 10000 10000 10000 1 1 1)
(AutomaticTransformInitializationMethod "CenterOfGravity")
(CheckNumberOfSamples "true")
(DefaultPixelValue 0.000000)
(FinalBSplineInterpolationOrder 1.000000)
(FixedImagePyramid "FixedRecursiveImagePyramid")
(NumberOfResolutions 4.000000)
(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1)
//(ImageSampler "RandomCoordinate")
(ImageSampler "Full")
(Interpolator "LinearInterpolator")
//(MaximumNumberOfIterations 200.000000 200.000000 200.000000 200.000000)
//(MaximumNumberOfIterations 300.000000 100.000000 50.000000 20.000000)
(MaximumNumberOfSamplingAttempts 1.000000)
(Metric "AdvancedNormalizedCorrelation")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(NewSamplesEveryIteration "true")
(NumberOfSamplesForExactGradient 4096.000000)
(NumberOfSpatialSamples 2048.000000)
//(Optimizer "AdaptiveStochasticGradientDescent")
(Optimizer "QuasiNewtonLBFGS")
(GradientMagnitudeTolerance 1e-7)
(Registration "MultiResolutionRegistration")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ResultImageFormat "mhd")
(Transform "SimilarityTransform")
(WriteIterationInfo "false")
(WriteResultImage "false")
(RequiredRatioOfValidSamples 0.02)
"""

params_affine = """
(SubtractMean "false")
(AutomaticParameterEstimation "false")
(AutomaticScalesEstimation "true")
//(Scales 10000 10000000 10000000 10000000 10000 10000000 10000000 10000000 10000 1 1 1)
(AutomaticTransformInitialization "false")
(AutomaticTransformInitializationMethod "CenterOfGravity")
(CheckNumberOfSamples "true")
(DefaultPixelValue 0.000000)
(FinalBSplineInterpolationOrder 1.000000)
(FixedImagePyramid "FixedRecursiveImagePyramid")
//(ImagePyramidSchedule 16 16 16  8.000000 8.000000 8.000000 4.000000 4.000000 4.000000 2.000000 2.000000 2.000000)
//(ImageSampler "RandomCoordinate")
(ImageSampler "Full")
(Interpolator "LinearInterpolator")
//(MaximumNumberOfIterations 200.000000 200.000000 200.000000 200.000000)
//(MaximumNumberOfIterations 5000.000000 2000.000000 1000.000000 200.000000)
//(MaximumNumberOfIterations 300.000000 100.000000 50.000000 20.000000)
(MaximumNumberOfSamplingAttempts 1.000000)
(Metric "AdvancedNormalizedCorrelation")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(NewSamplesEveryIteration "true")
(NumberOfSamplesForExactGradient 4096.000000)
(NumberOfSpatialSamples 2048.000000)
//(Optimizer "AdaptiveStochasticGradientDescent")
(Optimizer "QuasiNewtonLBFGS")
(GradientMagnitudeTolerance 1e-8)
(Registration "MultiResolutionRegistration")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ResultImageFormat "mhd")
(Transform "AffineTransform")
(WriteIterationInfo "false")
(WriteResultImage "false")
(RequiredRatioOfValidSamples 0.02)
//(NumberOfResolutions %(number_of_resolutions))
"""

params_bspline = """
(SubtractMean "false")
(AutomaticParameterEstimation "false")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "false")
(CheckNumberOfSamples "true")
(DefaultPixelValue 0.000000)
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3.000000)
(FixedImagePyramid "FixedRecursiveImagePyramid")
//(ImagePyramidSchedule 16 16 16  8.000000 8.000000 8.000000 4.000000 4.000000 4.000000 2.000000 2.000000 2.000000)
//(ImageSampler "RandomCoordinate")
(ImageSampler "Full")
(Interpolator "LinearInterpolator")
//(MaximumNumberOfIterations 200.000000 200.000000 200.000000 200.000000)
//(MaximumNumberOfIterations 5000.000000 2000.000000 1000.000000 200.000000)
//(MaximumNumberOfIterations 300.000000 100.000000 50.000000 20.000000)
(MaximumNumberOfSamplingAttempts 1.000000)
//(Metric "AdvancedMattesMutualInformation")
(Metric "AdvancedNormalizedCorrelation")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(NewSamplesEveryIteration "false")
(NumberOfSamplesForExactGradient 4096.000000)
(NumberOfSpatialSamples 2048.000000)
//(Optimizer "AdaptiveStochasticGradientDescent")
(Optimizer "QuasiNewtonLBFGS")
(GradientMagnitudeTolerance 1e-6)
(Registration "MultiResolutionRegistration")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(ResultImageFormat "mhd")
(Transform "BSplineTransform")
(WriteIterationInfo "false")
(WriteResultImage "false")
(RequiredRatioOfValidSamples 0.02)
//(NumberOfResolutions %(number_of_resolutions))
(FinalGridSpacingInPhysicalUnits 100 100 100)
"""