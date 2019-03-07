__author__ = 'malbert'

import os,tempfile,pdb,sys
import numpy as np
import czifile
import SimpleITK as sitk
from scipy import ndimage
from image_array import ImageArray
# import pyximport
# pyximport.install(setup_args={'include_dirs':'.'})
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   AffineTransform2D,TranslationTransform3D,
                                   # ShearTransform3D,
                                   RigidTransform3D,AffineTransform3D)
# from dipy_transforms import ShearTransform3D

from dipy.core import geometry
from dipy.align.imwarp import mult_aff

import io_utils
io_decorator = io_utils.io_decorator_local

@io_decorator
@io_decorator
def readStackFromMultiviewMultiChannelCzi(filepath,view=0,ch=0,background_level=200,infoDict=None,do_clean_pixels=True,do_smooth=True,extract_rotation=True,do_despeckle=False):
    print('reading %s view %s ch %s' %(filepath,view,ch))
    # return ImageArray(np.ones((10,10,10)))
    if infoDict is None:
        infoDict = getStackInfoFromCZI(filepath)
    stack = czifile.CziFile(filepath).asarray_view_ch(view,ch).squeeze()

    # fuse illuminations
    illuminations = infoDict['originalShape'][1]
    if illuminations > 1:
        print('fusing %s illuminations using simple mean' %illuminations)
        stack = np.mean([stack[i:stack.shape[0]:illuminations] for i in range(illuminations)],0).astype(np.uint16)
        for z in range(stack.shape[0]/illuminations):
            if not z%50: print('fusing z plane: %s' %z)
            stack[z] = np.mean(stack[z*illuminations:z*illuminations+illuminations],0).astype(np.uint16)
        stack = stack[:stack.shape[0]/illuminations]
        # print('choosing only illumination 1')
        # stack = np.array(stack[1:stack.shape[0]:illuminations]).astype(np.uint16)

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
    stack = ImageArray(stack,spacing=infoDict['spacing'][::-1],origin=infoDict['origins'][view][::-1],rotation=rotation)

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
def clean(im,cut=50):

    # axes = [im.shape[i] for i in [-2,-1]]
    axes = [-2,-1]

    # compute forward
    d = dctn(im,norm='ortho',axes=axes)

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
    origin = im.origin
    spacing = im.spacing
    rotation = im.rotation
    binned_spacing = spacing * bin_factors[::-1]
    # binned_origin = origin + (spacing*bin_factors[::-1])/2
    # binned_origin = origin
    binned_origin = origin + (binned_spacing-spacing)/2.

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
            nZs.append(float(baseNode.findall(".//SizeZ")[0].text))

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
    if 'win' in sys.platform:
        elastix_bin = os.path.join(elastix_dir,'elastix.exe')
    else:
        elastix_bin = os.path.join(elastix_dir,'bin/elastix')
        os.environ['LD_LIBRARY_PATH'] = os.path.join(elastix_dir,'lib')

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

    t00 = geometry.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    center_static = np.array(static.shape)/2.*static.spacing + static.origin
    center_mov = np.array(mov.shape)/2.*mov.spacing + mov.origin
    t00offset = center_mov - np.dot(t00[:3,:3],center_static)
    t00[:3,3] = t00offset
    t00 = matrix_to_params(t00)

    # reg_spacing = np.array([fixed.spacing[0]*4]*3)
    print('WARNING: 20180614: changed fft registration spacing')
    reg_iso_spacing = np.min([np.array(im.spacing)*np.array(im.shape)/160. for im in [static,mov]])
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
    # pdb.set_trace()
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

    # pdb.set_trace()

    return [t0, t1, t2]


def get_affine_parameters_from_elastix_output(filepath,t0=None):

    import transformations
    raw_out_params = open(filepath).read()

    elx_out_params = raw_out_params.split('\n')[2][:-1].split(' ')[1:]
    elx_out_params = np.array([float(i) for i in elx_out_params])

    if len(elx_out_params)==6:
        tmp = transformations.euler_matrix(elx_out_params[0],elx_out_params[1],elx_out_params[2])
        elx_affine_params = np.zeros(12)
        elx_affine_params[:9] = tmp[:3,:3].flatten()
        elx_affine_params[-3:] = np.array([elx_out_params[3],elx_out_params[4],elx_out_params[5]])

        # translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)

    if len(elx_out_params)==12: # affine case
        elx_affine_params = elx_out_params

    elif len(elx_out_params)==7: # similarity transform
        angles = transformations.euler_from_quaternion([np.sqrt(1-np.sum([np.power(elx_out_params[i],2) for i in range(3)])),
                                                        elx_out_params[0],elx_out_params[1],elx_out_params[2]])
        tmp = transformations.compose_matrix(angles=angles)
        elx_affine_params = np.zeros(12)
        elx_affine_params[:9] = tmp[:3,:3].flatten()*elx_out_params[6]
        elx_affine_params[-3:] = np.array([elx_out_params[3],elx_out_params[4],elx_out_params[5]])

        # translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)

    elif len(elx_out_params)==3: # translation transform

        elx_affine_params = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])
        elx_affine_params[9:] = elx_out_params

    if len(elx_out_params) in [6,7,12]:
        outCenterOfRotation = raw_out_params.split('\n')[19][:-1].split(' ')[1:]
        outCenterOfRotation = np.array([float(i) for i in outCenterOfRotation])

        # outCenterOfRotation = np.dot(params_to_matrix(params_invert_coordinates(t0)),np.array(list(outCenterOfRotation)+[1]))[:3]

        translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)


    # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)
    # elx_affine_params_numpy = np.concatenate([elx_affine_params[:9][::-1,::-1],translation[::-1]],0)
    dipy_parameters = np.diag((1,1,1,1)).astype(np.float64)
    dipy_parameters[:3,:3] = elx_affine_params[:9].reshape((3,3))[::-1,::-1]
    dipy_parameters[:3,3] = elx_affine_params[-3:][::-1]

    inv_elx_affine_params = params_invert_coordinates(elx_affine_params)
    dipy_parameters = params_to_matrix(inv_elx_affine_params)

    if t0 is not None:
        final_params = mult_aff(dipy_parameters,params_to_matrix(t0))

    return final_params

def get_affine_parameters_from_elastix_output_2d(filepath,t0=None):

    import transformations
    raw_out_params = open(filepath).read()

    elx_out_params = raw_out_params.split('\n')[2][:-1].split(' ')[1:]
    elx_out_params = np.array([float(i) for i in elx_out_params])

    if len(elx_out_params)==3:

        a = elx_out_params[0]
        elx_affine_params = np.eye(3)
        matrix2d = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        elx_affine_params[:2,:2] = matrix2d
        elx_affine_params[:2,2] = elx_out_params[1:]
        elx_affine_params = matrix_to_params(elx_affine_params)

        # tmp = transformations.euler_matrix(elx_out_params[0],elx_out_params[1],0)
        # elx_affine_params = np.zeros(12)
        # elx_affine_params[:9] = tmp[:3,:3].flatten()
        # elx_affine_params[-3:] = np.array([elx_out_params[2],elx_out_params[3],0])

        # translation = elx_affine_params[-3:] - np.dot(elx_affine_params[:9].reshape((3,3)),outCenterOfRotation) + outCenterOfRotation
        # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)

    elif len(elx_out_params)==6: # affine case
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

    elif len(elx_out_params)==2: # translation transform

        elx_affine_params = np.array([1.,0,0,1,0,0])
        elx_affine_params[4:] = elx_out_params


    if len(elx_out_params) in [3,6,7,12]:
        outCenterOfRotation = raw_out_params.split('\n')[19][:-1].split(' ')[1:]
        outCenterOfRotation = np.array([float(i) for i in outCenterOfRotation])

        # outCenterOfRotation = np.dot(params_to_matrix(params_invert_coordinates(t0)),np.array(list(outCenterOfRotation)+[1]))[:3]

        translation = elx_affine_params[-2:] - np.dot(elx_affine_params[:4].reshape((2,2)),outCenterOfRotation) + outCenterOfRotation
        elx_affine_params = np.concatenate([elx_affine_params[:4],translation],0)


    # elx_affine_params = np.concatenate([elx_affine_params[:9],translation],0)
    # elx_affine_params_numpy = np.concatenate([elx_affine_params[:9][::-1,::-1],translation[::-1]],0)
    # dipy_parameters = np.diag((1,1,1)).astype(np.float64)
    # dipy_parameters[:2,:2] = elx_affine_params[:4].reshape((2,2))[::-1,::-1]
    # dipy_parameters[:2,2] = elx_affine_params[-2:][::-1]

    inv_elx_affine_params = params_invert_coordinates(elx_affine_params)
    dipy_parameters = params_to_matrix(inv_elx_affine_params)

    if t0 is not None:
        final_params = mult_aff(dipy_parameters,params_to_matrix(t0))

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

def register_nonlinear_dipy(static,moving,t0=None):

    static_origin = static.origin
    static_spacing = static.spacing
    moving_origin = moving.origin
    moving_spacing = moving.spacing

    if not np.any(np.array(static.shape)-np.array(moving.shape)):
        if not np.any(static-moving):
            return np.array([1,0,0,0,1,0,0,0,1,0,0,0]),[0]

    if t0 is None:
        t0 = np.array([1,0,0,0,1,0,0,0,1,0,0,0])

    static = ImageArray(clahe(static,40,clip_limit=0.02),spacing=static.spacing,origin=static.origin)
    moving = ImageArray(clahe(moving,40,clip_limit=0.02),spacing=moving.spacing,origin=moving.origin)

    static_grid2world = np.eye(4)
    moving_grid2world = np.eye(4)

    np.fill_diagonal(static_grid2world,list(static_spacing)+[1])
    np.fill_diagonal(moving_grid2world,list(moving_spacing)+[1])

    static_grid2world[:3,3] = static_origin
    moving_grid2world[:3,3] = moving_origin

    # from dipy.align.metrics import CCMetric_affine as CCMetric
    from dipy.align.metrics import CCMetric as CCMetric
    from dipy_helpers_test import register_dipy_test

    t0 = params_to_matrix(t0)

    # mapping = register_dipy_test(
    #               np.array(static),
    #               np.array(moving),
    #               t0,
    #               scaling_factors = [4],
    #               sigmas_image = [0],
    #               sigmas_defo = [2],
    #               clahe_kernel_sizes=None,
    #               level_iters = [100]*1,
    #               ccmetric_radius=4,
    #               precomputed_clahe_fixed=None,
    #               precomputed_clahe_moving=None,
    #               pad=0,
    #               CCMetric=CCMetric,
    #               opt_tol = 1e-5,
    #               )
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.imwarp import DiffeomorphicMap
    from dipy.align.metrics import CCMetric

    metric = CCMetric(3,100)
    level_iters = [50, 50]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving, static_grid2world, moving_grid2world, t0)

    return mapping


def get_grid2world(image):
    w2g = np.diag(list(image.spacing)+[1])
    w2g[:-1,-1] = image.origin
    return w2g

def get_world2grid(image):
    # return params_to_matrix(invert_params(matrix_to_params(get_grid2world(image))))
    return np.linalg.inv(get_grid2world(image))


def transform_stack_dipy(stack,p=None,out_shape=None,out_spacing=None,out_origin=None,interp='linear'):

    """
    In [19]: %timeit transform_stack_numpy(stack00,a0)
    4.17 s 8.1 ms per loop (mean std. dev. of 7 runs, 1 loop each)

    In [20]: %timeit transform_stack_dipy(stack00,a0)
    382 ms 2.85 ms per loop (mean std. dev. of 7 runs, 1 loop each)
    """


    if p is None:
        p = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])

    p = np.array(p)
    params = np.eye(4)
    params[:3,:3] = p[:9].reshape((3,3))
    params[:3,3] = p[9:]

    if out_shape is None:
        out_shape = stack.shape

    if out_origin is None:
        out_origin = stack.origin

    if out_spacing is None:
        out_spacing = stack.spacing

    static_grid2world = np.eye(4)
    moving_grid2world = np.eye(4)

    np.fill_diagonal(static_grid2world,list(out_spacing)+[1])
    np.fill_diagonal(moving_grid2world,list(stack.spacing)+[1])

    static_grid2world[:3,3] = out_origin
    moving_grid2world[:3,3] = stack.origin

    affine_map = AffineMap(params,
                           out_shape, static_grid2world,
                           stack.shape, moving_grid2world)

    resampled = affine_map.transform(stack,interp=interp)

    resampled = ImageArray(resampled,origin=out_origin,spacing=out_spacing)

    return resampled

def transform_stack_sitk(stack,p=None,out_shape=None,out_spacing=None,out_origin=None,interp='linear'):
    if p is None:
        p = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])

    p = np.array(p)

    p = params_invert_coordinates(p)

    if out_shape is None:
        out_shape = stack.shape

    if out_origin is None:
        out_origin = stack.origin

    if out_spacing is None:
        out_spacing = stack.spacing

    sstack = sitk.GetImageFromArray(stack)
    sstack.SetSpacing(stack.spacing[::-1])
    sstack.SetOrigin(stack.origin[::-1])

    sstack = transformStack(p,sstack,
                            outShape=out_shape[::-1],
                            outSpacing=out_spacing[::-1],
                            outOrigin=out_origin[::-1],interp = interp)

    sstack = sitk.GetArrayFromImage(sstack)
    sstack = ImageArray(sstack,origin = out_origin, spacing=out_spacing)

    return sstack

def transform_stack_numpy(stack,p,out_shape=None):

    """
    In [19]: %timeit transform_stack_numpy(stack00,a0)
    4.17 s ± 8.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [20]: %timeit transform_stack_dipy(stack00,a0)
    382 ms ± 2.85 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """

    p = np.array(p)
    return ndimage.affine_transform(stack,p[:9].reshape((3,3)),p[9:],output_shape=out_shape)

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
    # if io_utils.is_io_path(p):
    #     p = io_utils.process_input_element(p)
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

        matrix = geometry.euler_matrix(0,+ positions[iview0][3] - positions[iview1][3],0)
        # matrix = geometry.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
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

    matrix = geometry.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    # matrix = geometry.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
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
    seg = im > thresh
    seg = ndimage.binary_erosion(seg,iterations=1)
    seg = ndimage.binary_dilation(seg,iterations=5)
    return seg.astype(np.uint16)

@io_decorator
def get_final_params(ref_view,pairs,params,time_alignment_params=None):
    """
    time_alignment_params: single params from longitudinal registration to be concatenated with view params
    """

    import networkx
    g = networkx.DiGraph()
    for ipair,pair in enumerate(pairs):
        # g.add_edge(pair[0],pair[1],{'p': params[ipair]})
        g.add_edge(pair[0],pair[1], p = params[ipair]) # after update 201809 networkx seems to have changed

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
                    path_params = mult_aff(tmp_params,path_params)
                    print(path_params)
                paths_params.append(matrix_to_params(path_params))

            final_view_params = np.mean(paths_params,0)

        # concatenate with time alignment if given
        if time_alignment_params is not None:
            final_view_params = concatenate_view_and_time_params(time_alignment_params,final_view_params)

        final_params.append(final_view_params)

    return np.array(final_params)

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

def fuse_views_simple(views,params,spacing=None):

    if spacing is None:
        spacing = np.max([view.spacing for view in views],0)

    volume = get_union_volume(views,params)
    stack_properties = calc_stack_properties_from_volume(volume,spacing)

    transformed = []
    for iview,view in enumerate(views):
        tmp = transform_stack_dipy(view,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'])
        transformed.append(np.array(tmp))

    fused = np.max(transformed,0)
    fused = ImageArray(fused,spacing=stack_properties['spacing'],origin=stack_properties['origin'])

    return fused

def fuse_views(views,params,spacing=None):

    if spacing is None:
        spacing = np.max([view.spacing for view in views],0)

    volume = get_union_volume(views,params)
    stack_properties = calc_stack_properties_from_volume(volume,spacing)

    transformed = []
    for iview,view in enumerate(views):
        tmp = transform_stack_dipy(view,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'])
        transformed.append(np.array(tmp))

    fused = np.max(transformed,0).astype(np.uint16)
    fused = ImageArray(fused,spacing=stack_properties['spacing'],origin=stack_properties['origin'])

    return fused

def blur_view(view,
              p,
              orig_properties,
              stack_properties,
              sz,
              sxy,
              ):

    print('blur view..')
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
    if sz:
        o = sitk.SmoothingRecursiveGaussian(o,[sxy,sxy,sz])
    else:
        print('not blurring! (not sz is True)')
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

def density_to_multiview_data(
                              density,
                              params,
                              orig_prop_list,
                              stack_properties,
                              sz,
                              sxy,
                              ):
    """
    Takes a 2D image input, returns a stack of multiview data
    """

    """
    Simulate the imaging process by applying multiple blurs
    """
    out = []
    for ip,p in enumerate(params):
        print('gauss dm %s' %ip)
        # o = sitk.SmoothingRecursiveGaussian(density,sigmas[ip])
        o = blur_view(density,p,orig_prop_list[ip],stack_properties,sz,sxy)
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
                              ):
    """
    The transpose of the density_to_multiview_data operation we perform above.
    """

    density = multiview_data[0]*0.
    density = sitk.Cast(density,sitk.sitkFloat32)
    # outs = multiview_data[0]*0.
    # outs = sitk.Cast(outs,sitk.sitkUInt16)
    for ip,p in enumerate(params):
        print('gauss md %s' %ip)
        o = multiview_data[ip]
        # o = sitk.SmoothingRecursiveGaussian(multiview_data[ip],sigmas[ip])

        # smooth and resample in original view
        o = blur_view(o,p,orig_prop_list[ip],stack_properties,sz,sxy)

        o = sitk.Cast(o,sitk.sitkFloat32)

        o = o*weights[ip]

        density += o

    density = sitk.Cast(density,sitk.sitkFloat32)
    return density

@io_decorator
def fuse_LR_with_weights_dct(
        views,
        params,
        stack_properties,
        num_iterations = 25,
        sz = 4,
        sxy = 0.1
):
    """
    works well:
    - sz6 it 10, some rings
    - sz5 it 20, looks good (good compromise between sz and its)
    - sz4 it 30, good and no rings

    :param views:
    :param params:
    :param stack_properties:
    :param num_iterations:
    :param sz:
    :param sxy:
    :return:
    """

    # get orig properties
    # zfactor = float(1)
    orig_prop_list = []
    for ip in range(len(params)):
        prop_dict = dict()
        prop_dict['size'] = views[ip].shape
        prop_dict['origin'] = views[ip].origin
        prop_dict['spacing'] = views[ip].spacing
        orig_prop_list.append(prop_dict)

    weights = get_weights_dct(
                       views,
                       params,
                       stack_properties,
                       # size=50,
                       size=None,
                       # max_kernel=10,
                       max_kernel=None,
                       # gaussian_kernel=10)
                       gaussian_kernel=None)

    for iw in range(len(weights)):
        tmp = sitk.GetImageFromArray(weights[iw])
        tmp.SetSpacing(stack_properties['spacing'][::-1])
        tmp.SetOrigin(stack_properties['origin'][::-1])
        tmp = sitk.Cast(tmp,sitk.sitkFloat32)
        weights[iw] = tmp

    # print('getting weights..')
    # weight_stack_properties = stack_properties.copy()
    # # weight_stack_properties['spacing'] = weight_stack_properties['spacing']*
    # weights = get_lambda_weights(views,params,weight_stack_properties)
    #
    # for iw in range(len(weights)):
    #     tmp = sitk.GetImageFromArray(weights[iw])
    #     tmp.SetSpacing(weight_stack_properties['spacing'][::-1])
    #     tmp.SetOrigin(weight_stack_properties['origin'][::-1])
    #     tmp = sitk.Cast(tmp,sitk.sitkFloat32)
    #     weights[iw] = tmp

    # views = []
    # masks = []

    nviews = []
    for iview,view in enumerate(views):
        tmp = transform_stack_sitk(view,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'])

        # make sure to take only interpolations with full data
        # tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
        tmp_view = ImageArray(views[iview][:-1,:-1,:-1]+1,spacing=views[iview].spacing,origin=views[iview].origin+views[iview].spacing/2.,rotation=views[iview].rotation)
        mask = transform_stack_sitk(tmp_view,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'],
                                interp='nearest')
        # tmp[tmp==0] = 10
        mask = mask > 0
        nviews.append(tmp*(mask))
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
        # pdb.set_trace()

    noisy_multiview_data = views

    """
    Time for deconvolution!!!
    """

    estimate = sitk.Image(
        int(stack_properties['size'][2]),
        int(stack_properties['size'][1]),
        int(stack_properties['size'][0]),
        sitk.sitkFloat32,
    )
    estimate *= 0.
    estimate += 1.

    estimate.SetSpacing(stack_properties['spacing'][::-1])
    estimate.SetOrigin(stack_properties['origin'][::-1])

    # estimate = np.ones(views[0].shape, dtype=np.float64)
    # expected_data = np.zeros_like(noisy_multiview_data)
    # correction_factor = np.zeros_like(estimate)
    # history = np.zeros(((1+num_iterations,) + estimate.shape), dtype=np.float64)
    # history[0, :, :, :] = estimate
    for i in range(num_iterations):
        print("Iteration", i)
        """
        Construct the expected data from the estimate
        """
        print("Constructing estimated data...")

        expected_data = density_to_multiview_data(
              estimate,
              params,
              orig_prop_list,
              stack_properties,
              sz,
              sxy,
        )
        # multiview_data_to_visualization(expected_data, outfile='expected_data.tif')
        "Done constructing."
        """
        Take the ratio between the measured data and the expected data.
        Store this ratio in 'expected_data'
        """
        for ip in range(len(params)):
            expected_data[ip] += 1e-6 #Don't want to divide by 0!
        expected_data = [noisy_multiview_data[ip] / expected_data[ip] for ip in range(len(params))]

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
        )#, out=correction_factor)
        """
        Multiply the old estimate by the correction factor to get the new estimate
        """
        estimate = estimate * correction_factor

        estimate = estimate * sitk.Cast(estimate<2**16,sitk.sitkFloat32)
        estimate = estimate * sitk.Cast(estimate>0,    sitk.sitkFloat32)

        """
        Update the history
        """
    print("Done deconvolving")

    estimate = ImageArray(sitk.GetArrayFromImage(estimate).astype(np.uint16),spacing=np.array(estimate.GetSpacing())[::-1],origin=np.array(estimate.GetOrigin())[::-1])

    return estimate

@io_decorator
def fuse_views_weights(views,params,stack_properties,weights=None):

    # if spacing is None:
    #     spacing = np.max([view.spacing for view in views],0)

    # volume = get_union_volume(views,params)
    # stack_properties = calc_stack_properties_from_volume(volume,spacing)

    transformed = []
    for iview,view in enumerate(views):
        tmp = transform_stack_dipy(view,params[iview],
                               out_origin=stack_properties['origin'],
                               out_shape=stack_properties['size'],
                               out_spacing=stack_properties['spacing'])

        transformed.append(np.array(tmp))

    if weights is not None:
        f = np.zeros_like(transformed[0])
        for iw in range(len(transformed)):
                f += (weights[iw]*transformed[iw].astype(np.float)).astype(np.uint16)
    else:
        f = np.mean(transformed,0)


    f = ImageArray(f,spacing=stack_properties['spacing'],origin=stack_properties['origin'])

    return f

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
def calc_stack_properties_from_views_and_params(views,params,spacing=None,mode='sample'):

    if spacing is None:
        spacing = np.max([view.spacing for view in views],0).astype(np.float)

    if mode == 'sample':
        volume = get_sample_volume(views,params)
    elif mode == 'union':
        volume = get_union_volume(views,params)
    elif mode == 'intersection':
        volume = get_intersection_volume(views,params)

    stack_properties = calc_stack_properties_from_volume(volume,spacing)
    return stack_properties

@io_decorator
def transform_view_with_decorator(view,params,iview,stack_properties):


    sview = sitk.GetImageFromArray(view)
    sview.SetSpacing(view.spacing[::-1])
    sview.SetOrigin(view.origin[::-1])
    sview = transformStack(params_invert_coordinates(params[iview]),
                           sview,
                           outOrigin=stack_properties['origin'][::-1],
                           outShape=stack_properties['size'][::-1],
                           outSpacing=stack_properties['spacing'][::-1])

    return ImageArray(sitk.GetArrayFromImage(sview),spacing=stack_properties['spacing'])


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
        # t = mv_utils.transform_stack_dipy(
                                                # vsr[i],final_params[i],
                                                views[i]+1,params[i], # adding one because of weights (taken away again further down)
                                                out_shape=stack_properties['size'],
                                                out_spacing=stack_properties['spacing'],
                                                out_origin=stack_properties['origin'],
                                                )

        # make sure that only those pixels are kept which are interpolated from 100% valid interpolation pixels
        tmp_view = ImageArray(views[i][:-1,:-1,:-1]+1,spacing=views[i].spacing,origin=views[i].origin+views[i].spacing/2.,rotation=views[i].rotation)

        mask = transform_stack_sitk(
        # t = mv_utils.transform_stack_dipy(
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
        # t = mv_utils.transform_stack_dipy(
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
        # t = mv_utils.transform_stack_dipy(
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
    if outSpacing is None:
        outSpacing = stack.GetSpacing()
    else: outSpacing = np.array(outSpacing)
    outSpacing = np.array(outSpacing).astype(np.float)
    if outOrigin is None: outOrigin = stack.GetOrigin()
    else: outOrigin = np.array(outOrigin)

    # don't do anything if stack nothing is to be done
    def vectors_are_same(v1,v2):
        return np.sum(np.abs(np.array(v1).astype(np.uint8) - np.array(v2).astype(np.uint8))) == 0
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

    newim = sitk.Resample(stack,shape,transf,interpolator,outOrigin,outSpacing)
    if numpyarray:
        newim = sitk.GetArrayFromImage(newim)

    return newim


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

def register_linear_projections_iter(im0,im1,t0=None):

    # im0 = ImageArray(clahe(im0,10,clip_limit=0.02),spacing=im0.spacing,origin=im0.origin)
    # im1 = ImageArray(clahe(im1,10,clip_limit=0.02),spacing=im1.spacing,origin=im1.origin)

    # if t0 is None:
    #     t0 = np.eye(4)
    # else:
    #     t0 = params_to_matrix(t0)

    # projs0 = [np.max(im0,dim) for dim in range(3)]
    # # projs1 = [np.max(im1,dim) for dim in range(3)]
    #
    # for dim in range(3):
    #     projs0[dim].spacing = np.delete(projs0[dim].spacing,dim,axis=0)
    #     # projs1[dim].spacing = np.delete(projs1[dim].spacing,dim,axis=0)
    #     projs0[dim].origin = np.delete(projs0[dim].origin,dim,axis=0)
    #     # projs1[dim].origin = np.delete(projs1[dim].origin,dim,axis=0)

    cur_params = t0
    # for dim in [0,1,2]:
    for i in range(2):
        for dim in [0,1,2]:
            cur_params = register_linear_2d_from3d(im0,im1,cur_params,dim,transform='translation')

    for i in range(10):
        for dim in [0,1,2]:
            cur_params = register_linear_2d_from3d(im0,im1,cur_params,dim)
            # cur_params = get_affine_parameters_from_elastix_output_2d(im0,im1,cur_params,dim)

    # return matrix_to_params(cur_params)
    return cur_params

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

    matrix = geometry.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    # matrix = geometry.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
    t0 = np.append(matrix[:3, :3].flatten(), matrix[:3, 3])

    offset = mean1 - np.dot(matrix[:3,:3],mean0)
    t0[9:] = offset
    t0[10] = 0

    return t0

def register_linear_projections(fixed,moving):

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

    matrix = geometry.euler_matrix(0,+ fixed.rotation - moving.rotation,0)
    # matrix = geometry.euler_matrix(0,- positions[iview0][3] + positions[iview1][3],0)
    t0 = np.append(matrix[:3, :3].flatten(), matrix[:3, 3])

    offset = mean1 - np.dot(matrix[:3,:3],mean0)
    t0[9:] = offset
    t0[10] = 0

    parameters = register_linear_projections_iter(static,mov,t0)

    return parameters

def register_linear_2d_from3d(im0,im1,t0=None,dim=0,
                              iterations=10,
                              transform='rigid',
                              ):

    if t0 is None:
        t0 = matrix_to_params(np.eye(4))
    # im0 = ImageArray(clahe(im0,10,clip_limit=0.02),spacing=im0.spacing,origin=im0.origin)
    # im1 = ImageArray(clahe(im1,10,clip_limit=0.02),spacing=im1.spacing,origin=im1.origin)

    t0 = params_to_matrix(t0)
    im10 = transform_stack_dipy(im1,matrix_to_params(t0),
                                   out_shape=im0.shape,
                                   out_spacing=im0.spacing,
                                   out_origin=im0.origin)

    proj0 = np.max(im0.astype(np.float),dim)
    proj1 = np.max(im10.astype(np.float),dim)
    proj0.spacing = np.delete(proj0.spacing,dim,axis=0)
    proj1.spacing = np.delete(proj1.spacing,dim,axis=0)
    proj0.origin = np.delete(proj0.origin,dim,axis=0)
    proj1.origin = np.delete(proj1.origin,dim,axis=0)

    # params2d = register_linear_2d(proj0,proj1)#),cur_params2d)
    params2d = register_linear_2d_pyelastix(proj0,proj1,
                                            iterations=iterations,
                                            transform=transform,
                                            )#),cur_params2d)
    params3d = params2d
    params3d = np.insert(params3d,dim,np.zeros(3),axis=0)
    params3d = np.insert(params3d,dim,np.zeros(4),axis=1)
    params3d[dim,dim] = 1
    # cur_params = params3d
    # cur_params = mult_aff(params3d,t0)
    cur_params = np.dot(t0,params3d)
    return matrix_to_params(cur_params)
    # return matrix_to_params(params3d)

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



if __name__ == '__main__':

    filepath = '/data/malbert/data/dbspim/chemokine/20170718_44star_mutmut7/wt_01.czi'
    # filepath = '/data/malbert/data/dbspim/Erika/20140911_cxcr7_wt/wt_01.czi'

    infoDict = getStackInfoFromCZI(filepath)
    res = readMultiviewCzi(filepath,ds=16,infoDict=infoDict)

    pairs = [[4,4],[4,5]]
    # pairs = [[0,0],[0,1],[0,2],[0,3]]
    # pairs = [[1,0],[1,1],[1,2],[1,3],[1,4],[1,5]]
    ip = calc_initial_parameters(pairs,infoDict,res[0])
    # regs = [register_warp(res[0][iview0],res[0][iview1],ip[ipair]) for ipair,[iview0,iview1] in enumerate(pairs)]
    regs_el = [register_linear_elastix_seq(res[0][iview0],res[0][iview1],ip[ipair]) for ipair,[iview0,iview1] in enumerate(pairs)]
    # ps = [regs[ipair][0] for ipair,[iview0,iview1] in enumerate(pairs)]
    # es = [regs[ipair][1] for ipair,[iview0,iview1] in enumerate(pairs)]
    # t = np.array([transform_stack_dipy(res[0][iview1],ps[ipair],out_origin=res[0][iview0].origin,out_shape=res[0][iview0].shape,out_spacing=res[0][iview0].spacing) for ipair,[iview0,iview1] in enumerate(pairs)])
    t_init = np.array([res[0][0]]+[transform_stack_dipy(res[0][iview1],ip[ipair],out_origin=res[0][iview0].origin,out_shape=res[0][iview0].shape,out_spacing=res[0][iview0].spacing) for ipair,[iview0,iview1] in enumerate(pairs) if ipair])
    # t_affine = np.array([res[0][0]]+[transform_stack_dipy(res[0][iview1],regs[ipair][1][1],out_origin=res[0][iview0].origin,out_shape=res[0][iview0].shape,out_spacing=res[0][iview0].spacing) for ipair,[iview0,iview1] in enumerate(pairs) if ipair])
    t_elx = np.array([res[0][0]]+[transform_stack_dipy(res[0][iview1],regs_el[ipair],out_origin=res[0][iview0].origin,out_shape=res[0][iview0].shape,out_spacing=res[0][iview0].spacing) for ipair,[iview0,iview1] in enumerate(pairs) if ipair])
    # t_waffine = np.array([res[0][0]]+[transform_stack_dipy(res[0][iview1],regs[ipair][1][2],out_origin=res[0][iview0].origin,out_shape=res[0][iview0].shape,out_spacing=res[0][iview0].spacing) for ipair,[iview0,iview1] in enumerate(pairs) if ipair])
    # t_warp = np.array([res[0][0]]+[regs[ipair][1][0].transform(res[0][iview1]) for ipair,[iview0,iview1] in enumerate(pairs) if ipair])

    # import dipy_helpers
    # from dipy.align.metrics import CCMetric as CCMetric
    # s =
    # dipy_helpers.register_dipy(
    #     res[0][0],
    #     res[0][1],
    #     np.eye(4),
    #               scaling_factors = [2,1],
    #               sigmas_image = [1,0],
    #               sigmas_defo = [1,1],
    #               clahe_kernel_sizes=None,
    #               level_iters = [100,100],
    #               ccmetric_radius=4,
    #               precomputed_clahe_fixed=None,
    #               precomputed_clahe_moving=None,
    #               pad=0,
    #               CCMetric=CCMetric
    #               )

    # tifffile.imshow(np.max(np.array([res[0][0]]*2+[regs[i][1][0].transform(res[0][i]) for i in range(1,4)]),-2),vmax=300)
    # tifffile.imshow(np.max(np.array(np.linalg.norm((regs[2][1][0].backward-linearise_displacement_field(regs[2][1][0].backward)[0]),axis=-1)*(res[0][0]>10)),-1))