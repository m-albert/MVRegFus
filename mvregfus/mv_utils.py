import SimpleITK as sitk
import numpy as np
from scipy.linalg import expm

from mvregfus.image_array import ImageArray


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """

    # taken from https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    import numpy,math

    # epsilon for testing whether a number is close to zero
    _EPS = numpy.finfo(float).eps * 4.0

    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    # map axes strings to/from tuples of inner axis, parity, repetition, frame
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = numpy.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


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


def image_to_sitk(im):
    sim = sitk.GetImageFromArray(im)
    sim.SetOrigin(im.origin[::-1])
    sim.SetSpacing(im.spacing[::-1])
    return sim


def sitk_to_image(sim):
    im = sitk.GetArrayFromImage(sim)
    im = ImageArray(im,spacing=np.array(sim.GetSpacing())[::-1],origin=np.array(sim.GetOrigin())[::-1])
    return im


def transform_points(pts, p):
    A = p[:9].reshape((3,3))
    c = p[9:]
    pts_t = np.array([np.dot(A, pt) + c for pt in pts]) # should vectorize this
    return pts_t


def bin_stack(im,bin_factors=np.array([1,1,1])):
    if np.allclose(bin_factors, [1, 1, 1]):
        return im
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
    im = ImageArray(im, spacing=binned_spacing, origin=binned_origin, rotation=rotation)
    return im