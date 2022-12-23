import numpy as np

import napari
import matplotlib.colors as mcolors
from vispy.color.colormap import Colormap

from mvregfus import mv_utils
import napari

def visualize_views(views,
                    transf_params=None,
                    stack_props=None,
                    view_stack_props=None,
                    viewer=None,
                    clims=None,
                    fused=None,
                    edge_width=10,
                    **kwargs):

    """
    Use napari to visualize views in target space
    :param views:
    :param transf_params:
    :param stack_props:
    :param view_stack_props:
    :param kwargs:
    :return:
    """
    
    ndim = len(view_stack_props[0]['origin'])

    if viewer is None:
        viewer = napari.Viewer(ndisplay=ndim)
    # elif not len(viewer.layers) == len(views)*2:
    #     viewer.layers.clear()

    if fused is not None:
        viewer.add_image(fused,
                        contrast_limits=clims, 
                        opacity=1,
                        name='fused',
                        # affine=p,
                        # colormap=cmap,
                        colormap='gray_r',
                        gamma=0.6,
                        **kwargs)

    # import pdb; pdb.set_trace()
    print('lalalal', transf_params, ndim)
    if transf_params is None:
        transf_params = [mv_utils.matrix_to_params(np.eye(ndim+1)) for i in range(len(views))]

    if stack_props is None:
        stack_props = {}
        stack_props['origin'] = np.min([view_stack_props[iview]['origin'] for iview in range(len(views))], axis=0)
        stack_props['spacing'] = np.min([view_stack_props[iview]['spacing'] for iview in range(len(views))], axis=0)

    # get affine parameters
    ps = []
    for iview in range(len(views)):

        p = mv_utils.params_to_matrix(transf_params[iview])

        """
        y = Ax+c
        y=sy*yp+oy
        x=sx*xp+ox
        sy*yp+oy = A(sx*xp+ox)+c
        yp = syi * A*sx*xp + syi  *A*ox +syi*(c-oy)
        A' = syi * A * sx
        c' = syi  *A*ox +syi*(c-oy)
        """
        sx = np.diag(list((stack_props['spacing'])))
        sy = np.diag(list((view_stack_props[iview]['spacing'])))
        syi = np.linalg.inv(sy)
        p[:ndim, ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], stack_props['origin'])) \
                   + np.dot(syi, (p[:ndim, ndim] - view_stack_props[iview]['origin']))
        p[:ndim, :ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], sx))
        p = np.linalg.inv(p)

        ps.append(p)

    # add view images
    for iview in range(len(views)):

        if clims is None:
            print('warning: computing contrast limits from data, consider providing them manually for dask arrays')
            data = np.array(views[iview])
            clims = [np.percentile(data, i) for i in [5, 96]]

        layer_names = [l.name for l in viewer.layers]
        name='view %s' % iview

        if name in layer_names:
        # if 0:
            ind = layer_names.index(name)
            viewer.layers[ind].data = views[iview]
            viewer.layers[ind].affine.affine_matrix = ps[iview]
            viewer.layers[ind].refresh()

        else:
            viewer.add_image(views[iview],
                            contrast_limits=clims, 
                            opacity=1,
                            name=name,
                            affine=ps[iview],
                            # colormap=cmap,
                            colormap='gray_r',
                            #  blending='additive',
                            gamma=0.6,
                            **kwargs)

    # add bounding boxes
    for iview in range(len(views)):

        p = ps[iview]

        pts = np.array([pt for pt in np.ndindex(tuple([2]*ndim))])

        lines = np.array([[pt1, pt2] for pt1 in pts for pt2 in pts if np.sum(np.abs(pt1 - pt2)) == 1])
        # lines = lines * np.array(views[iview].shape)
        lines = lines * np.array(view_stack_props[iview]['shape'])
        
        for il, l in enumerate(lines):
            for ipt, pt in enumerate(l):
                lines[il][ipt] = np.dot(p[:ndim, :ndim], lines[il][ipt]) + p[:ndim, ndim]

        colors = [i[1] for i in mcolors.TABLEAU_COLORS.items()]
        colors = colors * 10

        layer_names = [l.name for l in viewer.layers]
        name='bbox %s' % iview
        if name in layer_names:
        # if 0:
            ind = layer_names.index(name)
            viewer.layers[ind].data = lines
            viewer.layers[ind].refresh()
        else:
            viewer.add_shapes(data=lines, ndim=ndim, shape_type='line', name=name,
                                edge_color=colors[iview], opacity=1, edge_width=edge_width, blending='translucent_no_depth')

        # cmap = Colormap([[0, 0, 0, 1], colors[iview]])

    return viewer