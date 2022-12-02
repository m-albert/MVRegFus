import numpy as np

import napari
import matplotlib.colors as mcolors
from vispy.color.colormap import Colormap

from mvregfus import mv_utils
import napari


def visualize_views(views,
                    transf_params,
                    stack_props,
                    view_stack_props,
                    viewer=None,
                    clims=None,
                    fused=None,
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
    
    ndim = views[0].ndim

    if viewer is None:
        viewer = napari.Viewer(ndisplay=ndim)
    else:
        viewer.layers.clear()

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

    for iview in range(len(views)):

        colors = [i[1] for i in mcolors.TABLEAU_COLORS.items()]
        colors = colors * 10

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

        pts = np.array([pt for pt in np.ndindex(tuple([2]*ndim))])

        lines = np.array([[pt1, pt2] for pt1 in pts for pt2 in pts if np.sum(np.abs(pt1 - pt2)) == 1])
        lines = lines * np.array(views[iview].shape)
        
        for il, l in enumerate(lines):
            for ipt, pt in enumerate(l):
                lines[il][ipt] = np.dot(p[:ndim, :ndim], lines[il][ipt]) + p[:ndim, ndim]

        viewer.add_shapes(data=lines, ndim=ndim, shape_type='line', name='bbox %s' % iview,
                          edge_color=colors[iview], opacity=0.8, edge_width=0.5, blending='translucent_no_depth')

        # cmap = Colormap([[0, 0, 0, 1], colors[iview]])

        if clims is None:
            clims = [np.percentile(views[iview], i) for i in [5, 96]]
        
        # # add view images
        # viewer.add_image(views[iview],
        #                  contrast_limits=clims, 
        #                  opacity=1,
        #                  name='view %s' % iview,
        #                  affine=p,
        #                  # colormap=cmap,
        #                  colormap='gray_r',
        #                 #  blending='additive',
        #                  gamma=0.6,
        #                  **kwargs)

    return viewer