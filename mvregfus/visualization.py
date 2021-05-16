import numpy as np

import napari
import matplotlib.colors as mcolors
from vispy.color.colormap import Colormap

from . import mv_utils
from . import io_utils


def visualize_views(views,
                    transf_params,
                    stack_props,
                    view_stack_props,
                    viewer=None,
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

    if viewer is None:
        viewer = napari.Viewer(ndisplay=3)

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
        p[:3, 3] = np.dot(syi, np.dot(p[:3, :3], stack_props['origin'])) \
                   + np.dot(syi, (p[:3, 3] - view_stack_props[iview]['origin']))
        p[:3, :3] = np.dot(syi, np.dot(p[:3, :3], sx))
        p = np.linalg.inv(p)

        # draw bounding boxes
        rel_coords = [0, 1]
        pts = np.array([[i, j, k] for i in rel_coords
                        for j in rel_coords for k in rel_coords])

        lines = np.array([[pt1, pt2] for pt1 in pts for pt2 in pts if np.sum(np.abs(pt1 - pt2)) == 1])
        lines = lines * np.array(views[iview].shape)
        for il, l in enumerate(lines):
            for ipt, pt in enumerate(l):
                lines[il][ipt] = np.dot(p[:3, :3], lines[il][ipt]) + p[:3, 3]

        viewer.add_shapes(data=lines, ndim=3, shape_type='line', name='bbox %s' % iview,
                          edge_color=colors[iview], opacity=1., edge_width=.5)

        cmap = Colormap([[0, 0, 0, 1], colors[iview]])

        # add view images
        viewer.add_image(views[iview],
                         opacity=1,
                         name='view %s' % iview,
                         affine=p,
                         colormap=cmap,
                         blending='additive',
                         **kwargs)

    return viewer
