from __future__ import absolute_import

"""

This file both
- contains the configuration of the mvregfus run
- and starts the run

Therefore create a copy of this file for each run. Soon configuration file handling will be added.

"""

import logging
import os

import numpy as np

from mvregfus import mv_graph, io_utils

logging.basicConfig(level=logging.WARN)

##########################
#### parameters to modify
#### approx. in descending order of relevance
##########################

# where elastix can be found (top folder)
elastix_dir = '/home/user/elastix'

# list of files to fuse
filepaths = ['/home/user/some_folder/images/file01.czi',
             '/home/user/some_folder/images/file02.czi']

# where to save the output
out_dir = os.path.dirname(filepaths[0])  # e.g. same folder as first file in filepaths

# channels to fuse
channels = [0,1]
channelss = [channels]*len(filepaths)

# channel to use for registration
reg_channel = 0
reg_channels = [reg_channel] *len(filepaths)

# reference view for final fusion
ref_view = 0
ref_views = [ref_view] *len(filepaths)

# list of pairwise view indices to perform registration on
# registration_pairs = [[0,1]]
registration_pairs = None
registration_pairss = [registration_pairs] *len(filepaths)

# optionally, specify the meanings of the indices
# occuring in the list of pairs
# this can be used to fuse illuminations independently
# using view_dict, which is a dictionary containing
# the indices as keys and dictionaries defining the indices as items, such as:
# >>> view_dict[0] = {'view': 0 # view 0 within the file
#                     'ill' : 1 # illumination 1 within that view
#                     }
# another example:
# >>> view_dict[0] = {'view': 0    # view 0 within the file
#                     'ill' : None # like this, both ills of this view are fused using the average of ills
#                     }
# another example:
# >>> view_dict[0] = {'view': 0    # view 0 within the file
#                     'ill' : 2    # like this, both ills of this view are fused using blending weights
#                     }

# in case of treating ills as independent views:
#              - illumination 0 comes from left
#              - illumination 1 comes from right
#              - rotating in positive direction (in angles)
#                brings left to the front
# so it makes sense to define the registration pairs like this: (view, ill)
# (0,1),(0,0)
# (0,0),(1,1)
# (1,1),(1,0)
# (1,0),(2,1)
# etc.

# four view example:
# view_dict = {i:{'view':i, 'ill': 2} for i in [0, 1, 2, 3]}

# if ills of all views should be averaged, set view_dict to None:
view_dict = None

# how to calculate final fusion volume
# 'sample': takes best quality z plane of every view to define the volume
# 'union': takes the union of all view volumes
final_volume_mode = 'sample'

# whether to perform an affine chromatic correction
# and which channel to use as reference
perform_chromatic_correction = False
ref_channel_chrom = 0

# binning of raw input from views (x,y,z)
# [1,1,1]: no binning
# shapes of views to be registered should not significantly
# exceed ~(400, 400, 400)
raw_input_binning = [4,4,1]

# background level to subtract
background_level = 200

# which binning to use for registration
# mv_registration_bin_factors = np.array([1,1,1])
mv_registration_bin_factors = np.array([4,4,4])

# registration mode for pairwise view registration
# (default is 2)
# -1: only preregistration (translation, no elastix)
# 0: only translation
# 1: translation + rotation
# 2: translation + rotation + affine
pairwise_registration_mode = 2

# final output spacing in um
mv_final_spacing = np.array([5.]*3)

# options for fusion
# fusion_method
# 'weighted_average': weighted average of views using the given weights
# 'LR': Lucy-Richardson multi-view deconvolution
fusion_method = 'LR'
# fusion_method = 'weighted_average'

# fusion weights
# 'blending': uniform weights with blending at the stack borders
# 'dct': weights derived from DCT image quality metric
fusion_weights = 'dct'
# fusion_weights = 'blending'

# options for DCT image quality metric for fusion
# setting None automatically calculates good values

# size of the cubic volume blocks on which to calc quality
dct_size = None
# size of maximum filter kernel
dct_max_kernel = None
# size of gaussian kernel
dct_gaussian_kernel = None

# weight normalisation parameters
# normalise such that approx. <dct_cumulative_weight_best_views> weight is
# contained in the <dct_how_many_best_views> best views
dct_how_many_best_views = 2
dct_cumulative_weight_best_views = 0.9

# options for weighted Lucy Richardson multi-view deconvolution
# maximum number of iterations
LR_niter = 25  # iters
# convergence criterion
LR_tol = 5e-5  # tol
# gaussian PSF sigmas
LR_sigma_z = 4  # sigma z
LR_sigma_xy = 0.5  # sigma xy


##########################
#### end of parameters to modify
##########################

# graph_multiview.multiview_fused_label = graph_multiview.multiview_fused_label[:-2] + 'mhd'
# graph_multiview.transformed_view_label = graph_multiview.transformed_view_label[:-2] + 'mhd'

graph = dict()
result_keys = []
for ifile,filepath in enumerate(filepaths):
    channels = channelss[ifile]
    # pairs = pairss[ifile]

    graph.update(
        mv_graph.build_multiview_graph(
        filepath = filepath,
        pairs = registration_pairss[ifile],
        view_dict = view_dict,
        ref_view = ref_views[ifile],
        # mv_registration_bin_factors = np.array([8,8,2]),
        mv_registration_bin_factors = mv_registration_bin_factors, # x,y,z
        mv_final_spacing = mv_final_spacing, # orig resolution
        reg_channel = reg_channel,
        channels = channels,
        ds = 0,
        sample = ifile,
        out_dir = out_dir,
        perform_chromatic_correction = perform_chromatic_correction,
        ref_channel_chrom = ref_channel_chrom,
        final_volume_mode = final_volume_mode,
        elastix_dir = elastix_dir,
        raw_input_binning = raw_input_binning, # x,y,z
        background_level = background_level,
        dct_size = dct_size,
        dct_max_kernel = dct_max_kernel,
        dct_gaussian_kernel = dct_gaussian_kernel,
        LR_niter = LR_niter,  # iters
        LR_sigma_z = LR_sigma_z,  # sigma z
        LR_sigma_xy = LR_sigma_xy,  # sigma xy
        LR_tol = LR_tol,  # tol
        fusion_method = fusion_method,
        fusion_weights = fusion_weights,
        dct_how_many_best_views=dct_how_many_best_views,
        dct_cumulative_weight_best_views=dct_cumulative_weight_best_views,
        pairwise_registration_mode = pairwise_registration_mode,
        debug_pairwise_registration=True,
        )
    )

    # choose same reference coordinate system
    # if ifile:
    #     graph[graph_multiview.stack_properties_label %(0,ifile)] = graph_multiview.stack_properties_label %(0,0)

    # out_file = os.path.join(os.path.dirname(filepath),graph_multiview.multiview_fused_label %(0,ifile,0))
    # if os.path.exists(out_file):
    #     print('WARNING: skipping %s because %s already exists' %(filepath,out_file))
    #     continue

    multiview_fused_labels              = [mv_graph.multiview_fused_label % (0, ifile, ch) for ch in channels]
    # fusion_params_label                 = 'mv_params_%03d_%03d.prealignment.h5' %(ikey,s)
    result_keys += multiview_fused_labels
        # p = threaded.get(graph,fusion_params_label)


# run
if __name__ == '__main__':

    # Number of files to fuse in parallel.
    # Bottleneck here is GPU memory (if used):
    # Each parallel file requires approx. 8GB of GPU memory

    N = 1
    results = []
    for i in range(0, len(result_keys), N * len(channels)):
        results.append(io_utils.get(graph, result_keys[i:i + N * len(channels)]))