from __future__ import absolute_import
import logging
import os
import glob
import numpy as np
import json

from mvregfus import mv_graph, io_utils

logging.basicConfig(level=logging.WARN)


def read_config_file(config_file):
    """Import the MVRegFus parameters from the config json file and retrun a dictionary that can be used with `run_fusion`.
    """
    with open(config_file, 'r') as f:
        params = json.load(f)
    return params

def run_fusion(parameters):
    """Run MVRegFus fusion using a set of parameters.

    Those parameters can be specified in the `run_fusion.ipynb` notebook.
    """
    filepaths = parameters['filepaths']
    out_dir = parameters['out_dir']
    generate_file_order = parameters['generate_file_order']
    channels = parameters['channels']
    reg_channel = parameters['reg_channel']
    ref_view = parameters['ref_view']
    registration_pairs = parameters['registration_pairs']
    n_volumes = parameters['n_volumes']
    n_views = parameters['n_views']
    raw_input_binning = parameters['raw_input_binning']
    mv_registration_bin_factors = parameters['mv_registration_bin_factors']
    mv_final_spacing = parameters['mv_final_spacing']
    background_level = parameters['background_level']
    fusion_method = parameters['fusion_method']
    fusion_weights = parameters['fusion_weights']
    
    if isinstance(filepaths, str):
        filepaths = glob.glob(os.path.join(filepaths, '*.czi'))

    if generate_file_order:
        with open(out_dir + '/file_order.txt', 'w') as f:
            for item in filepaths:
                f.write("%s\n" % item)

    if registration_pairs is None:
        total_views = np.arange(n_volumes * n_views)
        reg_pairs = []
        for i in total_views:
            if i == 0:
                continue
            if (i+1) % n_views == 0:
                reg_pairs.append([i, i-n_views+1])
            if i % n_views == 0:
                reg_pairs.append([i-n_views, i])
                continue
            reg_pairs.append([i-1, i])
        registration_pairs = reg_pairs


    # ----- other parameters ---- 
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


    # how to calculate final fusion volume
        # 'sample': takes best quality z plane of every view to define the volume
        # 'union': takes the union of all view volumes
    final_volume_mode = 'union'
    # whether to perform an affine chromatic correction
    # and which channel to use as reference
    perform_chromatic_correction = False
    ref_channel_chrom = 0


    # ----- derivatives of the set parameters -----
    channelss = [channels]*len(filepaths)
    reg_channels = [reg_channel] *len(filepaths)
    ref_views = [ref_view] *len(filepaths)
    registration_pairss = [registration_pairs] *len(filepaths)
    view_dict = {i:{'view':i, 'ill':2} for i in list(range(16))}
    # where elastix can be found (top folder) ## urrently not used
    elastix_dir = '/data/shared/elastix'


    graph = dict()
    result_keys = []
    for ifile,filepath in enumerate(filepaths):
        channels = channelss[ifile]

        graph.update(
            mv_graph.build_multiview_graph(
            filepath = filepath,
            pairs = registration_pairss[ifile],
            view_dict = view_dict,
            ref_view = ref_views[ifile],
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
            pairwise_registration_mode=-1, #no elastix
            )
        )

        multiview_fused_labels = [mv_graph.multiview_fused_label % (0, ifile, ch) for ch in channels]
        result_keys += multiview_fused_labels

    for k in result_keys:
        io_utils.get(graph, k, local=True)

if __name__ == '__main__':
    for k in result_keys:
        io_utils.get(graph, k, local=True)
