__author__ = 'malbert'

import copy
import os
import sys

import numpy as np
from distributed import Client

from mvregfus import multiview, io_utils, mv_utils

# import pickle
# client = Client(processes=True)
client = Client(processes=False)#,threads_per_worker=1)
# client = Client(processes=False,threads_per_worker=1)
dashboard_link = 'http://localhost:%s' % int(client.cluster.scheduler.service_ports['dashboard'])
print(dashboard_link)

if sys.platform.startswith("win"):
    try:
        os.system("title "+"multi-view fusion: "+dashboard_link)
    except: pass
elif sys.platform.startswith("lin") or sys.platform.startswith("dar"):
    print('\33]0;multi-view fusion: %s\a' %dashboard_link, end='', flush=True)


# multiview_fused_label               = 'mv_%03d_%03d_c%02d.imagear.h5'
# multiview_fused_label               = 'mv_%03d_%03d_c%02d.mhd'
multiview_fused_label               = 'mv_%03d_%03d_c%02d.ims'
multiview_fusion_seg_label          = 'mv_fusion_seg_%03d_%03d_c%02d.imagear.h5'
multiview_view_reg_label            = 'view_reg_%03d_%03d_v%03d_c%02d'
multiview_view_fullres_label        = 'view_fullres_%03d_%03d_v%03d_c%02d'
multiview_weights_label             = 'mv_weights_%03d_%03d_v%03d_c%02d.imagear.h5'
multiview_view_corr_label           = 'view_corr_%03d_%03d_v%03d_c%02d'
multiview_metadata_label            = 'mv_metadata_%03d_%03d.dict.h5'
multiview_data_label                = 'mv_data_%03d_%03d.image.h5'
fusion_params_pair_label            = 'mv_params_%03d_%03d_vfix%03d_vmov%03d.prealignment.h5'
time_alignment_pair_params_label    = 'mv_time_alignment_pair_params_%03d_%03d.prealignment.h5'
time_alignment_params_label         = 'mv_time_alignment_params_%03d_%03d.prealignment.h5'
fusion_view_params_label            = 'mv_view_params_%03d_%03d.prealignment.h5'
fusion_params0_label                = 'mv_params0_%03d_%03d.prealignment.h5' # params from pairs, after combining view and time alignment
fusion_params_label                 = 'mv_params_gw_%03d_%03d.prealignment.h5' # final params after groupwise registration
# chromatic_correction_params_label   = 'mv_params_%03d_%03d_c%02d.prealignment.h5'
chromatic_correction_params_label   = 'chromcorr_params_%03d_%03d_refch%02d_ch%02d.prealignment.h5'
stack_properties_label              = 'mv_stack_props_%03d_%03d.dict.h5'
orig_stack_properties_label         = 'mv_orig_stack_props_%03d_%03d_v%03d.dict.h5'
# transformed_view_label              = 'mv_transf_view_%03d_%03d_v%03d_c%02d.imagear.h5'
transformed_view_label              = 'mv_transf_view_%03d_%03d_v%03d_c%02d.image.ims'
multiview_chrom_correction_channel_label = 'mv_chrom_corr_%03d_%03d_c%02d.imagear.h5'


def build_multiview_graph(
    out_dir,
    filepath,
    input_graph=None,
    channels = [0],
    reg_channel = 0,
    ref_view = 0,
    ds = 0,
    sample = 0,
    mv_registration_bin_factors = np.array([4,4,1]),
    # mv_registration_bin_factors = np.array([4,4,1]),
    mv_final_spacing = np.array([3.]*3), # orig resolution
    # pairs = [[1,0],[1,2],[0,3],[2,3],[1,4],[4,5],[4,6],[6,7],[5,7]],
    pairs = [[1,0],[1,2],[0,3],[2,3]],
    view_dict = None,
    background_level = 200,
    perform_chromatic_correction = True,
    ref_channel_chrom = 0,
    fusion_method = 'LR',
    fusion_weights = 'dct',
    fusion_chunk_size = 128,
    dct_size = None,
    dct_max_kernel = None,
    dct_gaussian_kernel = None,
    dct_how_many_best_views=1,
    dct_cumulative_weight_best_views=0.9,
    LR_niter = 25,  # iters
    LR_sigma_z = 4,  # sigma z
    LR_sigma_xy = 0.5,  # sigma xy
    LR_tol = 5e-5,  # tol
    final_volume_mode = 'sample',
    # ref_channel_index_chrom = 0,
    chromatic_correction_file = None,
    time_alignment = False,
    time_alignment_ref_sample = 0,
    time_alignment_ref_view = 0,
    raw_input_binning = None,
    clean_pixels = False,
    elastix_dir = '/scratch/malbert/dependencies_linux/elastix_linux64_v4.8',
    pairwise_registration_mode = 2,
    debug_pairwise_registration = True,
    ):

    if input_graph is None:
        graph = dict()
    else:
        graph = input_graph

    # otherwise an inmutable dict is passed
    if view_dict is not None:
        view_dict = copy.deepcopy(view_dict)

    graph['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)
        except:
            raise(Exception('cannot create output directory %s' %out_dir))

    mv_final_spacing = np.array(mv_final_spacing).astype(np.float64)

    # ref_channel_chrom = channels[ref_channel_index_chrom]
    # ref_channel_chrom = reg_channel

    if view_dict is None or 'filename' not in view_dict[list(view_dict.keys())[0]]:
        info_dict = multiview.getStackInfoFromCZI(filepath)
    else:
        info_dict = None

    if pairs is None:

        if view_dict is None:
            n_views = len(info_dict['origins'])
            pairs = [(i,i+1) for i in range(n_views-1)]# + [(n_views-1,0)]
            # pairs = [(i,i+1) for i in range(n_views-1)]# + [(n_views-1,0)]
            print('Assuming linear chain of overlap in views')

        if view_dict is not None:
            view_indices = [k for k in view_dict.keys()]
            view_indices.sort()
            # pairs = [(view_indices[i],view_indices[i+1]) for i in range(len(view_indices)-1)] + [(view_indices[-1],view_indices[0])]
            pairs = [(view_indices[i],view_indices[i+1]) for i in range(len(view_indices)-1)]# + [(view_indices[-1],view_indices[0])]
            print('Assuming linear chain of overlap in views indicated in view_dict')


    # add details for views
    # in case of defining registration pairs when having illuminations:
    # consider views A and B
    # A.rotation<B.rotation
    # then register A.ill0 with B.ill1 (20190425, derived from Max file zurich z1)
    all_views = np.sort(np.unique(np.array(pairs).flatten()))

    if view_dict is None:
        view_dict = dict()
        for view in all_views:
            view_dict[view] = dict()
            view_dict[view]['view'] = view
            view_dict[view]['ill'] = None

    for k,v in view_dict.items():
        if 'filename' not in v.keys():
            v['filename'] = filepath

    if ref_view not in all_views: raise(Exception('chosen reference view is incompatible with chose registration pairs'))

    # print(''.join(['#']*10))
    # print('These pairs of keys will be registered:\n%s' %pairs)
    print('These pairs of keys will be registered:')
    import pprint
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(pairs)
    print("They refer to the keys in 'view_dict':")
    pp.pprint(view_dict)
    # print(''.join(['#'] * 10))

    # print('collecting stack properties')
    # orig_stack_propss = []
    # for view in all_views:
    #     orig_stack_props = dipy_multiview.get_stack_properties_from_view_dict(view_dict[view],raw_input_binning)
    #     orig_stack_propss.append(orig_stack_props)
    #     print(orig_stack_props)

    # graph[multiview_metadata_label % (ds, sample)] = info_dict

    orig_stack_propss = []
    for view in all_views:
        graph[orig_stack_properties_label %(ds, sample, view)] = (multiview.get_stack_properties_from_view_dict,
                                                                view_dict[view],
                                                                # multiview_metadata_label %(ds,sample),
                                                                info_dict,
                                                                raw_input_binning,
                                                                )
        orig_stack_propss.append(orig_stack_properties_label %(ds,sample,view))

    if time_alignment and sample:

        # graph[time_alignment_pair_params_label %(ds,sample)] = (
        #                                                            dipy_multiview.register_linear_elastix,
        #                                                            multiview_view_reg_label %(ds,time_alignment_ref_sample,time_alignment_ref_view,reg_channel),
        #                                                            multiview_view_reg_label %(ds,sample,time_alignment_ref_view,reg_channel),
        #                                                            1, # degree = 1 (trans + rotation)
        #                                                            elastix_dir,
        #                                                            )

        graph[time_alignment_pair_params_label %(ds,sample)] = (
            multiview.register_linear_elastix,
            multiview_view_reg_label % (ds,sample-1,time_alignment_ref_view,reg_channel),
            multiview_view_reg_label % (ds,sample,time_alignment_ref_view,reg_channel),
            1,  # degree = 1 (trans + rotation)
            elastix_dir,
                                                                   )

        if os.path.exists(os.path.join(out_dir,time_alignment_params_label %(ds,sample))):
            graph[time_alignment_params_label %(ds,sample)] = os.path.join(out_dir,time_alignment_params_label %(ds,sample))
        else:
            graph[time_alignment_params_label %(ds,sample)] = (
                multiview.concatenate_view_and_time_params,
                os.path.join(out_dir,time_alignment_params_label %(ds,sample)),
                time_alignment_params_label % (ds,time_alignment_ref_sample),
                time_alignment_pair_params_label % (ds,sample),
                                                                )

    else:
        graph[time_alignment_params_label %(ds,sample)] = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])

    print('INFO: setting registration degree to 2 (trans+rot+aff)')
    for ipair,pair in enumerate(pairs):

        fusion_params_pair_file = os.path.join(out_dir,fusion_params_pair_label % (ds, sample, pair[0], pair[1]))
        if pair[0] == pair[1]:
            graph[fusion_params_pair_label % (ds, sample, pair[0], pair[1])] = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])
        elif os.path.exists(fusion_params_pair_file):
            graph[fusion_params_pair_label %(ds,sample,pair[0],pair[1])] = fusion_params_pair_file
        else:

            graph[fusion_params_pair_label %(ds,sample,pair[0],pair[1])] = (
                multiview.register_linear_elastix,
                os.path.join(out_dir,fusion_params_pair_label % (ds, sample, pair[0], pair[1])),
                # dipy_multiview.register_linear_projections,
                # os.path.join('/tmp/', fusion_params_pair_label %(ds,sample,ipair)),
                multiview_view_reg_label % (ds,sample,pair[0],reg_channel),
                multiview_view_reg_label % (ds,sample,pair[1],reg_channel),
                pairwise_registration_mode,  # degree = 1 (trans + rotation)
                elastix_dir,
                sample,
                pair[0],
                pair[1],
                out_dir if debug_pairwise_registration else None,
            )

    graph[fusion_params0_label %(ds,sample)] = (
        multiview.get_params_from_pairs,
        os.path.join(out_dir,fusion_params0_label %(ds,sample)),
        ref_view,
        pairs,
        [fusion_params_pair_label %(ds,sample,pair[0],pair[1]) for ipair,pair in enumerate(pairs)],
        time_alignment_params_label % (ds,sample),
                                               )

    # print('WARNING: groupwise registration with relative z scaling from pairwise registration')

    try:
        import SimpleITK as sitk
        sitk.ElastixImageFilter()
        simple_elastix_available = True
        # print('Using groupwise registration')
    except:
        # print("No groupwise registration because SimpleElastix is not available")
        simple_elastix_available = False

    # print('Not using groupwise registration for the moment')
    print('Skipping groupwise registration!')
    simple_elastix_available = False
    # simple_elastix_available = True

    if simple_elastix_available and len(all_views) >= 4:  # restiction given by current elastix groupwise registration implementatin
        graph[fusion_params_label % (ds, sample)] = (
            # dipy_multiview.register_groupwise,
            multiview.register_groupwise_euler_and_affine,
            os.path.join(out_dir, fusion_params_label % (ds, sample)),
            [multiview_view_corr_label % (ds, sample, view, reg_channel) for view in all_views],
            fusion_params0_label % (ds, sample),
            ref_view,
            int(mv_registration_bin_factors[2]),
            final_volume_mode,
        )
    else:
        graph[fusion_params_label % (ds, sample)] = fusion_params0_label % (ds, sample)

    # print('WARNING: skipping groupwise registration!')
    # graph[fusion_params_label % (ds, sample)] = fusion_params0_label %(ds,sample)

    # if os.path.exists(os.path.join(out_dir,stack_properties_label %(ds,sample))):
    #     graph[stack_properties_label %(ds,sample)] = os.path.join(out_dir,stack_properties_label %(ds,sample))
    # else:

    stack_properties_label_file = os.path.join(out_dir,stack_properties_label %(ds,sample))
    if os.path.exists(stack_properties_label_file):
        graph[stack_properties_label %(ds,sample)] = stack_properties_label_file

    elif not time_alignment or not sample:
        graph[stack_properties_label %(ds,sample)] = (
            multiview.calc_stack_properties_from_views_and_params,
            os.path.join(out_dir,stack_properties_label %(ds,sample)),
            # [multiview_view_corr_label %(ds,sample,view,reg_channel) for view in all_views],
            # [multiview_view_corr_label %(ds,sample,view,reg_channel) for view in all_views],
            # [multiview_properties_label %(ds,sample,view,reg_channel) for view in all_views],
            orig_stack_propss,
            fusion_params_label % (ds,sample),
            mv_final_spacing,
            final_volume_mode,
                                                )

    else:
        # graph[stack_properties_label % (ds, sample)] = graph[stack_properties_label %(ds,time_alignment_ref_sample)]
        graph[stack_properties_label % (ds, sample)] = stack_properties_label %(ds,time_alignment_ref_sample)


    for ich,ch in enumerate(channels):

        graph[multiview_fusion_seg_label %(ds,sample,ch)] = (
                                            # dipy_multiview_test.fuse_views,
            multiview.calc_lambda_fusion_seg,
            os.path.join(out_dir,multiview_fusion_seg_label %(ds,sample,ch)),
            [multiview_view_corr_label %(ds,sample,view,ch) for view in all_views],
            fusion_params_label % (ds,sample),
            # mv_final_spacing,
            stack_properties_label % (ds,sample),
                                            )

        weights_label_all_views = multiview_weights_label %(ds,sample,-1,ch)

        fusion_block_overlap = 0
        if fusion_weights == 'dct':

            weights_func = multiview.get_weights_dct
            weights_kwargs = {
                'size': dct_size,
                'max_kernel': dct_max_kernel,
                'gaussian_kernel': dct_gaussian_kernel,
                'how_many_best_views': dct_how_many_best_views,
                'cumulative_weight_best_views': dct_cumulative_weight_best_views,
            }

        elif fusion_weights == 'blending':

            weights_func = multiview.get_weights_simple

            weights_kwargs = {
                }

        else:
            raise(Exception("can't understand fusion weights mode"))

        if fusion_method == 'LR':

            # fusion_func = dipy_multiview.fuse_LR_with_weights
            fusion_func = multiview.fuse_LR_with_weights_np

            fusion_kwargs = {
                'num_iterations': LR_niter,
                'sz': LR_sigma_z,
                'sxy': LR_sigma_xy,
                'tol': LR_tol,
                # 'blur_func': dipy_multiview.blur_view_in_target_space,
            }

            fusion_block_overlap = np.max([fusion_block_overlap,
                                           np.max([LR_sigma_z*4/mv_final_spacing[0],
                                                   LR_sigma_xy*4/mv_final_spacing[0]])])

        elif fusion_method == 'weighted_average':

            fusion_func = multiview.fuse_views_weights
            fusion_kwargs = {}

        else:
            raise(Exception("can't understand fusion mode"))

        graph[multiview_fused_label %(ds,sample,ch)] = (
                                            # dipy_multiview_test.fuse_views,
                                            # dipy_multiview.fuse_views_lambda,
            multiview.fuse_blockwise,
            os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
            [transformed_view_label %(ds,sample,view,ch) for view in all_views],
            fusion_params_label % (ds, sample),
            stack_properties_label % (ds, sample),
            orig_stack_propss,
            fusion_block_overlap,
            weights_func,
            fusion_func,
            weights_kwargs,
            fusion_kwargs,
        )

        if chromatic_correction_file is not None:
            graph[multiview_chrom_correction_channel_label %(ds,sample,ch)] = (multiview.readStackFromMultiviewMultiChannelCzi,
                                                                               chromatic_correction_file,
                                                                               0,
                                                                               ch,
                                                                               background_level,
                                                                               # multiview_metadata_label %(ds,sample),  #infoDict
                                                                               info_dict,  #infoDict
                                                                               False,  #do_clean_pixels
                                                                               False,  #do_smooth
                                                                               False,  #extract_rotation
                                                                               False,  #despeckle
                                                                               raw_input_binning,
                                                                               None,  # default: fuse illuminations for correction
                                                                               )
        else:
            graph[multiview_chrom_correction_channel_label %(ds,sample,ref_channel_chrom)]  = multiview_view_reg_label %(ds,sample,ref_view,ref_channel_chrom)
            graph[multiview_chrom_correction_channel_label %(ds,sample,ch)]                 = multiview_view_reg_label %(ds,sample,ref_view,ch)

        if ch != ref_channel_chrom:
            graph[chromatic_correction_params_label %(ds,sample,ref_channel_chrom,ch)] = (
                multiview.get_chromatic_correction_parameters_center,
                os.path.join(out_dir,chromatic_correction_params_label %(ds,sample,ref_channel_chrom,ch)),
                # multiview_view_reg_label %(ds,sample,ref_view,ref_channel_chrom),
                # multiview_view_reg_label %(ds,sample,ref_view,ch),
                multiview_chrom_correction_channel_label % (ds,sample,ref_channel_chrom),
                multiview_chrom_correction_channel_label % (ds,sample,ch),
                                           )
            # select view from big image container
        for iview,view in enumerate(all_views):

            weights_label = multiview_weights_label %(ds,sample,view,ch)
            weights_file = os.path.join(out_dir,weights_label)
            if os.path.exists(weights_file):
                graph[weights_label] = weights_file
            else:
                graph[weights_label] = (multiview.get_image_from_list_of_images,
                                        os.path.join(out_dir,weights_label),
                                        weights_label_all_views,
                                        iview,
                                        )


            graph[multiview_view_reg_label %(ds,sample,view,ch)] = (
                                                                # lambda x,c,v: x[c][v],
                mv_utils.bin_stack,
                multiview_view_fullres_label % (ds,sample,view,ch),
                mv_registration_bin_factors,
                                                                )

            if view_dict[view]['filename'].endswith('czi'):
                graph[multiview_view_fullres_label %(ds, sample, view, ch)] = (multiview.readStackFromMultiviewMultiChannelCzi,
                                                                        view_dict[view]['filename'],
                                                                        # view,
                                                                        view_dict[view]['view'],
                                                                        ch,
                                                                        background_level,
                                                                        # multiview_metadata_label %(ds,sample),
                                                                        info_dict,
                                                                        clean_pixels,  # do clean pixels
                                                                        False,  # do smooth pixels
                                                                        True,  #extract_rotation
                                                                        # True,#despeckle
                                                                        False,  #despeckle
                                                                        raw_input_binning,
                                                                        view_dict[view]['ill'],
                                                                        )

            elif 'tif' in view_dict[view]['filename'][-4:]:
                graph[multiview_view_fullres_label % (ds, sample, view, ch)] = (io_utils.read_stack_flexible,
                                                                        view_dict[view]['filename'],
                                                                        ch,
                                                                        view_dict[view]['origin'],
                                                                        view_dict[view]['spacing'],
                                                                        view_dict[view]['rotation'],
                                                                        background_level,
                                                                        )


            transf_label = transformed_view_label %(ds,sample,view,ch)
            transf_file = os.path.join(out_dir,transf_label)
            if os.path.exists(transf_file):
                graph[transf_label] = transf_file
            else:
                graph[transf_label] = (
                    # multiview.transform_view_and_save_chunked,
                    multiview.transform_view_dask_and_save_chunked,
                    os.path.join(out_dir,transformed_view_label %(ds,sample,view,ch)),
                    multiview_view_corr_label % (ds,sample,view,ch),
                    fusion_params_label % (ds,sample),
                    iview,
                    stack_properties_label % (ds,sample),
                    fusion_chunk_size,
                                                                )

            if ch == ref_channel_chrom or not perform_chromatic_correction:
                graph[multiview_view_corr_label %(ds,sample,view,ch)] = multiview_view_fullres_label %(ds,sample,view,ch)

            else:

                graph[multiview_view_corr_label %(ds,sample,view,ch)] = (
                    multiview.apply_chromatic_correction_parameters_center,
                    multiview_view_fullres_label % (ds,sample,view,ch),
                    chromatic_correction_params_label % (ds,sample,ref_channel_chrom,ch),
                                                                    )


    graph = io_utils.process_graph(graph)
    return graph