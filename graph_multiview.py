__author__ = 'malbert'

import dipy_multiview
import numpy as np
import os
# import pickle
import io_utils

multiview_fused_label               = 'mv_%03d_%03d_c%02d.imagear.h5'
multiview_fusion_seg_label          = 'mv_fusion_seg_%03d_%03d_c%02d.imagear.h5'
multiview_view_reg_label            = 'view_reg_%03d_%03d_v%03d_c%02d'
multiview_view_fullres_label        = 'view_fullres_%03d_%03d_v%03d_c%02d'
multiview_view_corr_label           = 'view_corr_%03d_%03d_v%03d_c%02d'
multiview_metadata_label            = 'mv_metadata_%03d_%03d.dict.h5'
multiview_data_label                = 'mv_data_%03d_%03d.image.h5'
fusion_params_pair_label            = 'mv_params_%03d_%03d_pair%03d.prealignment.h5'
time_alignment_pair_params_label    = 'mv_time_alignment_pair_params_%03d_%03d.prealignment.h5'
time_alignment_params_label         = 'mv_time_alignment_params_%03d_%03d.prealignment.h5'
fusion_view_params_label            = 'mv_view_params_%03d_%03d.prealignment.h5'
fusion_params_label                 = 'mv_params_%03d_%03d.prealignment.h5' # final params, after combining view and time alignment
# chromatic_correction_params_label   = 'mv_params_%03d_%03d_c%02d.prealignment.h5'
chromatic_correction_params_label   = 'chromcorr_params_%03d_%03d_refch%02d_ch%02d.prealignment.h5'
stack_properties_label              = 'mv_stack_props_%03d_%03d.dict.h5'
transformed_view_label              = 'mv_transf_view_%03d_%03d_v%03d_c%02d.imagear.h5'
multiview_chrom_correction_channel_label = 'mv_chrom_corr_%03d_%03d_c%02d.imagear.h5'


def build_multiview_graph(
    out_dir,
    filepath,
    channels = [0,1],
    reg_channel = 0,
    ref_view = 1,
    ds = 0,
    sample = 0,
    mv_registration_bin_factors = np.array([4,4,1]),
    # mv_registration_bin_factors = np.array([4,4,1]),
    mv_final_spacing = np.array([3.]*3), # orig resolution
    # pairs = [[1,0],[1,2],[0,3],[2,3],[1,4],[4,5],[4,6],[6,7],[5,7]],
    pairs = [[1,0],[1,2],[0,3],[2,3]],
    background_level = 200,
    perform_chromatic_correction = True,
    dct_size = 10,
    dct_max_kernel = 5,
    dct_gaussian_kernel = 3,
    final_volume_mode = 'sample',
    # ref_channel_index_chrom = 0,
    chromatic_correction_file = None,
    time_alignment = False,
    time_alignment_ref_sample = 0,
    time_alignment_ref_view = 0,
    elastix_dir = '/scratch/malbert/dependencies_linux/elastix_linux64_v4.8',
    raw_input_binning = None, # x,y,z
    ):

    # ref_channel_chrom = channels[ref_channel_index_chrom]
    ref_channel_chrom = reg_channel

    all_views = np.sort(np.unique(np.array(pairs).flatten()))

    graph = dict()

    if time_alignment:
        graph[time_alignment_pair_params_label %(ds,sample)] = (
                                                                   dipy_multiview.register_linear_elastix,
                                                                   multiview_view_reg_label %(ds,time_alignment_ref_sample,time_alignment_ref_view,reg_channel),
                                                                   multiview_view_reg_label %(ds,sample,time_alignment_ref_view,reg_channel),
                                                                   1, # degree = 1 (trans + rotation)
                                                                   elastix_dir,
                                                                   )

        if os.path.exists(os.path.join(out_dir,time_alignment_params_label %(ds,sample))):
            graph[time_alignment_params_label %(ds,sample)] = os.path.join(out_dir,time_alignment_params_label %(ds,sample))
        else:
            graph[time_alignment_params_label %(ds,sample)] = (
                                                                        dipy_multiview.concatenate_view_and_time_params,
                                                                        os.path.join(out_dir,time_alignment_params_label %(ds,sample)),
                                                                        time_alignment_params_label %(ds,time_alignment_ref_sample),
                                                                        time_alignment_pair_params_label %(ds,sample),
                                                                )

    else:
        graph[time_alignment_params_label %(ds,sample)] = np.array([1.,0,0,0,1,0,0,0,1,0,0,0])

    for ipair,pair in enumerate(pairs):

        print('WARNING: set registration degree to 2 (trans+rot+aff) (standard for chemoatlas!)')
        graph[fusion_params_pair_label %(ds,sample,ipair)] = (
                                                                   dipy_multiview.register_linear_elastix,
                                                                   # dipy_multiview.register_linear_projections,
                                                                   # os.path.join('/tmp/', fusion_params_pair_label %(ds,sample,ipair)),
                                                                   multiview_view_reg_label %(ds,sample,pair[0],reg_channel),
                                                                   multiview_view_reg_label %(ds,sample,pair[1],reg_channel),
                                                                   2, # degree = 1 (trans + rotation)
                                                                   elastix_dir,
                                                                   )

    graph[fusion_params_label %(ds,sample)] = (
                                               dipy_multiview.get_final_params,
                                               os.path.join(out_dir,fusion_params_label %(ds,sample)),
                                               ref_view,
                                               pairs,
                                               [fusion_params_pair_label %(ds,sample,ipair) for ipair,pair in enumerate(pairs)],
                                               time_alignment_params_label %(ds,sample),
                                               )

    # graph[fusion_params_label %(ds,sample)] = io_utils.process_input_element('/data/malbert/atlas/data8/new_fusions_large/mv_params_%03d_%03d.prealignment.h5' %(ds,sample))

    if os.path.exists(os.path.join(out_dir,stack_properties_label %(ds,sample))):
        graph[stack_properties_label %(ds,sample)] = os.path.join(out_dir,stack_properties_label %(ds,sample))
    else:
        graph[stack_properties_label %(ds,sample)] = (
                                                    dipy_multiview.calc_stack_properties_from_views_and_params,
                                                    os.path.join(out_dir,stack_properties_label %(ds,sample)),
                                                    [multiview_view_corr_label %(ds,sample,view,reg_channel) for view in all_views],
                                                    fusion_params_label %(ds,sample),
                                                    mv_final_spacing,
                                                    final_volume_mode,
                                                    )

    graph[multiview_metadata_label %(ds,sample)] = (dipy_multiview.getStackInfoFromCZI,
                                                   filepath,
                                                   )

    # graph[multiview_data_label %(ds,sample)] = (dipy_multiview.readMultiviewCzi,
    #                                            filepath,
    #                                            background_level,
    #                                            multiview_metadata_label %(ds,sample),
    #                                            )


    for ich,ch in enumerate(channels):

        # graph[multiview_fused_label %(ds,sample,ch)] = (
        #                                     # dipy_multiview_test.fuse_views,
        #                                     dipy_multiview.fuse_views_content_orthogonal,
        #                                     os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
        #                                    [transformed_view_label %(ds,sample,view,ch) for view in all_views],
        #                                     ref_view,
        #                                    # fusion_params_label %(ds,sample),
        #                                    # mv_final_spacing,
        #                                    #  stack_properties_label %(ds,sample),
        #                                     1, # ax of rot
        #                                     3,# gaussian kernel size
        #                                     5,# window size
        #                                     50, # max_proj
        #                                    )

        # graph[multiview_fused_label %(ds,sample,ch)] = (
        #                                     # dipy_multiview_test.fuse_views,
        #                                     dipy_multiview.fuse_dct,
        #                                     os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
        #                                    [transformed_view_label %(ds,sample,view,ch) for view in all_views],
        #                                     dct_size, #size
        #                                     dct_max_kernel, #max_kernel
        #                                     dct_gaussian_kernel, #gauss_kernel
        #                                    )


        graph[multiview_fusion_seg_label %(ds,sample,ch)] = (
                                            # dipy_multiview_test.fuse_views,
                                            dipy_multiview.calc_lambda_fusion_seg,
                                            os.path.join(out_dir,multiview_fusion_seg_label %(ds,sample,ch)),
                                           [multiview_view_corr_label %(ds,sample,view,ch) for view in all_views],
                                            fusion_params_label %(ds,sample),
                                            # mv_final_spacing,
                                            stack_properties_label %(ds,sample),
                                            )

        # graph[multiview_fused_label %(ds,sample,ch)] = (
        #                                     # dipy_multiview_test.fuse_views,
        #                                     dipy_multiview.fuse_views_lambda,
        #                                     os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
        #                                    [multiview_view_corr_label %(ds,sample,view,ch) for view in all_views],
        #                                     fusion_params_label %(ds,sample),
        #                                     # mv_final_spacing,
        #                                     stack_properties_label %(ds,sample),
        #                                     multiview_fusion_seg_label %(ds,sample,ch),
        #                                     )

        print('WARNING: FUSING WITH LR (with dct weights)')
        # graph[multiview_fused_label %(ds,sample,ch)] = (
        #                                     dipy_multiview.fuse_LR_with_weights_dct,
        #                                     os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
        #                                    [multiview_view_corr_label %(ds,sample,view,ch) for view in all_views],
        #                                     fusion_params_label %(ds,sample),
        #                                     # mv_final_spacing,
        #                                     stack_properties_label %(ds,sample),
        #                                     20, # iters
        #                                     5,  # sigma z
        #                                     0.1,# sigma xy
        #                                     )

        # print('WARNING: FUSING ADDITIVE WITH DCT WEIGHTS')
        # graph[multiview_fused_label %(ds,sample,ch)] = (
        #                                     dipy_multiview.fuse_dct,
        #                                     os.path.join(out_dir,multiview_fused_label %(ds,sample,ch)),
        #                                    [multiview_view_corr_label %(ds,sample,view,ch) for view in all_views],
        #                                     fusion_params_label %(ds,sample),
        #                                     # mv_final_spacing,
        #                                     stack_properties_label %(ds,sample),
        #                                     )

        if chromatic_correction_file is not None:
            graph[multiview_chrom_correction_channel_label %(ds,sample,ch)] = (dipy_multiview.readStackFromMultiviewMultiChannelCzi,
                                                                chromatic_correction_file,
                                                                0,
                                                                ch,
                                                                background_level,
                                                                None, #infoDict
                                                                False, #do_clean_pixels
                                                                False,#do_smooth
                                                                False,#extract_rotation
                                                                False,#despeckle
                                                                raw_input_binning,
                                                                )
        else:
            graph[multiview_chrom_correction_channel_label %(ds,sample,ref_channel_chrom)]  = multiview_view_reg_label %(ds,sample,ref_view,ref_channel_chrom)
            graph[multiview_chrom_correction_channel_label %(ds,sample,ch)]                 = multiview_view_reg_label %(ds,sample,ref_view,ch)

        if ch != ref_channel_chrom:
            graph[chromatic_correction_params_label %(ds,sample,ref_channel_chrom,ch)] = (
                                           dipy_multiview.get_chromatic_correction_parameters_center,
                                           os.path.join(out_dir,chromatic_correction_params_label %(ds,sample,ref_channel_chrom,ch)),
                                           # multiview_view_reg_label %(ds,sample,ref_view,ref_channel_chrom),
                                           # multiview_view_reg_label %(ds,sample,ref_view,ch),
                                           multiview_chrom_correction_channel_label %(ds,sample,ref_channel_chrom),
                                           multiview_chrom_correction_channel_label %(ds,sample,ch),
                                           )
            # select view from big image container
        for view in all_views:

            # graph[multiview_view_reg_label %(ds,sample,view,ch)] = (
            #                                                     # lambda x,c,v: x[c][v],
            #                                                     dipy_multiview.return_binned_view_ch,
            #                                                     multiview_data_label %(ds,sample),
            #                                                     view,
            #                                                     ch,
            #                                                     mv_registration_bin_factors,
            #                                                     )

            # graph[multiview_view_fullres_label %(ds,sample,view,ch)] = (lambda x,v,c: x[c][v],
            #                                                     multiview_data_label %(ds,sample),
            #                                                     view,
            #                                                     ch,
            #                                                     )
            graph[multiview_view_reg_label %(ds,sample,view,ch)] = (
                                                                # lambda x,c,v: x[c][v],
                                                                dipy_multiview.bin_stack,
                                                                multiview_view_fullres_label %(ds,sample,view,ch),
                                                                mv_registration_bin_factors,
                                                                )

            # print('NOT CLEANING NOR SMOOTHIN PIXELS!')
            print('NOT CLEANING PIXELS!')
            # print('WARNING: NOOOOOOOO DESPECKLING')
            graph[multiview_view_fullres_label %(ds,sample,view,ch)] = (dipy_multiview.readStackFromMultiviewMultiChannelCzi,
                                                                filepath,
                                                                view,
                                                                ch,
                                                                background_level,
                                                                multiview_metadata_label %(ds,sample),
                                                                False, # do clean pixels
                                                                False, # do smooth pixels
                                                                True,#extract_rotation
                                                                # True,#despeckle
                                                                False,#despeckle
                                                                raw_input_binning,
                                                                )

            graph[transformed_view_label %(ds,sample,view,ch)] = (
                                                                dipy_multiview.transform_view_with_decorator,
                                                                os.path.join(out_dir,transformed_view_label %(ds,sample,view,ch)),
                                                                multiview_view_corr_label %(ds,sample,view,ch),
                                                                fusion_params_label %(ds,sample),
                                                                view,
                                                                stack_properties_label %(ds,sample)
                                                                )

            if ch == ref_channel_chrom or not perform_chromatic_correction:
                graph[multiview_view_corr_label %(ds,sample,view,ch)] = multiview_view_fullres_label %(ds,sample,view,ch)

            else:

                graph[multiview_view_corr_label %(ds,sample,view,ch)] = (
                                                                    dipy_multiview.apply_chromatic_correction_parameters_center,
                                                                    multiview_view_fullres_label %(ds,sample,view,ch),
                                                                    chromatic_correction_params_label %(ds,sample,ref_channel_chrom,ch),
                                                                    )

    return graph