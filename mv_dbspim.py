import sys
# import dipy_multiview_test
import numpy as np
import pdb,os,sys
from scipy import misc
import dask
import pickle

# import tifffile
import importlib

# print('watch out, cache!')
# from dask.cache import Cache
# cache_size = 50e9 #20gb
# cache = Cache(cache_size)  # Leverage two gigabytes of memory
# cache.register()    # Turn cache on globall

import sys
elastix_dir = os.path.join(sys.path[0],'elastix-4.9.0-win64')

import graph_multiview

graph_multiview.multiview_fused_label = graph_multiview.multiview_fused_label[:-2]+'mhd'
graph_multiview.transformed_view_label = graph_multiview.transformed_view_label[:-2]+'mhd'

if __name__ == '__main__':

    from dask import multiprocessing,threaded

    # filepaths = [
    #     '/data/malbert/lucien/Fish1.czi',
    #     '/data/malbert/lucien/Fish2.czi',
    #     '/data/malbert/lucien/Venus1.czi',
    #     '/data/malbert/lucien/Venus2.czi',
    #     '/data/malbert/lucien/Venus3.czi',
    #     '/data/malbert/lucien/Venus4.czi',
    #     '/data/malbert/lucien/Venus5.czi',
    #     '/data/malbert/lucien/Venus6.czi',
    # ]
    # filepaths = ['/data/malbert/data/dbspim/chemokine/20180319_36hpf_4_gfp_7_rfp/4g7r_06.czi']+['/data/malbert/data/dbspim/chemokine/20180319_36hpf_4_gfp_7_rfp/4g7r_06(%s).czi' %i for i in range(1,18)]
    # filepaths = ['/data/malbert/data/dbspim/chemokine/20180606_36dpf_lyntimer/lyntimer_7m_hetero_tl_2(%s).czi' %i for i in range(1,11)]
    # filepaths = ['/data/malbert/data/dbspim/Lucien/2018-07-11/%s' %fn for fn in ['Fish1_t2.czi','Fish2_t2.czi','Fish3.czi','Fish4.czi','Fish5.czi']]
    # filepaths = ['/data/malbert/data/dbspim/Lucien/2018-07-25/%s' %fn for fn in ['Fish1.czi','Stack1.czi','Stack2.czi']]
    # filepaths = ['/data/malbert/lucien/2018-06-14/Fish1.czi','/data/malbert/lucien/2018-06-14/Fish2.czi']
    filepaths = [os.path.join(sys.path[0],'../__for_Marvin/SPIRIT-cldnbGFP-bact_h2a_mcherry_24hpf.czi')]

    channelss = [[0]]*len(filepaths)
    reg_channel = 0
    # pairss = [[[0,1],[1,2],[2,3],[3,0],[1,4],[4,5],[4,6],[6,7]]]*len(filepaths)
    # pairss = [[[0,1],[1,2],[2,3],[3,0],[1,4],[4,5],[4,6],[6,7]]]*len(filepaths)

    graph = dict()
    result_keys = []
    for ifile,filepath in enumerate(filepaths):
        channels = channelss[ifile]
        # pairs = pairss[ifile]

        print('Warning: Using different registration factors, default is 882')
        graph.update(
            graph_multiview.build_multiview_graph(
            filepath = filepath,
            # pairs = [[0,1],[1,2],[2,3],[3,0]],
            pairs = [[0,1],[1,2],[2,3],[4,3],[5,4],[5,0]],
            # pairs = [[0,1]],#,[1,2],[2,3],[3,0]],
            # pairs = pairs,
            ref_view = 0,
            # mv_registration_bin_factors = np.array([8,8,2]),
            mv_registration_bin_factors = np.array([8,8,4]),
            # mv_final_spacing = np.array([10.]*3), # orig resolution
            # mv_final_spacing = np.array([1.]*3), # orig resolution
            # mv_final_spacing = np.array([1.06]*3), # orig resolution
            mv_final_spacing = np.array([4.]*3), # orig resolution
            reg_channel = reg_channel,
            channels = channels,
            ds = 0,
            sample = ifile,
            out_dir = os.path.dirname(filepath),
            perform_chromatic_correction = False,
            # perform_chromatic_correction = False,
            # dct_size = 10, #size
            # dct_max_kernel = 7, #max_kernel
            # dct_gaussian_kernel = 7, #gauss_kernel
            # dct_size = 50, #size
            # dct_max_kernel = 11, #max_kernel
            # dct_gaussian_kernel = 7, #gauss_kernel
            final_volume_mode = 'sample',
            elastix_dir = elastix_dir,
            )
        )

        # choose same reference coordinate system
        # if ifile:
        #     graph[graph_multiview.stack_properties_label %(0,ifile)] = graph_multiview.stack_properties_label %(0,0)

        if os.path.exists(os.path.join(os.path.dirname(filepath),graph_multiview.multiview_fused_label %(0,ifile,0))):
            continue

        multiview_fused_labels              = [graph_multiview.multiview_fused_label %(0,ifile,ch) for ch in channels]
        # fusion_params_label                 = 'mv_params_%03d_%03d.prealignment.h5' %(ikey,s)
        result_keys += multiview_fused_labels
            # p = threaded.get(graph,fusion_params_label)
    o = dask.local.get_sync(graph,result_keys)
    #
    # N = 1
    # results = []
    # for i in range(0,len(result_keys),N*len(channelss[0])):
    #     # try:
    #     # results.append(threaded.get(graph,result_keys[i:i+N*len(channels)],num_workers=23))
    #     results.append(multiprocessing.get(graph,result_keys[i:i+N*len(channelss[0])],num_workers=23))
    #     # results.append(dask.local.get_sync(graph,result_keys[i:i+N*len(channels)]))
    #     # results.append(multiprocessing.get(graph,result_keys[i:i+N*len(channels)],num_workers=23))
    #     # except:
    #     #     pass

    # def execute(key):
    #     return dask.local.get_sync(graph,key)
    #
    # import multiprocessing
    # # p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    # p = multiprocessing.Pool(processes = 3)
    # p.map(execute, [result_keys[i:i+len(channelss[0])] for i in range(0,len(result_keys),len(channelss[0]))])

    # N = 1
    # results = []
    # for i in range(0,len(result_keys),N*len(channels)):
    #     results.append(dask.local.get_sync(graph,result_keys[i:i+N*len(channels)]))




    # o = threaded.get(graph,result_keys,num_workers=10)
    # o = dask.local.get_sync(graph,result_keys)
    # o = dask.local.get_sync(graph,'')
    # o = threaded.get(graph,'mv_001_003_c01.imagear.h5',num_workers=4)
    # o = dask.local.get_sync(graph,result_keys)
    # o = dask.local.get_sync(graph,graph_multiview.transformed_view_label %(13,0,0,0))


            # io_utils.process_output_element(p,os.path.join(out_dir,fusion_params_label))
            # [io_utils.process_output_element(o[i],os.path.join(out_dir,multiview_fused_labels[i])) for i in range(len(multiview_fused_labels))]
            # tifffile.imsave(tmp_out,o)
