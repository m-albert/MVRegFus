__author__ = 'malbert'

import os

import SimpleITK as sitk
import h5py
import numpy as np
from dask.diagnostics import ProgressBar

# from dipy.align.imwarp import DiffeomorphicMap
from mvregfus.image_array import ImageArray


# import h5pyswmr # for locking
# for locking:
# import redis_lock
# from redis import StrictRedis
# conn = StrictRedis()


def get(graph,key,local=True):
    # new function because from version 19.1 on, dask no longer culls graphs (apparently it's only done by distributed)
    # so the function combines a get call with previous culling (which means to modify a graph in such a way that only the relevant bit to the key remains)

    import dask
    from dask.optimization import cull
    cgraph = cull(graph,key)[0]

    # with ProgressBar():
    if local:
        return dask.local.get_sync(cgraph,key)
    else:
        return dask.threaded.get(cgraph,key)
        # return dask.multiprocessing.get(cgraph,key)

def recursive_func_application(l,f):
    if type(l) == list:
        for ie,e in enumerate(l):
            l[ie] = recursive_func_application(e,f)
        return l
    else:
        return f(l)

# def recursive_func_application_with_linear_output(l,f,outlist):
#     if type(l) == list:
#         for ie,e in enumerate(l):
#             recursive_func_application_with_linear_output(e,f,outlist)
#     else:
#         outlist.append(f(l))
#     return

def recursive_func_application_with_list_output(l,f):
    result = []
    if type(l) == list:
        for ie,e in enumerate(l):
            result += recursive_func_application_with_list_output(e,f)
            # result.append(recursive_func_application_with_list_output(e,f))
    else:
        # outlist.append(f(l))
        result += [f(l)]
    # print(result)
    return result

def process_input_element(path):

    if not is_io_path(path): return path

    # if '_list_of_' in path:
    #     split = path.split('_list_of_')
    #     import glob
    #     files = glob.glob('*'.join(split))
    #     files.sort()
    #     out = []
    #     for ii in range(len(files)):
    #         it_path = split[0] + '%03d' %ii + split[1]
    #         tmp = process_input_element(it_path)
    #         out.append(tmp)
    #     return out

    # if not type(path) == str or (type(path) == str and len(path.split('.'))==1) or 'elastix_' in path:
    #     return path
    # else:
    #     try:
    #         lock_identifier = 'malbert_lock_'+path
    #         lock = redis_lock.Lock(conn, lock_identifier)
    #         lock.acquire()
    #         # print('acquired lock %s' %lock_identifier)
    #     except:
    #         print('locking not working')


    if path.endswith('.mhd') or path.endswith('.tif'):
        s = sitk.ReadImage(path)
        ar = ImageArray(sitk.GetArrayFromImage(s))
        ar.spacing = np.array(s.GetSpacing()[::-1])
        ar.origin = np.array(s.GetOrigin()[::-1])
        res = ar

    elif path.endswith('ims'):
        res = h5py.File(path, mode='r')['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'][()]

        if path.endswith('image.ims'):
            from mvregfus import imaris
            meta = imaris.get_meta_from_ims(path)
            res = ImageArray(res,
                             spacing = meta['spacing'][::-1],
                             origin = meta['origin'][::-1],
                             rotation = 0,
                             )

    elif path.startswith('prealignment') and path.endswith('.h5'):
        res =  h5py.File(path,mode='r')['prealignment'][()]

    elif path.endswith('dict.h5'):
        tmpFile = h5py.File(path,mode='r')
        tmpdict = dict()
        # for key,value in enumerate(tmpFile):
        for key,value in tmpFile.items():
            tmpdict[key] = tmpFile[key][()]
        tmpFile.close()
        res = tmpdict

    elif path.endswith('.image.h5'):
        res = h5py.File(path,mode='r')['image'][()]

    elif path.endswith('.imagear.h5'):
        tmpFile = h5py.File(path,mode='r')
        tmp = ImageArray(tmpFile['array'][()])
        tmp.origin = np.array(tmpFile['origin'][()])
        tmp.spacing = np.array(tmpFile['spacing'][()])
        tmp.rotation = np.array(tmpFile['rotation'][()])
        res = tmp

    elif path.endswith('hdf'):
        res = pd.read_hdf(path)

    elif 'prealignment' in path and path.endswith('.h5'):
        res = h5py.File(path,mode='r')['prealignment'][()]

    elif path.endswith('ilp'):
        res = path

    elif path.endswith('pmap'):
        tmpFile = open(path, 'w')
        import pickle
        res = pickle.load(tmpFile)
        tmpFile.close()

    else:
        raise(Exception('unrecognized string input to function'))

    # try:
    #     lock.release()
    #     # print('released lock %s' %lock_identifier)
    # except:
    #     pass

    ## h5pyswmr.locking.release_lock(h5pyswmr.locking.redis_conn,'process_input_element',lock_identifier)


    return res


def get_mtime_from_path(path):
    # print('GETMTIME...')
    if not is_io_path(path): return 0
    # if not type(path) == str or (type(path) == str and len(path.split('.'))==1):
    #     return 0
    elif os.path.exists(path):
        # return os.path.getmtime(path)
        return os.lstat(path).st_mtime # this function doesn't follow symlinks which is good when base data is linked
    else:
        return -1


def process_output_element(element,path):
    # if os.path.exists(path):
    #     print('SKIPPING THE WRITING OF THE RESULT %s, SINCE IT ALREADY EXISTS. WAS IT WRITTEN BY SOME OTHER PROCESS IN THE MEANTIME?' %path)
    #     return path

    # if '_list_of_' in path:
    #     for ii in range(len(element)):
    #         split = path.split('_list_of_')
    #         it_path = ('%03d' %ii).join(split)
    #         process_output_element(element[ii],it_path)
    #     return path

    # l = h5pyswmr.locking.acquire_lock(h5pyswmr.locking.redis_conn,'process_input_element',lock_identifier)
    # try:
    #     lock_identifier = 'malbert_lock_'+path
    #     lock = redis_lock.Lock(conn, lock_identifier)
    #     lock.acquire()
    #     # print('acquired lock %s' %lock_identifier)
    # except:
    #     print('locking not working (maybe try io_utils.redis_lock.reset_all(io_utils.conn)')

    if path.endswith('.mhd'):
        s = sitk.GetImageFromArray(element)
        if type(element) == ImageArray:
            s.SetSpacing(element.spacing[::-1])
            s.SetOrigin(element.origin[::-1])
        sitk.WriteImage(s,path)
    # elif type(element) == DiffeomorphicMap:
    #     diffmap = diffmap_on_disk(path)
    #     diffmap.save(element)
    if path.endswith('.ims'):
        if os.path.exists(path):
            print('removing existing imaris file')
            os.remove(path)
        from mvregfus import imaris
        if type(element) == ImageArray:
            dx = element.spacing[1]
            dz = element.spacing[0] # is z the first or last coordinate in an imaris hdf5?
            origin = element.origin[::-1]
        elif type(element) == np.ndarray:
            dx = 1.
            dz = 1.
            origin = np.zeros(3,dtype=np.float32)
        imaris.np_to_ims(element, path,
                         subsamp=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)),
                         chunks=((16, 128, 128), (64, 64, 64), (32, 32, 32), (16, 16, 16)),
                         compression='gzip',
                         dx=float(dx), dz=float(dz),
                         origin=origin,
                         )
    elif path.endswith('.image.h5') and type(element) == np.ndarray:
        tmpFile = h5py.File(path)
        tmpFile.clear()
        tmpFile['image'] = element
        tmpFile.close()
    #image_array.ImageArray
    elif path.endswith('.imagear.h5') and type(element) == ImageArray:
        tmpFile = h5py.File(path)
        tmpFile.clear()
        chunks = np.min([[128]*3,element.shape],0)
        chunks = tuple(chunks)
        tmpFile.create_dataset("array", data=np.array(element), chunks=chunks, compression="gzip")
        # tmpFile['array'] = np.array(element)
        tmpFile['spacing'] = element.spacing
        tmpFile['origin'] = element.origin
        tmpFile['rotation'] = element.rotation
        tmpFile.close()

    # elif path.endswith('.imagear.n5') and type(element) == ImageArray:
    #     chunks = np.min([[50,50,50],element.shape],0)
    #     chunks = tuple(chunks)
    #     import zarr
    #     store = zarr.N5Store(path)
    #     z = zarr.zeros(element.shape, chunks=chunks, store=store, overwrite=True)
    #     z[...] = element

    # elif path.startswith('prealignment') and path.endswith('.h5') and type(element) == np.ndarray:
    elif 'prealignment' in path and path.endswith('.h5') and type(element) == np.ndarray:
        tmpFile = h5py.File(path)
        tmpFile.clear()
        tmpFile['prealignment'] = element
        tmpFile.close()
    elif path.endswith('dict.h5') and type(element) == dict:
        tmpFile = h5py.File(path)
        tmpFile.clear()
        for key,value in element.items():
            tmpFile[key] = value
        tmpFile.close()
    elif path.endswith('hdf'): #pandas dataframe or panel
        # element.to_hdf(path,'pandas',format='table')
        element.to_hdf(path,'pandas')
    elif path.endswith('pmap'):
        tmpFile = open(path,'w')
        import pickle
        pickle.dump(element,tmpFile)
        tmpFile.close()
    # elif path.endswith('.mesh.h5'):
    #     # dict with keys: 'labels' and 'vertices_%06d', 'faces_%06d' %label
    #     tmpFile = h5py.File(path)
    #     for
    #     tmpFile['labels'] = element['labels']
    #     tmpFile.clear()
    #     for i in
    #     # tmpFile['prealignment'] = element
    #     tmpFile.close()
    else:
        raise(Exception('unrecognized string output from function'))

    try:
        lock.release()
        # print('released lock %s' %lock_identifier)
    except:
        pass
    # h5pyswmr.locking.release_lock(h5pyswmr.locking.redis_conn,'process_input_element',lock_identifier)
    # print('released lock %s' %lock_identifier)

    return path


# def process_output_element(element,path):
#     # if os.path.exists(path):
#     #     print('SKIPPING THE WRITING OF THE RESULT %s, SINCE IT ALREADY EXISTS. WAS IT WRITTEN BY SOME OTHER PROCESS IN THE MEANTIME?' %path)
#     #     return path
#     import time
#     not_successful = True
#     while not_successful:
#
#         if not type(path) == str:
#             return path
#         elif path.endswith('.mhd'):
#             s = sitk.GetImageFromArray(element)
#             if type(element) == ImageArray:
#                 s.SetSpacing(element.spacing[::-1])
#                 s.SetOrigin(element.origin[::-1])
#             sitk.WriteImage(s,path)
#             return path
#         elif type(element) == DiffeomorphicMap:
#             diffmap = diffmap_on_disk(path)
#             diffmap.save(element)
#             return path
#         elif path.endswith('.image.h5') and type(element) == np.ndarray:
#             tmpFile = h5py.File(path)
#             tmpFile.clear()
#             tmpFile['image'] = element
#             tmpFile.close()
#             return path
#         #image_array.ImageArray
#         elif path.endswith('.imagear.h5') and type(element) == ImageArray:
#             tmpFile = h5py.File(path)
#             tmpFile.clear()
#             tmpFile['array'] = np.array(element)
#             tmpFile['spacing'] = element.spacing
#             tmpFile['origin'] = element.origin
#             tmpFile['rotation'] = element.rotation
#             tmpFile.close()
#             return path
#         # elif path.startswith('prealignment') and path.endswith('.h5') and type(element) == np.ndarray:
#         elif 'prealignment' in path and path.endswith('.h5') and type(element) == np.ndarray:
#             tmpFile = h5py.File(path)
#             tmpFile.clear()
#             tmpFile['prealignment'] = element
#             tmpFile.close()
#             return path
#         elif path.endswith('dict.h5') and type(element) == dict:
#             tmpFile = h5py.File(path)
#             tmpFile.clear()
#             for key,value in element.items():
#                 tmpFile[key] = value
#             tmpFile.close()
#             return path
#         elif path.endswith('hdf'): #pandas dataframe or panel
#             # element.to_hdf(path,'pandas',format='table')
#             element.to_hdf(path,'pandas')
#             return path
#         else:
#             raise(Exception('unrecognized string output from function'))
#
#         not_successful = False
#         print('PROBLEM WHILE WRITING RESULT %s, RETRYING...' %path)
#         time.sleep(0.5)
#     return


def is_io_path(path):
    if not type(path) == str or (type(path) == str and len(path.split('.'))==1) or 'elastix' in path or path.endswith('.czi'):
        return False
    else:
        return True


def io_decorator_distributed(func):

    """
    decorator to use for io handling with functions that take as first argument an output filepath and  as further arguments strings that
    :param func:
    :return:
    """

    def full_func(*args, **kwargs):
        # global funcs_to_debug

        # funcs_to_debug = []
        print('DECORATOR distributed...',args)

        if not is_io_path(args[0]):
            return func(*args,**kwargs)

        # fileworker_address = '10.11.8.149:8795'
        # fileworker_address = '10.11.8.149'

        # mtimes = []
         # this should be processed on a fileworker
        # recursive_func_application_with_linear_output(list(args),get_mtime_from_path,mtimes)
        # try:

        from distributed import worker_client
        with worker_client(timeout=1000) as e:

            mtimes = e.submit(recursive_func_application_with_list_output,
                     *(list(args),get_mtime_from_path),
                     resources={'files':1}).result()
            # except:
            #     mtimes = recursive_func_application_with_list_output(list(args),get_mtime_from_path)


            # recursive_func_application_with_linear_output(list(args),get_mtime_from_path,mtimes)
            # print(mtimes)
            highest_mtime = np.array(mtimes[1:]).max()

            # print funcs_to_debug[0].orig_name
            # print func

            if not mtimes[0] == -1:
            # if os.path.exists(args[0]):
            #     if func.func_name not in [i.orig_name for i in funcs_to_debug]:
            #     if os.path.getmtime(args[0]) >= highest_mtime:

                if mtimes[0] >= highest_mtime:
                    return args[0]

            # print('here its calculating!')

            nargs = []
            for iarg,arg in enumerate(args):
                if not iarg: continue
                # try:
                # with worker_client() as e:
                res = e.submit(
                    recursive_func_application,
                    *(arg,process_input_element),
                     resources={'files':1}).result()
                # except:
                #     res = recursive_func_application(arg,process_input_element)
                # print(arg,res)
                nargs.append(res)
                # nargs.append(recursive_func_application(arg,process_input_element))

            # print('la'+str(args))
            result = func(*nargs,**kwargs)
             # this should be processed on a fileworker
            # try:
            nresult = e.submit(process_output_element,*(result,args[0]),
                               resources={'files':1}).result()
            # except:
            #     nresult = process_output_element(result,args[0])
        return nresult

    # full_func.orig_name = func.func_name
    full_func.orig_name = func.__name__

    return full_func

def process_graph(graph):
    new_graph = graph.copy()
    for key,value in graph.items():
        if type(value) == tuple:
            if 'is_io_func' in value[0].__dict__.keys():
                if is_io_path(value[1]) and os.path.exists(value[1]):
                    print('using existing %s' %value[1])
                    new_graph[key] = value[1]

    return new_graph


def io_decorator_local(func):

    """
    decorator to use for io handling

    checks whether arguments are io paths using is_io_path
    and if so, loads them into memory

    if the first argument is an is_io_path, it writes the result
    of the function application to this path

    :param func:
    :return:
    """

    def full_func(*args, **kwargs):

        print('DECORATOR local... %s' %(func.__name__,))
        # if not is_io_path(args[0]):
        #     return func(*args,**kwargs)

        if is_io_path(args[0]):
            mtimes = recursive_func_application_with_list_output(list(args),get_mtime_from_path)

            highest_mtime = np.array([i for i in mtimes[1:]]).max()

            if not mtimes[0] == -1:
                if not np.any(np.array(mtimes[1:]) == -1) and get_mtime_from_path(args[0]) >= highest_mtime:
                    return args[0]

        nargs = []
        for iarg,arg in enumerate(args):
            if is_io_path(args[0]) and not iarg: continue
            res = recursive_func_application(arg,process_input_element)

            nargs.append(res)

        # if 'map2' in args[0]:
        # #     print(args)
        # #     print(mtimes)

        if is_io_path(args[0]):
            print('producing %s' %args[0])

        result = func(*nargs,**kwargs)

        if is_io_path(args[0]):
            result = process_output_element(result,args[0])
        return result

    full_func.orig_name = func.__name__
    full_func.is_io_func = True

    return full_func

# io_decorator = io_decorator_distributed
io_decorator = io_decorator_local
