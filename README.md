# z1regfus

Code to register and fuse multi-view light sheet data from Zeiss Z1

## Dependencies

(probably not complete)

Python libraries:
- numpy, scipy
- h5py
- dask
- distributed
- dipy
- redis
- redis-lock
- SimpleITK
- scikit-image
- bcolz
- tifffile
- czifile

External:
- elastix (install binary and indicate path at the beginning of dipy_multiview.py)

## installation commands

### create a new conda environment and type:

conda install ipython h5py numpy scipy scikit-learn pandas dask distributed bokeh bcolz
pip install SimpleITK redis redis_lock dipy scikit-image transformations czifile tifffile

an addition might be needed:
pip install --upgrade scikit-image

download elastix binaries and place them in the same folder as z1regfus (this one)

## usage

- open anaconda prompt
- navigate to this directory "cd Z:\transfer\Z1_Multi_View_Fusion\z1regfus"
- type "conda activate Z:\transfer\Z1_Multi_View_Fusion\z1regfus_conda_env"
- use mv_dbspim.py as a template to create your own mv_dbspim_<project>.py in this folder
- run fusion with ipython mv_dbspim_<project>.py