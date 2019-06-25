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
- tifffile (included)
- czifileczifile==2019.1.26 (included)

External:
- elastix (install binary and indicate path at the beginning of dipy_multiview.py)
- SimpleElastix (optional)

## installation instructions

- download anaconda, then:

### option 1: install an environment from the .yml file:
conda env create --file mv_environment.yml

### option 2 (deprecated): create a new conda environment and type:

# conda install ipython h5py numpy scipy scikit-learn pandas dask distributed bokeh bcolz
# pip install SimpleITK
# pip install SimpleITK redis redis_lock dipy scikit-image transformations czifile==2019.1.26 tifffile

download elastix (binary version) and place files in a folder 'elastix' in the same folder as z1regfus (this one)

## usage

- open anaconda prompt
- navigate to this directory "cd Z:\transfer\Z1_Multi_View_Fusion\z1regfus"
- type "conda activate Z:\transfer\Z1_Multi_View_Fusion\z1regfus_conda_env"
- use mv_dbspim.py as a template to create your own mv_dbspim_<project>.py in this folder
- run fusion with ipython mv_dbspim_<project>.py