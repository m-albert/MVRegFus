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

External:
- elastix (install binary and indicate path at the beginning of dipy_multiview.py)

## installation commands


conda install ipython h5py numpy scipy scikit-image scikit-learn pandas dask distributed bokeh
pip install SimpleITK bcolz redis redis-lock