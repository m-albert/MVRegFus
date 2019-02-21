# z1regfus

Code to register and fuse multi-view light sheet data from Zeiss Z1

## Dependencies

(probably not complete)

Python libraries:
- numpy, scipy
- dask
- distributed
- dipy
- czifile (C. Goehlke)
- transformations (C. Goehlke)
- redis

External:
- elastix (install binary and indicate path at the beginning of dipy_multiview.py)