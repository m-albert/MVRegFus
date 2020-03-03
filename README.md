# MVRegFus

Multi-View Registration and Fusion

Cross platform python module to process multi-view light sheet data. This includes

1) image registration
    - view registration
    - time registration (drift correction)
    - channel registration (affine chromatic aberration correction)
    
2) view fusion
    - weighted additive fusion
    - multi-view deconvolution (Richardson Lucy)
        - traditional
        - weighted
    - for both fusion methods, weights can be chosen from:
        - blending weights
        - image quality based weights
        
- Additional features:

    - cross-platform
    - fusion is performed in a blockwise manner, allowing large datasets to be fused on a laptop
    - can be deployed on cluster using dask.distributed
    - GPU accelerated multi-view deconvolution

Notes:
- currently, only czi files from Z1 microscopes are supported out of the box

## Dependencies

Python libraries:
- numpy, scipy
- h5py
- dask
- distributed
- dipy
- SimpleITK
- scikit-image
- bcolz
- tifffile (included)
- czifileczifile==2019.1.26 (included)
- cupy (optional)

External:
- elastix (install binary and indicate path at the beginning of dipy_multiview.py)
- SimpleElastix (optional)

## Installation instructions

1) Use anaconda and install an environment from the provided .yml file:
"conda env create --file mv_environment.yml"

2) Download elastix (binary version suitable for your platform) and place files into a folder 'elastix' in the same folder as this project

## Usage

- open terminal / anaconda prompt
- activate the previously installed anaconda environment: "conda activate mvregfus" (Win,MacOS) or "source activate mvregfus" (Linux)
- copy mvregfus/bin/mvregfus_bin.py to <your_mvregfus_bin.py> (can be placed next to data) and use as a template to indicate the location of your files, resolution, etc. (see comments in file). Soon configuration file handling will be added
- run fusion with python <your_mvregfus_bin.py>