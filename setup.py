from os import path

from setuptools import setup, find_packages

# from setuptools import setup, find_packages, Extension

_dir = path.dirname(__file__)
with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='MVRegFus',
    version='0.1',
    packages=find_packages(),
    url='github.com/m-albert/MVRegFus',
    license='BSD 3-Clause License',
    author='Marvin Albert',
    author_email='marvin.albert@gmail.com',
    description='Python module for registration and fusion of multi-view light sheet imaging data',
    python_requires='>=3.5',

    classifiers=
        ['Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
     ],

     install_requires = [
                            'ipython',
                            'h5py',
                            'numpy',
                            'scipy',
                            'dask',
                            'distributed',
                            'bokeh',
                            'bcolz',
                            'dipy',
                            'scikit-image',
                            # 'numba',
                            'SimpleITK',
                        ],
)
