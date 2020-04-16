#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='banddownfolder',
    version='0.1',
    description='Downfold Hamiltonian',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    package_data={},
    install_requires=['numpy', 'scipy',  'matplotlib', 'ase', 'numba', 
        'tbmodels', 
        'pythtb',
        'netcdf4',
        'jupyter',
        ],
    scripts=[
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
    ])
