import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='apply_coeff',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('apply_coeff', [
            'src/apply_coeff.cpp',
            'src/apply_coeff_cpu.cpp',
            'src/apply_coeff_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })