#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import print_function

from setuptools import setup, find_packages, distutils, Extension, command
from setuptools.command import develop
import posixpath
import contextlib
import distutils.ccompiler
import distutils.command.clean
import glob
import inspect
import multiprocessing
import multiprocessing.pool
import os
import platform
import re
import requests
import shutil
import subprocess
import sys
import tempfile
import zipfile
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
base_dir = os.path.dirname(os.path.abspath(__file__))


def get_build_version():
  version = os.getenv("LYNX_VERSION", "0.1.0")
  return version


# Read in README.md for our long_description
cwd = os.path.dirname(os.path.abspath(__file__))

# dylib_path = {
#     "build/bazel-bin/lynx/csrc/runtime/_LYNXC_RT.so": "lynx/_LYNXC_RT.so"
# }

# for src, dst in dylib_path.items():
#   os.system("cp %s/%s %s/%s" % (cwd, src, cwd, dst))
os.environ["TORCH_CUDA_ARCH_LIST"]="8.0;8.6;9.0"
setup(
    name="lynx",
    version=get_build_version(),
    description="Asynchronous Distributed Dataflow ML System",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8.0",
    packages=find_packages(include=["lynx*"]),
    package_data={
        "lynx": ["*.so*",],
    },
    ext_modules=[
        CUDAExtension(
            name='lynx.models.cuda_extension',
            sources=['lynx/models/custom_func.cu'],
            extra_compile_args={
                "nvcc": [
                "-arch=sm_86","-arch=sm_90","-g","-G"]
            },
            libraries=['cuda', 'cudart'],  # 链接CUDA驱动和运行时库
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    extras_require={},
)

# for src, dst in dylib_path.items():
  # os.remove("%s/%s" % (cwd, dst))
