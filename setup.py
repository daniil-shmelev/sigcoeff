import os.path
from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

ext_modules = [
    Extension(
        name="cython_backend",
        sources=["sigcoeff/cython_backend.pyx"],
        include_dirs = [".", np.get_include()],
        extra_compile_args=['-openmp'],
        extra_link_args=['-openmp'],
    )
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name="sigcoeff",
    version="1.0",
    description="Sparse signature coefficient computations via kernels",
    packages=["sigcoeff"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Win32 (MS Windows)",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    url="https://github.com/daniil-shmelev/sigcoeff",
    author="Daniil Shmelev",
    author_email="daniil.shmelev23@imperial.ac.uk",
    setup_requires=["cython"],
    install_requires=["torch>=2.4.1",
    "numba>=0.60.0",
    "cython>=3.0.11",
    "numpy>=1.26.4",
    "setuptools>=74.1.2"
    ],
    extras_require={
        "dev": [
            "check-manifest",
            "twine",
            "black",
        ],
    },
    ext_modules=cythonize(ext_modules),
    package_data = {"sigcoeff": ["*.pyx", "*.pxd", "*.so", "*.pyd"]},
    include_package_data=True
)