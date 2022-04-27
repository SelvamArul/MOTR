from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(
        [Extension('match', ['match.pyx'], include_dirs=[numpy.get_include()])]
    )
)
