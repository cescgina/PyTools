# Run the following line to compile atomset package
# python setup.py build_ext --inplace

import numpy
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from distutils.extension import Extension
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True
from distutils.command.sdist import sdist as _sdist

class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are
        # up-to-date
        from Cython.Build import cythonize
        cythonize(['cython/mycythonmodule.pyx'])
        _sdist.run(self)
        cmdclass['sdist'] = sdist

here = path.abspath(path.dirname(__file__))
ext_modules = []
cmdclass = {}
# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

if use_cython:
        ext_modules += [
            Extension("PyTools.utils.utils", ["PyTools/utils/utils.pyx"], include_dirs = ["PyTools", "PyTools/utils"]),
                ]
        cmdclass.update({ 'build_ext': build_ext })
else:
        ext_modules += [
            Extension("PyTools.utils.utils", ["PyTools/utils/utils.c"], include_dirs = ["PyTools", "PyTools/utils"]),
                ]

setup(
    name="PyTools",
    version="0.1",
    description='Bundle of useful scripts for my PhD',
    long_description=long_description,
    url="https://github.com/cescgina/PyTools",
    author='Joan Francesc Gilabert',
    author_email='cescgina@gmail.com',
    license='',
    packages=find_packages(exclude=['docs', 'tests']),
    package_data={},
    install_requires=['numpy'],
    cmdclass = cmdclass,
    ext_modules = cythonize(ext_modules),  # accepts a glob pattern
    include_dirs=[numpy.get_include()]
)
