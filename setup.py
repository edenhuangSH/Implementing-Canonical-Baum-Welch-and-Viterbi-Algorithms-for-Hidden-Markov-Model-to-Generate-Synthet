
"""Hidden Markov Models implemented in linear memory/running time"""

from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("linearhmm.score", [ "linearhmm/score.pyx" ], include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("linearhmm.score", [ "linearhmm/score.c" ],include_dirs=[numpy.get_include()]),
    ]



import linearhmm

VERSION = linearhmm.__version__
# MAINTAINER = "Sergei Lebedev"
# MAINTAINER_EMAIL = "superbobry@gmail.com"

install_requires = ["numpy"]
tests_require = install_requires + ["pytest"]


setup_options = dict(
    name="linearhmm",
    version=VERSION,
    # maintainer=MAINTAINER,
    # maintainer_email=MAINTAINER_EMAIL,
    url="https://github.com/hmmlearn/hmmlearn",
    packages=["linearhmm"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
)


if __name__ == "__main__":
    setup(**setup_options)
