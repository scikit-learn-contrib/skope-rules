import sys
from setuptools import setup, find_packages


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='skope-rules',
      version='0.0.1',
      description='Machine Learning with Interpretable Rules',
      url='https://github.com/skope-rules/skope-rules',
      author='see AUTHORS.rst',
      license='BSD 3 clause',
      packages=find_packages(),
      keywords=['learning with rules',
                'interpretable machine learning'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'scipy>=0.17.0',
                        'pandas>=0.18.1'
                        ])
