from setuptools import setup, find_packages


setup(name='skope-rules',
      version='1.1.0',
      description='Machine Learning with Interpretable Rules',
      url='https://github.com/scikit-learn-contrib/skope-rules',
      author='see AUTHORS.rst',
      license='BSD 3 clause',
      packages=find_packages(),
      keywords=['learning with rules',
                'interpretable machine learning'],
      install_requires=['numpy>=1.13.3',
                        'scikit-learn>=0.23',
                        'scipy>=0.19.1',
                        'pandas>=0.18.1'
                        ])
