from setuptools import setup, find_packages


setup(name='skope-rules',
      version='1.0.0',
      description='Machine Learning with Interpretable Rules',
      url='https://github.com/scikit-learn-contrib/skope-rules',
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
