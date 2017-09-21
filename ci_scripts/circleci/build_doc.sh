#!/bin/bash


# Installing required system packages to support the rendering of math
# notation in the HTML documentation
sudo -E apt-get -yq update
sudo -E apt-get -yq remove texlive-binaries --purge
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes \
    install dvipng texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended\
    latexmk

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n $CONDA_ENV_NAME --yes --quiet python numpy scipy \
  cython nose coverage matplotlib sphinx=1.6.2 pillow
source activate testenv
pip install sphinx-gallery numpydoc xlrd

# Build and install skope-rules in dev mode
python setup.py develop
set -o pipefail && cd doc && make html 2>&1 | tee ~/log.txt
