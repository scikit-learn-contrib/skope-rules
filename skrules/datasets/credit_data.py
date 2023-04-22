"""default of credit card clients dataset.

The original database is available from UCI Machine Learning Repository:

    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

The data contains 30000 observations on 24 variables.

References
----------

Lichman, M. (2013). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer
Science.

"""


from os.path import exists, join
from urllib.request import urlretrieve
from collections import namedtuple
import hashlib

import pandas as pd
import numpy as np

from sklearn.datasets import get_data_home
from sklearn.utils import Bunch


# Because of sklearn.datasets.base and  module is  deprecated in version 0.22
# and will be removed in version 0.24
# We delete "from sklearn.datasets.base import _fetch_remote, RemoteFileMetadata"
# The function _sha256 of sklearn.datasets.base is redefined

def calculate_sha256(file_path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(file_path, "rb") as file:
        while True:
            buffer = file.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()

def load_credit_data():
    sk_data_dir = get_data_home()
    RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])
    archive = RemoteFileMetadata(
        filename='default of credit card clients.xls',
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00350/default%20of%20credit%20card%20clients.xls',
        checksum=('30c6be3abd8dcfd3e6096c828bad8c2f'
                  '011238620f5369220bd60cfc82700933'))
    file_path = join(sk_data_dir, archive.filename)
    if not exists(file_path):
        urlretrieve(archive.url, file_path)
        checksum = calculate_sha256(file_path)
        if archive.checksum != checksum:
            raise IOError("{} has an SHA256 checksum ({}) "
                          "differing from expected ({}), "
                          "file may be corrupted.".format(file_path,
                                                          checksum,
                                                          archive.checksum))
    data = pd.read_excel(file_path,
                         sheet_name='Data',
                         header=1)
    dataset = Bunch(
        data=(data.drop('default payment next month', axis=1)),
        target=np.array(data['default payment next month'])
        )
    return dataset
