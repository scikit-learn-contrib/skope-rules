#project-template - A template for scikit-learn extensions

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/project-template)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/project-template/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/project-template)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master)

**project-template** is a template project for 
[scikit-learn](http://scikit-learn.org/) 
compatible extensions.

It aids development of estimators that can be used in scikit-learn pipelines
and (hyper)parameter search, while facilitating testing (including some API
compliance), documentation, open source development, packaging, and continuous
integration.

## Important Links
HTML Documentation - http://contrib.scikit-learn.org/project-template/

## Installation and Usage
The package by itself comes with a single module and an estimator. Before
installing the module you will need `numpy` and `scipy`.
To install the module execute:
```shell
$ python setup.py install
```
or 
```
pip install sklearn-template
```

If the installation is successful, and `scikit-learn` is correctly installed,
you should be able to execute the following in Python:
```python
>>> from skltemplate import TemplateEstimator
>>> estimator = TemplateEstimator()
>>> estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))
```

`TemplateEstimator` by itself does nothing useful, but it serves as an example
of how other Estimators should be written. It also comes with its own unit
tests under `template/tests` which can be run using `nosetests`.

## Creating your own library

### 1. Cloning
Clone the project into your computer by executing
```shell
$ git clone https://github.com/scikit-learn-contrib/project-template.git
```
You should rename the `project-template` folder to the name of your project.
To host the project on Github, visit https://github.com/new and create a new
repository. To upload your project on Github execute
```shell
$ git remote set-url origin https://github.com/username/project-name.git
$ git push origin master
```

### 2. Modifying the Source
You are free to modify the source as you want, but at the very least, all your
estimators should pass the [`check_estimator`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator)
test to be scikit-learn compatible.
(If there are valid reasons your estimator cannot pass `check_estimator`, please
[raise an issue](https://github.com/scikit-learn/scikit-learn/issues/new) at
scikit-learn so we can make `check_estimator` more flexible.)

This template is particularly useful for publishing open-source versions of
algorithms that do not meet the criteria for inclusion in the core scikit-learn
package (see [FAQ](http://scikit-learn.org/stable/faq.html)), such as recent
and unpopular developments in machine learning.
However, developing using this template may also be a stepping stone to
eventual inclusion in the core package.

In any case, developers should endeavor to adhere to scikit-learn's
[Contributor's Guide](http://scikit-learn.org/stable/developers/) which promotes
the use of:
* algorithm-specific unit tests, in addition to `check_estimator`'s common tests
* [PEP8](https://www.python.org/dev/peps/pep-0008/)-compliant code
* a clearly documented API using [NumpyDoc](https://github.com/numpy/numpydoc)
  and [PEP257](https://www.python.org/dev/peps/pep-0257/)-compliant docstrings
* references to relevant scientific literature in standard citation formats
* [doctests](https://docs.python.org/3/library/doctest.html) to provide
  succinct usage examples
* standalone examples to illustrate the usage, model visualisation, and
  benefits/benchmarks of particular algorithms
* efficient code when the need for optimization is supported by benchmarks

### 3. Modifying the Documentation

The documentation is built using [sphinx](http://www.sphinx-doc.org/en/stable/).
It incorporates narrative documentation from the `doc/` directory, standalone
examples from the `examples/` directory, and API reference compiled from
estimator docstrings.

To build the documentation locally, ensure that you have `sphinx`,
`sphinx-gallery` and `matplotlib` by executing:
```shell
$ pip install sphinx matplotlib sphinx-gallery
```
The documentation contains a home page (`doc/index.rst`), an API
documentation page (`doc/api.rst`) and a page documenting the `template` module 
(`doc/template.rst`). Sphinx allows you to automatically document your modules
and classes by using the `autodoc` directive (see `template.rst`). To change the
asthetics of the docs and other paramteres, edit the `doc/conf.py` file. For
more information visit the [Sphinx Documentation](http://www.sphinx-doc.org/en/stable/contents.html).

You can also add code examples in the `examples` folder. All files inside
the folder of the form `plot_*.py` will be executed and their generated
plots will be available for viewing in the `/auto_examples` URL.

To build the documentation locally execute
```shell
$ cd doc
$ make html
```

### 4. Setting up Travis CI
[TravisCI](https://travis-ci.org/) allows you to continuously build and test
your code from Github to ensure that no code-breaking changes are pushed. After
you sign up and authourize TravisCI, add your new repository to TravisCI so that
it can start building it. The `travis.yml` contains the configuration required
for Travis to build the project. You will have to update the variable `MODULE`
with the name of your module for Travis to test it. Once you add the project on
TravisCI, all subsequent pushes on the master branch will trigger a Travis
build. By default, the project is tested on Python 2.7 and Python 3.5.

### 5. Setting up Coveralls
[Coveralls](https://coveralls.io/) reports code coverage statistics of your
tests on each push. Sign up on Coveralls and add your repository so that
Coveralls can start monitoring it. The project already contains the required
configuration for Coveralls to work. All subsequent builds after adding your
project will generate a coverage report.

### 6. Setting up Appveyor
[Appveyor](https://www.appveyor.com/) provides continuous intergration on the
windows  platform. Currently, Appveyor can also be used to build platform
specific Windows wheels, which can be uploaded to a Cloud Service provider and
be made available via a Content Delivery Network (CDN). To setup Appveyor to
build your project you need to sign up on Appveyor and authorize it. Appveyor
configaration is governed by the `appveyor.yml` file. You have to change the
following variables in it to match the requirements of your project.

| Variable | Value|
|----------|------|
| `PROJECT_NAME`  | The name of your project. This should be the same as the `name` field in `setup.py`  |
| `MODULE` | The name of the module you want to be tested |
| `CLOUD_STORAGE` | A constant which indicates which Cloud Storage service provider to use. It should be one among the [Supported Providers](https://libcloud.readthedocs.io/en/latest/storage/supported_providers.html) |
| `CLOUD_CONTAINER` | The name of a container with your Cloud Storage service provider where the built files will be uploaded.|
| `WHEELHOUSE_UPLOADER_USERNAME` | The username you have used to register with your Cloud Storage procider |
| `WHEELHOUSE_UPLOADER_SECRET` | An API key you have obtained from your Cloud Storage provider, which will authenticate you to upload files to it. This should **never** be stored in plain text. To make Appveyor encrypt your API key, use Appveyor's [Encrypt Tool](https://ci.appveyor.com/tools/encrypt) and store the returned value using a `secure:` prefix. |

Maintainers of an official [scikit-learn contrib](
https://contrib.scikit-learn.org) repository can request [Rackspace]
(https://mycloud.rackspace.com/) credentials from the scikit-learn developers.


### 7. Setting up Circle CI
The project uses [CircleCI](https://circleci.com/) to build its documentation
from the `master` branch and host it using [Github Pages](https://pages.github.com/).
Again,  you will need to Sign Up and authorize CircleCI. The configuration
of CircleCI is governed by the `circle.yml` file, which needs to be mofified
if you want to setup the docs on your own website. The values to be changed
are

| Variable | Value|
|----------|------|
| `USERNAME`  | The name of the user or organization of the repository where the project and documentation is hosted  |
| `DOC_REPO` | The repository where the documentation will be hosted. This can be the same as the project repository |
| `DOC_URL` | The relative URL where the documentation will be hosted |
| `EMAIL` | The email id to use while pushing the documentation, this can be any valid email address |

In addition to this, you will need to grant access to the CircleCI computers
to push to your documentation repository. To do this, visit the Project Settings
page of your project in CircleCI. Select `Checkout SSH keys` option and then
choose `Create and add user key` option. This should grant CircleCI privileges
to push to the repository `https://github.com/USERNAME/DOC_REPO/`.

If all goes well, you should be able to visit the documentation of your project
on 
```
https://github.com/USERNAME/DOC_REPO/DOC_URL
```

### 8. Adding Badges

Follow the instructions to add a [Travis Badge](https://docs.travis-ci.com/user/status-images/), 
[Coveralls Badge](https://coveralls.io) and 
[CircleCI Badge](https://circleci.com/docs/status-badges) to your repository's
`README`.

### 9. Advertising your package

Once your work is mature enough for the general public to use it, you should
submit a Pull Request to modify scikit-learn's
[related projects listing](https://github.com/scikit-learn/scikit-learn/edit/master/doc/related_projects.rst).
Please insert brief description of your project and a link to its code
repository or PyPI page.
You may also wish to announce your work on the
[`scikit-learn-general` mailing list](https://lists.sourceforge.net/lists/listinfo/scikit-learn-general).

### 10. Uploading your package to PyPI

Uploading your package to [PyPI](https://pypi.python.org/pypi) allows users to
install your package through `pip`. Python provides two repositories to upload
your packages. The [PyPI Test](https://testpypi.python.org/pypi) repository,
which is to be used for testing packages before their release, and the
[PyPI](https://pypi.python.org/pypi) repository, where you can make your
releases. You need to register a username and password with both these sites.
The username and passwords for both these sites need not be the same. To upload
your package through the command line, you need to store your username and
password in a file called `.pypirc` in your `$HOME` directory with the
following format.

```shell
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
repository=https://pypi.python.org/pypi
username=<your-pypi-username>
password=<your-pypi-passowrd>

[pypitest]
repository=https://testpypi.python.org/pypi
username=<your-pypitest-username>
password=<your-pypitest-passowrd>
```
Make sure that all details in `setup.py` are up to date. To upload your package
to the Test server, execute:
```
python setup.py register -r pypitest
python setup.py sdist upload -r pypitest
```
Your package should now be visible on: https://testpypi.python.org/pypi

To install a package from the test server, execute:
```
pip install -i https://testpypi.python.org/pypi <package-name>
```

Similary, to upload your package to the PyPI server execute
```
python setup.py register -r pypi
python setup.py sdist upload -r pypi
```
To install your package, execute:
```
pip install <package-name>
```

*Thank you for cleanly contributing to the scikit-learn ecosystem!*
