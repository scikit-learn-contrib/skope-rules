# -*- coding: utf-8 -*-
import types
import six
import numbers
import numpy as np
from warnings import warn

INTEGER_TYPES = (numbers.Integral, np.integer)


def check_filtering_criteria(criteria,
                             valid_criteria=['precision',
                                             'recall',
                                             'f1',
                                             'mcc',
                                             'myfunc']
                             ):
    if not isinstance(criteria, dict):
        raise TypeError("filtering_criteria should be a dict")
    elif len(criteria) < 1:
        raise TypeError("filtering_criteria should contain at least one key")
    elif not all(isinstance(key, str) for key in criteria) or \
            not all(isinstance(value, (float, int))
                    for value in criteria.values()
                    ):
        raise TypeError("filtering_criteria should be {str: int or float}")
    elif not all((x in valid_criteria) for x in criteria):
        raise TypeError("The keys of filtering_criteria should be in: "
                         + str(valid_criteria))


def check_deduplication_criterion(criterion,
                                  valid_criteria=['f1', 'mcc', 'myfunc']
                                  ):
    if not isinstance(criterion, str) or criterion not in valid_criteria:
        raise TypeError("deduplication_criterion should be a string in: "
                         + str(valid_criteria))


def check_myfunc(myfunc):
    if myfunc is not None:
        if not isinstance(myfunc,
                          (types.FunctionType, types.BuiltinFunctionType)
                          ):
            raise TypeError("myfunc should be a function")
        elif myfunc.__code__.co_argcount != 4:
            raise ValueError("myfunc must have 4 arguments: tp, fp, fn, tp")

        try:
            if not isinstance(myfunc(0, 0, 0, 0), (float, int)):
                raise TypeError("myfunc should return a float or an int")
        except:
            raise TypeError("myfunc should take int as parameters")


def check_consistency(myfunc, deduplication_criterion, filtering_criteria):
    if 'myfunc' in filtering_criteria or 'myfunc' in deduplication_criterion:
        if myfunc is None:
            raise TypeError("myfunc is not define !")
    if myfunc is not None:
        if 'myfunc' not in filtering_criteria and \
                'myfunc' not in deduplication_criterion:
            warn("myfunc is define but not used !")


def check_max_depth_duplication(max_depth_duplication):
    if not isinstance(max_depth_duplication, int) \
            and max_depth_duplication is not None:
        raise TypeError("max_depth_duplication should be an integer")


def check_max_samples(max_samples, n_samples):
    if isinstance(max_samples, six.string_types):
        raise ValueError('max_samples (%s) is not supported.'
                            'Valid choices are: "auto", int or'
                            'float' % max_samples)

    elif isinstance(max_samples, INTEGER_TYPES):
        if max_samples > n_samples:
            warn("max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (max_samples, n_samples))
            max_samples_ = n_samples
        else:
            max_samples_ = max_samples

    else:  # float
        if not (0. < max_samples <= 1.):
            raise ValueError("max_samples must be in (0, 1], got %r"
                                % max_samples)
        max_samples_ = int(max_samples * n_samples)
    
    return max_samples_
