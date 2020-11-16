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
                                             'custom_func']
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
        raise ValueError("The keys of filtering_criteria should be in: "
                         + str(valid_criteria))


def check_deduplication_criterion(criterion,
                                  valid_criteria=['f1', 'mcc', 'custom_func']
                                  ):
    if not isinstance(criterion, str) or criterion not in valid_criteria:
        raise ValueError("deduplication_criterion should be a string in: "
                         + str(valid_criteria))


def check_custom_func(custom_func):
    if custom_func is not None:
        if not isinstance(custom_func,
                          (types.FunctionType, types.BuiltinFunctionType)
                          ):
            raise TypeError("custom_func should be a function")
        elif custom_func.__code__.co_argcount != 4:
            raise ValueError("custom_func must have 4 arguments: tp, fp, fn, tp")

        try:
            if not isinstance(custom_func(0, 0, 0, 0), (float, int)):
                raise TypeError("custom_func should return a float or an int")
        except TypeError:
            raise TypeError("custom_func should take int as parameters")


def check_consistency(custom_func, deduplication_criterion, filtering_criteria):
    if 'custom_func' in filtering_criteria or 'custom_func' in deduplication_criterion:
        if custom_func is None:
            raise TypeError("custom_func is not define !")
    if custom_func is not None:
        if 'custom_func' not in filtering_criteria and \
                'custom_func' not in deduplication_criterion:
            warn("custom_func is define but not used !")


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
