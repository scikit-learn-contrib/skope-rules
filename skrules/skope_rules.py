import numpy as np
from collections import Counter
from collections.abc import Iterable
import pandas
import numbers
from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import indices_to_mask
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import _tree

from .rule import Rule, replace_feature_name

from .utils import (check_filtering_criteria, check_deduplication_criterion,
                    check_custom_func, check_consistency,
                    check_max_depth_duplication, check_max_samples)
from .utils import (get_confusion_matrix, f1_score, mcc_score,
                    precision, recall)

INTEGER_TYPES = (numbers.Integral, np.integer)
BASE_FEATURE_NAME = "__C__"
FILTERING_CRITERIA_DEFAULT = {'precision': 0.5, 'recall': 0.01}


class SkopeRules(BaseEstimator):
    """An easy-interpretable classifier optimizing simple logical rules.

    Parameters
    ----------

    feature_names : list of str, optional
        The names of each feature to be used for returning rules in string
        format.

    filtering_criteria: dict, optional (default None)
        If None, filtering_criteria will be equal to {'precision': 0.5, 'recall': 0.01}.
        The criteria to be used for filtering the rules.
        In the form {criterion: min_value}.
        The keys can be among ('precision', 'recall', 'f1', 'mcc', 'custom_func').

    duplication_criterion: str, optional (default='f1')
        The criterion to be used for deduplicating the rules.
        Either 'f1', 'mcc' or 'custom_func'.

    custom_func: FunctionType, optional (default=None)
        A personalised function that can be used as either/both a filtering
        or/and deduplication criterion.
        Has to take tuple of size 4 which is supposed to be the confusion matrix
        elements (tn, fp, fn, tp) (in that order).

    n_estimators : int, optional (default=10)
        The number of base estimators (rules) to use for prediction. More are
        built before selection. All are available in the estimators_ attribute.

    max_samples : int or float, optional (default=.8)
        The number of samples to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_samples_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    max_depth : integer or List or None, optional (default=3)
        The maximum depth of the decision trees. If None, then nodes are
        expanded until all leaves are pure or until all leaves contain less
        than min_samples_split samples.
        If an iterable is passed, you will train n_estimators
        for each tree depth. It allows you to create and compare
        rules of different length.

    max_depth_duplication : integer, optional (default=None)
        The maximum depth of the decision tree for rule deduplication,
        if None then no deduplication occurs.

    max_features : int, float, string or None, optional (default="auto")
        The number of features considered (by each decision tree) when looking
        for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node for
        each decision tree.
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    rules_ : dict of tuples (rule, precision, recall, nb).
        The collection of `n_estimators` rules used in the ``predict`` method.
        The rules are generated by fitted sub-estimators (decision trees). Each
        rule satisfies recall_min and precision_min conditions. The selection
        is done according to OOB precisions.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators used to generate candidate
        rules.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    max_samples_ : integer
        The actual number of samples

    n_features_ : integer
        The number of features when ``fit`` is performed.

    classes_ : array, shape (n_classes,)
        The classes labels.
    """

    def __init__(self,
                 feature_names=None,
                 filtering_criteria=None,
                 duplication_criterion='f1',
                 custom_func=None,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=3,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        if filtering_criteria is not None:
            check_filtering_criteria(filtering_criteria)

        self.filtering_criteria = filtering_criteria

        check_deduplication_criterion(duplication_criterion)
        self.duplication_criterion = duplication_criterion
        check_custom_func(custom_func)
        self.custom_func = custom_func
        if self.filtering_criteria:
            check_consistency(self.custom_func,
                              self.duplication_criterion,
                              self.filtering_criteria
                              )
        else:
            check_consistency(self.custom_func,
                              self.duplication_criterion,
                              FILTERING_CRITERIA_DEFAULT
                              )

        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_samples_features = max_samples_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_depth = max_depth
        check_max_depth_duplication(max_depth_duplication)
        self.max_depth_duplication = max_depth_duplication
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weights=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention 0 for
            normal data, 1 for anomalies.

        sample_weights : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples, typically
            the amount in case of transactions data. Used to grow regression
            trees producing further rules to be tested.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_ = X.shape[1]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("This method needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        if not set(self.classes_) == set([0, 1]):
            warn("Found labels %s. This method assumes target class to be"
                 " labeled as 1 and normal data to be labeled as 0. Any label"
                 " different from 0 will be considered as being from the"
                 " target class."
                 % set(self.classes_))
            y = (y > 0)

        # ensure that max_samples is in [1, n_samples]:
        n_samples = X.shape[0]
        self.max_samples_ = check_max_samples(self.max_samples, n_samples)

        self.rules_ = {}
        self.estimators_ = []
        self.estimators_samples_ = []
        self.estimators_features_ = []

        # default columns names :
        feature_names_ = [BASE_FEATURE_NAME + x for x in
                          np.arange(X.shape[1]).astype(str)]
        if self.feature_names is not None:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(self.feature_names)}
        else:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(feature_names_)}
        self.feature_names_ = feature_names_

        clfs = []
        regs = []

        self._max_depths = self.max_depth \
            if isinstance(self.max_depth, Iterable) else [self.max_depth]

        for max_depth in self._max_depths:
            bagging_clf = BaggingClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split),
                n_estimators=self.n_estimators,
                max_samples=self.max_samples_,
                max_features=self.max_samples_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                # oob_score=... XXX may be added
                # if selection on tree perf needed.
                # warm_start=... XXX may be added to increase computation perf.
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose)

            bagging_reg = BaggingRegressor(
                base_estimator=DecisionTreeRegressor(
                    max_depth=max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split),
                n_estimators=self.n_estimators,
                max_samples=self.max_samples_,
                max_features=self.max_samples_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                # oob_score=... XXX may be added
                # if selection on tree perf needed.
                # warm_start=... XXX may be added to increase computation perf.
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose)

            clfs.append(bagging_clf)
            regs.append(bagging_reg)

        # define regression target:
        if sample_weights is not None:
            if sample_weights is not None:
                sample_weights = check_array(sample_weights, ensure_2d=False)
            weights = sample_weights - sample_weights.min()
            contamination = float(sum(y)) / len(y)
            y_reg = (
                pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                pow((weights).mean(), 0.5) * (y == 0))
            y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid
        else:
            y_reg = y  # same as an other classification bagging

        for clf in clfs:
            clf.fit(X, y)
            self.estimators_ += clf.estimators_
            self.estimators_samples_ += clf.estimators_samples_
            self.estimators_features_ += clf.estimators_features_

        for reg in regs:
            reg.fit(X, y_reg)
            self.estimators_ += reg.estimators_
            self.estimators_samples_ += reg.estimators_samples_
            self.estimators_features_ += reg.estimators_features_

        rules_ = []
        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):

            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if sum(mask) == 0:
                warn("OOB evaluation not possible: doing it in-bag."
                     " Performance evaluation is likely to be wrong"
                     " (overfitting) and selected rules are likely to"
                     " not perform well! Please use max_samples < 1.")
                mask = samples
            rules_from_tree = self._tree_to_rules(
                estimator, np.array(self.feature_names_)[features])

            # XXX todo: idem without dataframe
            X_oob = pandas.DataFrame((X[mask, :])[:, features],
                                     columns=np.array(
                                         self.feature_names_)[features])

            if X_oob.shape[1] > 1:  # otherwise pandas bug (cf. issue #16363)
                y_oob = y[mask]
                y_oob = np.array((y_oob != 0))

                # Add OOB performances to rules:
                # rule <-> (rule_string, confusion matrix)
                rules_from_tree = [(r, get_confusion_matrix(r, X_oob, y_oob))
                                   for r in set(rules_from_tree)]
                rules_ += rules_from_tree

        # Factorize rules before semantic tree filtering
        rules_ = [
            tuple(rule)
            for rule in
            [Rule(r, args=confusion_matrix)
             for r, confusion_matrix in rules_]
            ]

        # Filter the rules according to filtering_criteria
        for rule in rules_:
            rule_str, confusion_matrix = rule
            # retrieve all scores info about the rule
            info_rule = {'precision': precision(confusion_matrix),
                         'recall': recall(confusion_matrix),
                         'f1': f1_score(confusion_matrix),
                         'mcc': mcc_score(confusion_matrix)
                         }
            if self.custom_func is not None:
                info_rule['custom_func'] = self.custom_func(confusion_matrix)
            if self.filtering_criteria is None:
                _filtering_criteria = FILTERING_CRITERIA_DEFAULT
            else:
                _filtering_criteria = self.filtering_criteria

            if all(x < y for x, y in
                    zip(_filtering_criteria.values(),
                        [info_rule[criterion]
                         for criterion in _filtering_criteria
                         ]
                        )
                   ):
                if rule_str in self.rules_:
                    # update the confusion matrix to the new mean
                    def update(x, y, z):
                        return round(x + 1. / y * (z - x))
                    nb = self.rules_[rule_str][1] + 1
                    new_confusion_matrix = [update(self.rules_[rule_str][0][i],
                                                   nb,
                                                   confusion_matrix[i]
                                                   )
                                            for i in range(4)]
                    self.rules_[rule_str] = (tuple(new_confusion_matrix), nb)
                else:
                    self.rules_[rule_str] = (confusion_matrix, 1)

        # Replace (confusion_matrix, nb) tuple by confusion_matrix
        for key in self.rules_:
            self.rules_[key] = self.rules_[key][0]

        # Transform dic object into list
        self.rules_ = list(self.rules_.items())

        # Deduplicate the rule using semantic tree
        if self.max_depth_duplication is not None:
            self.rules_ = self._deduplicate(self.rules_)

        # Sort the rules according to duplication_criterion criterion
        func = self.custom_func
        if self.custom_func is None:
            func = globals()[self.duplication_criterion + '_score']

        self.rules_ = sorted(self.rules_, key=lambda x: func(x[1]), reverse=True)

        self.rules_without_feature_names_ = self.rules_

        # Replace generic feature names by real feature names
        self.rules_ = [(replace_feature_name(rule, self.feature_dict_),
                        args)
                       for rule, args in self.rules_
                       ]

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        Returns
        -------
        is_outlier : array, shape (n_samples,)
            For each observations, tells whether or not (1 or 0) it should
            be considered as an outlier according to the selected rules.
        """

        return np.array((self.decision_function(X) > 0), dtype=int)

    def decision_function(self, X):
        """Average anomaly score of X of the base classifiers (rules).

        The anomaly score of an input sample is computed as
        the weighted sum of the binary rules outputs, the weight being
        the respective precision of each rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The anomaly score of the input samples.
            The higher, the more abnormal. Positive scores represent outliers,
            null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (r, w) in selected_rules:
            scores[list(df.query(r).index)] += precision(w)

        return scores

    def rules_vote(self, X):
        """Score representing a vote of the base classifiers (rules).

        The score of an input sample is computed as the sum of the binary
        rules outputs: a score of k means than k rules have voted positively.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The score of the input samples.
            The higher, the more abnormal. Positive scores represent outliers,
            null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_

        scores = np.zeros(X.shape[0])
        for (r, _) in selected_rules:
            scores[list(df.query(r).index)] += 1

        return scores

    def score_top_rules(self, X):
        """Score representing an ordering between the base classifiers (rules).

        The score is high when the instance is detected by a performing rule.
        If there are n rules, ordered by increasing OOB precision, a score of k
        means than the kth rule has voted positively, but not the (k-1) first
        rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The score of the input samples.
            Positive scores represent outliers, null scores represent inliers.

        """
        # Check if fit had been called
        check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
                               'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pandas.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (k, r) in enumerate(list((selected_rules))):
            scores[list(df.query(r[0]).index)] = np.maximum(
                len(selected_rules) - k,
                scores[list(df.query(r[0]).index)])

        return scores

    def predict_top_rules(self, X, n_rules):
        """Predict if a particular sample is an outlier or not,
        using the n_rules most performing rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``

        n_rules : int
            The number of rules used for the prediction. If one of the
            n_rules most performing rules is activated, the prediction
            is equal to 1.

        Returns
        -------
        is_outlier : array, shape (n_samples,)
            For each observations, tells whether or not (1 or 0) it should
            be considered as an outlier according to the selected rules.
        """

        return np.array((self.score_top_rules(X) > len(self.rules_) - n_rules),
                        dtype=int)

    def _tree_to_rules(self, tree, feature_names):
        """
        Return a list of rules from a tree

        Parameters
        ----------
            tree : Decision Tree Classifier/Regressor
            feature_names: list of variable names

        Returns
        -------
        rules : list of rules.
        """
        # XXX todo: check the case where tree is build on subset of features,
        # ie max_features != None

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules = []

        def recurse(node, base_name):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                symbol = '<='
                symbol2 = '>'
                threshold = tree_.threshold[node]
                text = base_name + ["{} {} {}".format(name, symbol, threshold)]
                recurse(tree_.children_left[node], text)

                text = base_name + ["{} {} {}".format(name, symbol2,
                                                      threshold)]
                recurse(tree_.children_right[node], text)
            else:
                rule = str.join(' and ', base_name)
                rule = (rule if rule != ''
                        else ' == '.join([feature_names[0]] * 2))
                # a rule selecting all is set to "c0==c0"
                rules.append(rule)

        recurse(0, [])

        return rules if len(rules) > 0 else 'True'

    def _deduplicate(self, rules):
        """Select the best rules according to the duplication_criterion
        from the clusters of rules built by find_similar_rulesets

        Arguments:
            rules: list of rules

        Returns:
            list of rules
        """
        if self.custom_func is None:
            func = globals()[self.duplication_criterion + '_score']
        else:
            func = self.custom_func

        return [max(rules_set, key=lambda x: func(x[1]))
                for rules_set in self._find_similar_rulesets(rules)
                ]

    def _find_similar_rulesets(self, rules):
        """Create clusters of rules using a decision tree based
        on the terms of the rules

        Parameters
        ----------
        rules : List, List of rules
                The rules that should be splitted in subsets of similar rules

        Returns
        -------
        rules : List of list of rules
                The different set of rules. Each set should be homogeneous

        """
        def split_with_best_feature(rules, depth, exceptions=[]):
            """
            Method to find a split of rules given most represented feature
            """
            if depth == 0:
                return rules

            rulelist = [rule.split(' and ') for rule, score in rules]
            terms = [t.split(' ')[0] for term in rulelist for t in term]
            counter = Counter(terms)
            # Drop exception list
            for exception in exceptions:
                del counter[exception]

            if len(counter) == 0:
                return rules

            most_represented_term = counter.most_common()[0][0]
            # Proceed to split
            rules_splitted = [[], [], []]
            for rule in rules:
                if (most_represented_term + ' <=') in rule[0]:
                    rules_splitted[0].append(rule)
                elif (most_represented_term + ' >') in rule[0]:
                    rules_splitted[1].append(rule)
                else:
                    rules_splitted[2].append(rule)
            new_exceptions = exceptions+[most_represented_term]
            # Choose best term
            return [split_with_best_feature(ruleset,
                                            depth-1,
                                            exceptions=new_exceptions)
                    for ruleset in rules_splitted]

        def breadth_first_search(rules, leaves=None):
            if len(rules) == 0 or not isinstance(rules[0], list):
                if len(rules) > 0:
                    return leaves.append(rules)
            else:
                for rules_child in rules:
                    breadth_first_search(rules_child, leaves=leaves)
            return leaves
        leaves = []
        res = split_with_best_feature(rules, self.max_depth_duplication)
        breadth_first_search(res, leaves=leaves)
        return leaves
