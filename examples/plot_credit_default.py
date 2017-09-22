"""
==============================================================
Example: detecting defaults on retail credits with skope_rules
==============================================================


SkopeRules finds logical rules with high precision and fuse them. Finding
good rules is done by fitting classification or regression trees
to sub-samples.
A fitted tree defines a set of rules (each tree node defines a rule); rules
are then tested out of the bag, and the ones with higher precision are kept.
This set of rules is  decision function, reflecting for
each new samples how many rules have find it abnormal.

This example aims at finding logical rules to predict credit defaults. The
analysis shows that setting.

The dataset comes from BLABLABLA.
"""

###############################################################################
# Data import and preparation
# ..................
#
# There are 3 categorical variables (SEX, EDUCATION and MARRIAGE) and 20
# numerical variables.
# The target (credit defaults) is transformed in a binary variable with
# integers 0 (no default) and 1 (default).
# From the 30000 credits, 50% are used for training and 50% are used
# for testing. The target is unbalanced with a 22%/78% ratio.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from skrules import SkopeRules
from skrules.datasets import load_credit_data

print(__doc__)

rng = np.random.RandomState(42)

# Importing data
dataset = load_credit_data()
X = dataset.data
y = dataset.target
# Shuffling data, preparing target and variables
data, y = shuffle(np.array(X), y)
data = pd.DataFrame(data, columns=X.columns)

for col in ['ID']:
    del data[col]

# data = pd.get_dummies(data, columns = ['SEX', 'EDUCATION', 'MARRIAGE'])

# Quick feature engineering
data = data.rename(columns={"PAY_0": "PAY_1"})
old_PAY = ['PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
data['PAY_old_mean'] = data[old_PAY].apply(lambda x: np.mean(x), axis=1)

old_BILL_AMT = ['BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
data['BILL_AMT_old_mean'] = data[old_BILL_AMT].apply(
    lambda x: np.mean(x), axis=1)
data['BILL_AMT_old_std'] = data[old_BILL_AMT].apply(
    lambda x: np.std(x),
    axis=1)

old_PAY_AMT = ['PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
data['PAY_AMT_old_mean'] = data[old_PAY_AMT].apply(
    lambda x: np.mean(x), axis=1)
data['PAY_AMT_old_std'] = data[old_PAY_AMT].apply(
    lambda x: np.std(x), axis=1)

data = data.drop(old_PAY_AMT, axis=1)
data = data.drop(old_BILL_AMT, axis=1)
data = data.drop(old_PAY, axis=1)

# Creating the train/test split
feature_names = list(data.columns)
print(feature_names)
data = data.values
n_samples = data.shape[0]
n_samples_train = int(n_samples / 2)
y_train = y[:n_samples_train]
y_test = y[n_samples_train:]
X_train = data[:n_samples_train]
X_test = data[n_samples_train:]

###############################################################################
# Benchmark with a Decision Tree and Random Forests
# ..................
#
# This part shows the training and performance evaluation of
# two tree-based models.
# The objective remains to extract rules which targets credit defaults.
# This benchmark shows the performance reached with a decision tree and a
# random forest.

DT = GridSearchCV(DecisionTreeClassifier(),
                  param_grid={
                  'max_depth': range(3, 10, 1),
                  'min_samples_split': range(10, 1000, 200),
                  'criterion': ["gini", "entropy"]},
                  scoring={'AUC': 'roc_auc'}, cv=5, refit='AUC',
                  n_jobs=-1)

DT.fit(X_train, y_train)
scoring_DT = DT.predict_proba(X_test)[:, 1]

RF = GridSearchCV(
    RandomForestClassifier(
        n_estimators=30,
        class_weight='balanced'),
    param_grid={
        'max_depth': range(2, 7, 1),
        'max_features': np.linspace(0.1, 0.5, 5)
        },
    scoring={'AUC': 'roc_auc'}, cv=5,
    refit='AUC', n_jobs=-1)

RF.fit(X_train, y_train)
scoring_RF = RF.predict_proba(X_test)[:, 1]

print("Decision Tree selected parameters : "+str(DT.best_params_))
print("Random Forest selected parameters : "+str(RF.best_params_))

# Plot ROC and PR curves

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         sharex=True, sharey=True)

curves = [roc_curve, precision_recall_curve]
xlabels = ['False Positive Rate', 'Recall (True Positive Rate)']
ylabels = ['True Positive Rate (Recall)', 'Precision']

for ax, curve, xlabel, ylabel in zip(axes.flatten(),
                                     curves, xlabels, ylabels):
    if curve == precision_recall_curve:
        y_dt, x_dt, _ = curve(y_test, scoring_DT)
        y_rf, x_rf, _ = curve(y_test, scoring_RF)
        ax.scatter(x_dt, y_dt, c='b', s=10)
        ax.step(x_rf, y_rf, linestyle='-.', c='g', lw=1, where='post')
        ax.set_title("Precision-Recall Curves", fontsize=20)
    else:
        x_dt, y_dt, _ = curve(y_test, scoring_DT)
        x_rf, y_rf, _ = curve(y_test, scoring_RF)
        label = ('Decision Tree, AUC: %0.3f' % auc(x_dt, y_dt))
        ax.scatter(x_dt, y_dt, c='b', s=10, label=label)
        label = ('Random Forest, AUC: %0.3f' % auc(x_rf, y_rf))
        ax.plot(x_rf, y_rf, '-.', lw=1, label=label, c='g')
        ax.set_title("ROC Curves", fontsize=20)
        ax.legend(loc='upper center', fontsize=8)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

plt.show()

###############################################################################
# The ROC and Precision-Recall curves show that both models
# approximatively have the same performances for low True Positive Rates
# (also called Recall).
# This "low-recall" domain is visible around the bottom right hand side
# of the ROC curve and on the left side of the Precision-Recall curve.
# A good performance on this part of the curve means that the model can
# precisely detect a fraction of credit defaults (the easiest one).
# For highest recalls, Random Forests shows a better performance
# in this domain.

###############################################################################
# Getting rules with skrules
# ..................
#
# This part shows how SkopeRules can be fitted to detect credit defaults.
# Performances are compared with the random forest model previously trained.

# fit the model
rng = np.random.RandomState(42)

clf = SkopeRules(
    similarity_thres=1.0, max_depth=3, max_features=0.5,
    max_samples_features=0.5, random_state=rng, n_estimators=30,
    feature_names=feature_names, recall_min=0.02, precision_min=0.6
    )
clf.fit(X_train, y_train)
scoring = clf.decision_function(X_test)

print(str(len(clf.rules_)) + ' rules have been built.')
print('The most precise rules are the following:')
print(clf.rules_[:5])

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         sharex=True, sharey=True)

curves = [roc_curve, precision_recall_curve]
xlabels = ['False Positive Rate', 'Recall (True Positive Rate)']
ylabels = ['True Positive Rate (Recall)', 'Precision']

for ax, curve, xlabel, ylabel in zip(axes.flatten(), curves,
                                     xlabels, ylabels):
    if curve == precision_recall_curve:
        y_sr, x_sr, _ = curve(y_test, scoring)
        y_rf, x_rf, _ = curve(y_test, scoring_RF)
        ax.scatter(x_sr, y_sr, c='b', s=10, label=label)
        ax.step(x_rf, y_rf, linestyle='-.', c='g', lw=1, where='post')
        ax.set_title("Precision-Recall Curves", fontsize=20)
    else:
        x_sr, y_sr, _ = curve(y_test, scoring)
        x_rf, y_rf, _ = curve(y_test, scoring_RF)
        label = ('SkopeRules, AUC: %0.3f' % auc(x_sr, y_sr))
        ax.scatter(x_sr, y_sr, c='b', s=10, label=label)
        label = ('Random Forest, AUC: %0.3f' % auc(x_rf, y_rf))
        ax.plot(x_rf, y_rf, '-.', lw=1, label=label, c='g')
        ax.set_title("ROC Curves", fontsize=20)
        ax.legend(loc='upper center', fontsize=8)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

plt.show()

###############################################################################
# Refining rules with the similarity threshold parameter
# ..................
#
# This part shows how SkopeRules can be set up to discard unecessary rules.
# This rule selection consists in

# fit the model
rng = np.random.RandomState(42)

clf = SkopeRules(
    similarity_thres=1.0, max_depth=3, max_features=0.5,
    max_samples_features=0.5, random_state=rng, n_estimators=30,
    feature_names=feature_names, recall_min=0.05, precision_min=0.6
    )

clf.fit(X_train, y_train)
scoring = clf.decision_function(X_test)

rng = np.random.RandomState(42)

clf = SkopeRules(
    similarity_thres=0.9, max_depth=3, max_features=0.5,
    max_samples_features=0.5, random_state=rng, n_estimators=30,
    feature_names=feature_names, recall_min=0.05, precision_min=0.6
    )
clf.fit(X_train, y_train)
scoring_RF = clf.decision_function(X_test)

rng = np.random.RandomState(42)

clf = SkopeRules(
    similarity_thres=0.5, max_depth=3, max_features=0.5,
    max_samples_features=0.5, random_state=rng, n_estimators=30,
    feature_names=feature_names, recall_min=0.05, precision_min=0.6
    )
clf.fit(X_train, y_train)
scoring_ET = clf.decision_function(X_test)

# Plot models
fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         sharex=True, sharey=True)

curves = [roc_curve, precision_recall_curve]
xlabels = ['False Positive Rate', 'Recall (True Positive Rate)']
ylabels = ['True Positive Rate (Recall)', 'Precision']

for ax, curve, xlabel, ylabel in zip(axes.flatten(), curves,
                                     xlabels, ylabels):
    if curve == precision_recall_curve:
        y_sr1, x_sr1, _ = curve(y_test, scoring)
        y_sr2, x_sr2, _ = curve(y_test, scoring_RF)
        y_sr3, x_sr3, _ = curve(y_test, scoring_ET)
        ax.scatter(x_sr1, y_sr1, c='b', s=10, label=label)
        ax.step(x_sr2, y_sr2, linestyle='-.', c='g', lw=1, where='post')
        ax.scatter(x_sr3, y_sr3, c='r', s=10, label=label)
        ax.set_title("Precision-Recall Curves", fontsize=20)
    else:
        x_sr1, y_sr1, _ = curve(y_test, scoring)
        x_sr2, y_sr2, _ = curve(y_test, scoring_RF)
        x_sr3, y_sr3, _ = curve(y_test, scoring_ET)
        label = ('SkopeRules, AUC: %0.3f' % auc(x_sr1, y_sr1))
        ax.scatter(x_sr1, y_sr1, c='b', s=10, label=label)
        label = ('Random Forest, AUC: %0.3f' % auc(x_sr2, y_sr2))
        ax.plot(x_sr2, y_sr2, '-.', lw=1, label=label, c='g')
        label = ('ExtraTrees, AUC: %0.3f' % auc(x_sr3, y_sr3))
        ax.scatter(x_sr3, y_sr3, c='r', s=10, label=label)
        ax.set_title("ROC Curves", fontsize=20)
        ax.legend(loc='upper center', fontsize=8)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

plt.show()

###############################################################################
# Applying rules and predicting with skrules
# ..................
#
# This part shows how, once fitted, SkopeRules can be used to
# make predictions.
