"""
============================
Plotting Template Classifier
============================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
from skltemplate import TemplateClassifier
from matplotlib import pyplot as plt


X = [[0, 0], [1, 1]]
y = [0, 1]
clf = TemplateClassifier()
clf.fit(X, y)

rng = np.random.RandomState(13)
X_test = rng.rand(500, 2)
y_pred = clf.predict(X_test)

X_0 = X_test[y_pred == 0]
X_1 = X_test[y_pred == 1]

ax0 = plt.scatter(X_0[:, 0], X_0[:, 1], c='crimson', s=30)
ax1 = plt.scatter(X_1[:, 0], X_1[:, 1], c='deepskyblue', s=30)


plt.legend([ax0, ax1], ['Class 0', 'Class 1'])
plt.show()
