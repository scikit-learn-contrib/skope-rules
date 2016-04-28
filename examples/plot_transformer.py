"""
=============================
Plotting Template Transformer
=============================

An example plot of :class:`skltemplate.template.TemplateTransformer`
"""
import numpy as np
from skltemplate import TemplateTransformer
from matplotlib import pyplot as plt

X = np.arange(50).reshape(-1, 1)
estimator = TemplateTransformer()
X_transformed = estimator.fit_transform(X)

plt.plot(X.flatten()/X.max(), label='Original Data')
plt.plot(X_transformed.flatten()/X_transformed.max(), label='Transformed Data')
plt.title('Scaled plots of original and transformed data')

plt.legend(loc='best')
plt.show()
