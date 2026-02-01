from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#step 1: create imbalanced data
X,y= make_classification(n_classes=2,class_sep=2,
                         weights=[0.95,0.05],
                         n_informative=3,n_redundant=1,
                         n_clusters_per_class=1,
                         n_samples=1000, random_state=42)
'''
this line uses make_classifications from scikit-learn,
which generates a synthetic dataset suitable for
classification tasks.its particularly useful for testing
and demonstration machine learning techniques like SMOTES.

parameter                      description
'''
print("original class distribution:",Counter(y))

#optional: visualize original class distribution 
sns.countplot(x=y)
plt.title("original class distribution")
plt.show()

#step 2: apply SMOTE
smote=SMOTE(random_state=42)
X_resampled,y_resampled= smote.fit_resample(X,y)

print("resampled class distribution:",Counter(y_resampled))

#optional :visualize resampled class distribution
sns.countplot(x=y_resampled)
plt.title("after SMOTE: resampled class distribution")
plt.show()



