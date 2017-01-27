import parse_arff
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# preparing our data
X_train, X_test, y_train, y_test, feature_names = parse_arff.import_arff('data/jm1.arff')

# create complete sets
X = X_train + X_test
y = y_train + y_test

# running Variance Threshold
# set threshold
sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
sel.fit_transform(X)
mask = sel.get_support()

print('\n==== Variance Threshold ====')

for index, name in enumerate(feature_names) :
    print('Feature:', name, 'Required:', mask[index])

# fit_transform might have modified the original set
X = X_train + X_test

print('\n==== Univariate Feature Selection ====')
print('Top features:')

# running Univariate feature selection
X_new = SelectPercentile(chi2, percentile=50).fit_transform(X, y)

# computing the features that are left
sample_new = X_new[0]
sample = X[0]

for index, value in enumerate(sample):
    for value_new in sample_new:
        if value == value_new:
            print(feature_names[index])

''' COMPUTATIONALLY INTENSIVE!!!
print('\n==== Recursive Feature Elimination with Cross-Validation ====')

# running Recursive Feature Elimination (with Cross-Validation)
estimator = SVR(kernel="linear")
# step = number of features to remove at each iteration
# cv = cross-validation generator (default = 3-fold cross-validation)
selector = RFECV(estimator, step=1, cv=3, verbose = 1)
selector = selector.fit(X, y)
print(selector.support_)
'''

print('\n==== Select From Model Feature Selection ====')
print("L1 based feature selection, LinearSVC:")

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)

sample_new = X_new[0]
sample = X[0]

for index, value in enumerate(sample):
    for value_new in sample_new:
        if value == value_new:
            print(feature_names[index])

print("\nL1 based feature selection, Logistic Regression:")

clf = LogisticRegression(penalty='l1')

sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(X, y)
X_new = sfm.transform(X)

sample_new = X_new[0]
sample = X[0]

for index, value in enumerate(sample):
    for value_new in sample_new:
        if value == value_new:
            print(feature_names[index])

print('\n==== Tree Based Feature Selection ====')

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)

sample_new = X_new[0]
sample = X[0]

for index, value in enumerate(sample):
    for value_new in sample_new:
        if value == value_new:
            print(feature_names[index])