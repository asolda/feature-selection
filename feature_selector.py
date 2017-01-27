import parse_arff
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

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

print('\n==== Recursive Feature Elimination with Cross-Validation ====')

# running Recursive Feature Elimination (with Cross-Validation)
estimator = SVR(kernel="linear")
# step = number of features to remove at each iteration
# cv = cross-validation generator (default = 3-fold cross-validation)
selector = RFECV(estimator, step=1, cv=3, verbose = 1)
selector = selector.fit(X, y)
print(selector.support_)