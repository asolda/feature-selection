import arff
from sklearn.model_selection import train_test_split

def import_arff(filepath):
    # loading arff file
    data = arff.load(open(filepath, 'r'))

    # brief feature overview
    #for elem in data['attributes']:
        #print('Feature:', elem[0], 'Type:', elem[1])

    # removing the last non-numerical feature
    X = [sample[:-1] for sample in data['data']]

    # storing feature names
    names = [sample[0] for sample in data['attributes'][:-1]]

    # using last feature as a label; Y -> 0, N -> 1
    y = [0 if sample[-1] == 'Y' else 1 for sample in data['data']]

    # splitting dataset, creating training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test, names