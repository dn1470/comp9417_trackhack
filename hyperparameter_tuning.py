from sklearn import model_selection
from itertools import product

def param_tuning(model, params, data, target):
    [X_train, X_test, y_train, y_test] = model_selection.train_test_split(data, target, test_size=0.3)
    keys = list(params.keys())
    
    for p in list(product(*params.values())):
        curr_params = dict(zip(keys, p))
        classifier = model(**curr_params)
        classifier = classifier.fit(X_train, y_train)

        trainScore = classifier.score(X_train, y_train)
        testScore = classifier.score(X_test, y_test)

        print(f"params={curr_params} || train_score={trainScore}, test_score={testScore}")