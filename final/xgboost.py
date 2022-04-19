import xgboost as xgb
from utils import predict

def xgboost(X_train, y_train):
    '''
    Final tuned model using XGBoostClassifier
    '''
    xgb_clf = xgb.XGBClassifier(seed=0, alpha=0, learning_rate=0.1, colsample_bytree=0.25, max_depth=10, verbosity=0, use_label_encoder=False, eval_metric='logloss')
    xgb_clf = xgb_clf.fit(X_train, y_train)

    xgb_pred = predict(xgb_clf)
    xgb_score = xgb_clf.score(X_train, y_train)

    return xgb_pred, xgb_score
