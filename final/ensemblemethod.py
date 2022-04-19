import pandas as pd

def ensemblemethod(dt_pred, rf_pred, xgb_pred, cb_pred, adb_pred, hgb_pred):
    '''
    Custom ensemble method which combines the predictions of multiple models
    Utilises the F1 scores received and squares them to punish the lower-scoring models more
    '''
    ensemble_pred = (dt_pred*0.905381**2 + rf_pred*0.933422**2 + xgb_pred*0.920953**2 + cb_pred*0.946621**2 + adb_pred*0.896145**2 + hgb_pred*0.941695**2)/(0.905381**2 + 0.933422**2 + 0.920953**2 + 0.946621**2 + 0.896145**2 + 0.941695**2)
    ensemble_pred['ebb_eligible'] = ensemble_pred['ebb_eligible'].apply(lambda x: round(x))
    result = ensemble_pred
    result.to_csv("../submission/2022-04-17_final.csv")
