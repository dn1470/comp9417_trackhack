import pandas as pd

def feature_selection(train_df_final, eval_set):
    columns_to_drop=['storage_total', 'has_leased', 'storage_available', 'operating_system', 'bluetooth_on', 'memory_total']
    train_df_final = train_df_final.drop(columns=columns_to_drop)
    eval_set = eval_set.drop(columns=columns_to_drop)

    return train_df_final, eval_set