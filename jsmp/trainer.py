
## Functions for model training ##

import lightgbm as lgb
from jsmp.query_pq import query_train_pq
from jsmp.train_prep import split_data

def train_lgb_classifier(pq_dir, train_config, label="resp_bin",
                         ignore_cols=['date', 'ts_id', 'resp'],
                         verbose=False):
    """
    """
    
    df = query_train_pq(pq_dir, 
                        date_range=[train_config['date_splits'][i] 
                                    for i in [0,2]],
                        return_cols=['resp'])
    
    df_list = split_data(df, valid_start=train_config['date_splits'][1])
    
    return df_list
    ## TBC ##
    
