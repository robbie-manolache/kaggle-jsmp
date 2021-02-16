
## Functions for to train action optimization model ##

import os
import json
import numpy as np
from jsmp.query_pq import query_train_pq, query_preds_pq

def train_NN_action_model(train_config,
                          train_dir,                         
                          model_dir=None,
                          preds_dir=None):
    """
    """

    # Metadata check
    if model_dir is None:
        pass
    else:
        try:
            with open(os.path.join(model_dir, "meta.json")) as f:
                meta = json.load(f)
        except:
            meta = {}
        
        # check for duplicate runs
        for k, v in meta.items():
            if v['train_config'] == train_config:
                print("This train_config already used by %s"%k)
                return
            else:
                pass   
    
    # Load data
    df = query_train_pq(train_dir, date_range=train_config["date_range"], 
                        return_cols=['resp', 'weight'])
    df = df[df['weight'] > 0]
    
    # Compute target
    df.loc[:, "target"] = df['resp'] * (df['weight']  ** 
                                        train_config["weight_pwr"])
    X_col = [f for f in df.columns if f.startswith("feature")]
    
    # Return class prediction
    if preds_dir is None:
        pass
    else:
        rc = query_preds_pq(preds_dir)
        df = df.merge(rc, on=['ts_id', 'date'])
        rc = None
        X_col += [f for f in df.columns if f.startswith("resp_bin_")]
        
    return df, X_col
    
    