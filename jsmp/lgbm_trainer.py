
## Functions for model training ##

import os
import json
from datetime import datetime
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
from jsmp.query_pq import query_train_pq
from jsmp.train_prep import gen_return_bins, split_data, convert_to_lgb_dataset
from jsmp.eval_tools import predict_return_bin

def train_lgb_classifier(pq_dir, 
                         train_config, 
                         model_dir=None,
                         model_name=None,
                         pred_dir=None,
                         label="resp_bin",
                         ignore_cols=['date', 'ts_id', 'resp'],
                         check_splits_only=False,
                         verbose=False):
    """
    """
    
    # load metadata and cross-check train config history
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
    
    # load data
    df = query_train_pq(pq_dir, 
                        date_range=[train_config['date_splits'][i] for i in 
                                    [0, len(train_config['date_splits'])-1]],
                        return_cols=['resp'])
    
    # categorize returns checking class representation if required
    df = gen_return_bins(df, splits=train_config['resp_splits'])
    if check_splits_only:
        return df['resp_bin'].value_counts(sort=False)
    else:
        pass
    
    # train-test split and covert to LGB datasets  
    if len(train_config['date_splits']) == 4:
        eval_start = train_config['date_splits'][2]
    else:
        eval_start = None
    df_list = split_data(df, valid_start=train_config['date_splits'][1],
                         eval_start=eval_start)
    df = None
    lgb_sets = convert_to_lgb_dataset(df_list[:2], label=label, 
                                      ignore_cols=ignore_cols)
    
    # set aside eval_df 
    if len(train_config['date_splits']) == 4:
        eval_df = df_list[2]
    else:
        pass
    df_list = None
    
    # train model
    lgb_model = lgb.train(params=train_config['params'], 
                          train_set=lgb_sets[0], 
                          valid_sets=lgb_sets[1], 
                          num_boost_round=train_config['n_rounds']['total'], 
                          early_stopping_rounds=train_config['n_rounds']['early'],
                          verbose_eval=verbose)
    
    # save the model and update metadata
    if model_dir is None:
        pass
    else:
        if model_name is None:
            model_name = "lgbm_%s.txt"%datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            pass
        lgb_model.save_model(os.path.join(model_dir, model_name))
        meta[model_name] = {
            'train_config': train_config,
            'score': np.exp(-lgb_model.best_score["valid_0"]["multi_logloss"]),
            'benchmark': 1/train_config['params']['num_class']
        }
        with open(os.path.join(model_dir, "meta.json"), 'w') as f:
            json.dump(meta, f)
            
    # make evaluation set predictions and save to disk
    if pred_dir is None:
        pass
    else:
        pred_df = predict_return_bin(eval_df, lgb_model,
                                     eval_cols=['date', 'ts_id'])
        table = pa.Table.from_pandas(pred_df)
        pq.write_to_dataset(table, pred_dir, partition_cols=['date'])
    
    return lgb_model
    
def gen_date_splits(init_splits, split_config, final_splits):
    """
    init_splits = [start_date, valid_start, eval_start, eval_end]
    split_config = [{'start': start_date, 
                     'iter': number of iterations,
                     'eval': increment to eval_end each iteration},
                     ...]
    final_splits = [start_date, valid_start, max_date]
    """

    # set inititial date splits
    date_splits = [init_splits]
    max_date = final_splits[-1]

    # increment additional dates to make out-of-sample preds for
    for sc in split_config:
        for _ in range(sc['iter']):
            new_splits = date_splits[-1].copy()
            new_splits[0] = sc['start']
            new_splits[1] = new_splits[2]
            new_splits[2] = new_splits[3] + 1
            new_splits[3] = new_splits[2] + sc['eval'] - 1
            if max(new_splits) <= max_date:
                date_splits.append(new_splits)

    # ensure that out-of-sample pred dates include max date
    if date_splits[-1][-1] < max_date:
        new_splits = date_splits[-1].copy()
        new_splits[1] = new_splits[2]
        new_splits[2] = new_splits[3]
        new_splits[3] = max_date
        date_splits.append(new_splits)

    # add final model run dates and output date_splits
    date_splits.append(final_splits)
    return date_splits

def multi_date_lgbm_preds(pq_dir, model_dir, pred_dir,
                          train_config, date_splits):
    """
    """
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for ds in date_splits:
        
        train_config["date_splits"] = ds   
        
        if len(ds) == 3:
            save_preds_to = None
            save_model_to = model_dir
            model_name = "lgbm_%s.txt"%now 
            print("Now training final model!")
        else:
            save_preds_to = os.path.join(pred_dir, "lgbm_%s"%now)
            save_model_to = None
            model_name=None
            print("Now generating predictions for dates %d to %d"%
                  (ds[2], ds[3]))
            
        _ = train_lgb_classifier(pq_dir=pq_dir,
                                 model_dir=save_model_to,
                                 model_name=model_name,
                                 pred_dir=save_preds_to,
                                 train_config=train_config)

