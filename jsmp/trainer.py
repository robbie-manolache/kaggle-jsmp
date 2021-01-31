
## Functions for model training ##

import lightgbm as lgb
from jsmp.query_pq import query_train_pq
from jsmp.train_prep import gen_return_bins, split_data, convert_to_lgb_dataset

def train_lgb_classifier(pq_dir, 
                         train_config, 
                         model_path = None,
                         label="resp_bin",
                         ignore_cols=['date', 'ts_id', 'resp'],
                         verbose=False):
    """
    """
    
    ## load metadata and cross-check config history ##
        # to avoid repeating the same training runs
    
    df = query_train_pq(pq_dir, 
                        date_range=[train_config['date_splits'][i] 
                                    for i in [0,2]],
                        return_cols=['resp'])
    
    df = gen_return_bins(df, splits=train_config['resp_splits'])
    
    df_list = split_data(df, valid_start=train_config['date_splits'][1])
    
    lgb_sets = convert_to_lgb_dataset(df_list, label=label, 
                                      ignore_cols=ignore_cols)
    
    lgb_model = lgb.train(params=train_config['params'], 
                          train_set=lgb_sets[0], 
                          valid_sets=lgb_sets[1], 
                          num_boost_round=train_config['n_rounds']['total'], 
                          early_stopping_rounds=train_config['n_rounds']['early'],
                          verbose_eval=verbose)
    
    if model_path is None:
        pass
    else:
        lgb_model.save_model(model_path)
    
    ##  record the score and metadata ##
    # use model_path as key?
    
    return lgb_model
    
