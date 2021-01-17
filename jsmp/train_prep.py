
## Functions to prep data for training ##

import numpy as np
import pandas as pd
import lightgbm as lgb

def gen_return_bins(df, in_col='resp', out_col='resp_bin',
                    splits=[-0.05, -0.005, 0.005, 0.05],
                    labels=['shocker', 'will_survive', 'boring', 
                            'not_bad', 'legendary']):
    """
    """
    
    # copy input df
    df_out = df.copy()
    
    # make sure labels are the right length
    if labels is not None:
        if len(labels) != (len(splits) + 1):
            print("Must specify correct number of labels!")
            return

    # pad bin splits
    splits = [-np.inf] + splits + [np.inf]

    # create categorical bin column
    df_out.loc[:, out_col] = pd.cut(df[in_col], 
                                    bins=splits, 
                                    labels=labels)
    return df_out
    

def split_data(df, 
               valid_start, valid_end="auto", 
               eval_start=None, eval_end="auto", 
               train_start=0, train_end="auto"):
    """
    Split input data into training, validation (and evaluation)
    sets by date.
    """

    # set validation date range
    if valid_end is "auto":
        if eval_start is None:
            valid_end = df['date'].max()
        else:
            valid_end = eval_start - 1
    else:
        pass
    
    # set evaluation date range
    if eval_start is None:
        pass
    else:
        if eval_end is "auto":
            eval_end = df['date'].max()
        else:
            pass
        
    # set training date range
    if train_end is "auto":
        train_end = valid_start - 1
    else:
        pass        
    
    # filter data and return
    train_df = df[(df['date'] >= train_start) & 
                  (df['date'] <= train_end)]
    valid_df = df[(df['date'] >= valid_start) & 
                  (df['date'] <= valid_end)]

    if eval_start is None:
        return train_df, valid_df
    else:
        eval_df = df[(df['date'] >= eval_start) & 
                     (df['date'] <= eval_end)]
        return train_df, valid_df, eval_df

def convert_to_lgb_dataset(df_list, label="resp", 
                           ignore_cols=["date", "ts_id"]):
    """
    """
    
    df_col = df_list[0].columns
    X_col = [c for c in df_col if c != label and c not in ignore_cols]

    lgb_sets = []
    for df in df_list:
        lgb_sets.append(lgb.Dataset(df[X_col], label=df[label]))

    return lgb_sets
