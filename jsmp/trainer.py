
## Model training functions ##

# packages
import lightgbm as lgb

# internal functions
from jsmp.feat_eng import gen_tag_features

def split_data(df, 
               valid_start, valid_end=None, 
               eval_start=None, eval_end=None, 
               train_start=1, train_end=None):
    """
    Split input data into training, validation (and evaluation)
    sets by date.
    """

    # set validation date range
    if valid_end is None:
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
        if eval_end is None:
            eval_end = df['date'].max()
        else:
            pass
        
    # set training date range
    if train_end is None:
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
    