
## Functions to evaluate trading performance ##

import numpy as np

def compute_utility(df):
    """
    Note df must have 'action', 'resp', 'weight' and 'date' columns.
    TO DO:
    * expand options to provide more diagnostics.
    """
    df_0 = df.copy()
    df_0.loc[:, 'profit'] = df['weight'] * df['resp'] * df['action'] 
    daily_profit = df_0.groupby('date')['profit'].sum()
    t = (np.sum(daily_profit) / np.sqrt(np.sum(daily_profit ** 2))) \
        * np.sqrt(250/len(daily_profit))
    u = min(max(0, t), 6) * np.sum(daily_profit)
    return u
