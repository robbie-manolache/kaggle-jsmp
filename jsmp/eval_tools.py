
## Functions to evaluate trading performance ##

import numpy as np
import matplotlib.pyplot as plt

def compute_utility(df, verbose=False, ax=None):
    """
    Note df must have 'action', 'resp', 'weight' and 'date' columns.
    TO DO:
    * expand options to provide more diagnostics.
    """
    
    # compute daily profit
    df_0 = df.copy()
    df_0.loc[:, 'profit'] = df['weight'] * df['resp'] * df['action'] 
    daily_profit = df_0.groupby('date')['profit'].sum()
    
    # compute utility and its components 
    p = np.sum(daily_profit)
    v = np.sqrt(np.sum(daily_profit ** 2)*len(daily_profit)/250)
    t = p / v
    u = min(max(0, t), 6) * p
    
    if verbose:      
        print("Profit: %.2f | Volatility: %.2f | "%(p, v) +\
              "Sharpe Ratio: %.2f | Utility: %.2f"%(t, u))
        if ax is None:
            _, ax = plt.subplots(figsize=(10,4))
        else:
            pass
        daily_profit.plot(ax=ax)
        plt.ylabel("Daily Profit")
        plt.show()
    
    return u, daily_profit
