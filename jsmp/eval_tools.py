
## Functions to evaluate trading performance ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def predict_return_bin(eval_df, lgb_model,
                       eval_cols=["date", "ts_id", "weight", 
                                  "resp", "resp_bin"]):
    """
    Predict return bins using LGBM classifier.
    """
    
    # copy eval_df frame
    df = eval_df.copy()[eval_cols]
    
    # make predictions
    preds = lgb_model.predict(eval_df[lgb_model.feature_name()])
    
    # add predictions to df
    pred_cols = ["resp_bin_" + str(i) for i in range(preds.shape[1])]
    df.loc[:, pred_cols] = preds
    df.loc[:, "pred_bin"] = preds.argmax(axis=1)
    
    return df

def confusion_matrix(df, n_bins,
                     actual="resp_bin", prediction="pred_bin",
                     as_percent=True, ax=None):
    """
    Create confusion matrix to visualize classification performance.
    """
    
    # create matrix
    cfmat = pd.crosstab(df[prediction], df[actual])
    
    # convert to fraction
    if as_percent:
        cfmat = cfmat / cfmat.sum(axis=1).values.reshape((n_bins, 1))
        vmin, vmax, fmt = 0, 1, ".2%"
    else:
        vmin, vmax, fmt = None, None, ".0f"
    
    # define ax
    if ax is None:
        _, ax = plt.subplots(figsize=(10,10))
    else:
        pass    
    
    # generate plot
    sns.heatmap(cfmat, cbar=False, annot=True, fmt=fmt, 
                ax=ax, vmin=vmin, vmax=vmax, cmap='Blues')
    plt.xlabel("Actual Class")
    plt.ylabel("Predicted Class")
    plt.show()
    