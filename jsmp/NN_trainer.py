
## Functions for to train action optimization model ##

import os
import json
from datetime import datetime
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from jsmp.query_pq import query_train_pq, query_preds_pq
from jsmp.eval_tools import compute_utility

def __profit_maximizer__(y_true, y_pred):
    """
    """
    L = - K.sum(y_true * y_pred)
    return L   

def train_NN_action_model(train_config,
                          train_dir,                         
                          model_dir=None,
                          thresholds=None,
                          overwrite=False,
                          verbose=False,
                          show_train_progress=0):
    """
    """

    # Set seed
    np.random.seed(train_config["seed"])
    
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
                if overwrite:
                    pass
                else:
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
    if train_config["lgbm_preds"] is None:
        pass
    else:
        rc = query_preds_pq(train_config["lgbm_preds"])
        df = df.merge(rc, on=['ts_id', 'date'])
        rc = None
        X_col += [f for f in df.columns if f.startswith("resp_bin_")]
    
    # Eval cutoff
    eval_start = train_config["eval_start"]
    if eval_start is None:
        eval_start = df['date'].max() + 1
    else:
        pass
    
    # Training set 
    x_train = np.nan_to_num(df.query('date < @eval_start')[X_col].values)
    y_train = df.query('date < @eval_start')['target'].values
    
    # Test set & prediction frame
    if eval_start > df['date'].max():
        eval_sets = False
    else:
        x_test = np.nan_to_num(df.query('date >= @eval_start')[X_col].values)
        y_test = df.query('date >= @eval_start')['target'].values
        pred_df = df.query('date >= @eval_start')[['date', 'resp', 'weight']]
        eval_sets = True

    # Create model structure
    NN_params = train_config["NN_params"]
    n_layer = len(NN_params["layers"])
    model = keras.Sequential()
    for l in range(n_layer):
        if l == 0:
            if NN_params["pre_drop"] is None:
                pass
            else:
                model.add(keras.layers.Dropout(NN_params["pre_drop"]))
            model.add(keras.layers.Dense(
                NN_params["layers"][l],
                input_shape=[x_train.shape[1]],
                activation=NN_params["actifun"][l],
                kernel_regularizer=keras.regularizers.l2(NN_params["L2"][l])))
        else:
            model.add(keras.layers.Dense(
                NN_params["layers"][l],
                activation=NN_params["actifun"][l],
                kernel_regularizer=keras.regularizers.l2(NN_params["L2"][l])))
        model.add(keras.layers.Dropout(NN_params["dropout"][l]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile and fit model
    model.compile(loss=__profit_maximizer__, 
                  optimizer=keras.optimizers.Adam(learning_rate=NN_params["lr"]))
    if eval_sets:
        model.fit(x_train, y_train, 
                  epochs=NN_params["n_epoch"], 
                  batch_size=np.ceil(x_train.shape[0]/
                                     NN_params["n_batch"]).astype(int), 
                  validation_data=(x_test, y_test),
                  verbose=show_train_progress)
    else:
        model.fit(x_train, y_train, 
                  epochs=NN_params["n_epoch"], 
                  batch_size=np.ceil(x_train.shape[0]/
                                     NN_params["n_batch"]).astype(int),
                  verbose=show_train_progress)
    
    # Generate predictions
    results = {}
    if eval_sets and thresholds is not None:
        preds = model.predict(x_test)
        results = {}
        for t in thresholds:
            pred_df.loc[:, "action"] = (preds > t).astype(int)
            u, profit = compute_utility(pred_df, verbose=False)
            results[t] = (u/len(profit), np.mean(profit), np.std(profit))
            if verbose:
                print("Threshold: %.2f | Profit: %.1f | Volatility: %.1f | Utility: %.1f"
                      %(t, np.mean(profit), np.std(profit), u))
            else:
                pass
    else:
        pass
    
    # Save model
    if model_dir is None:
        pass
    else:
        model_name = "NN_%s"%datetime.now().strftime("%Y%m%d_%H%M%S")
        model_json = model.to_json()
        with open(os.path.join(model_dir, model_name+".json"), "w") as wf:
            json.dump(model_json, wf)
        model.save_weights(os.path.join(model_dir, model_name+".h5"))
    
        # Save metadata
        meta[model_name] = {
            "train_config": train_config,
            "results": results
        }
        with open(os.path.join(model_dir, "meta.json"), 'w') as f:
            json.dump(meta, f)
    
    return model

def load_model_from_disk(model_dir, model_name):
    """
    """
    with open(os.path.join(model_dir, model_name+".json")) as rf:
        model_json = json.load(rf)

    model = keras.models.model_from_json(model_json)
    model.load_weights(os.path.join(model_dir, model_name+".h5"))

    return model
    
def save_weights_to_json(model, n_layer, json_path):
    """
    """
    names = [s + str(i) for i in range(n_layer) for s in ["w", "b"]]
    weights = {}
    for i, a in enumerate(model.get_weights()):
        weights[names[i]] = a.tolist()
    with open(json_path, "w") as wf:
        json.dump(weights, wf)
   
    