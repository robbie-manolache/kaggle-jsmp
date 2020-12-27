
## Functions to navigate Parquet version of JSMP training data ##

import pyarrow.parquet as pq

def query_train_pq(pq_dir, date_range=None, 
                   id_cols=['date', 'ts_id'], 
                   return_cols='all', features='all'):
    """
    Allows fast, seamless querying of training data parquet database.
    ** ARGS **
    pq_dir: str, path to parquet directory
    date_range: list of int, value of first and last date to query between
    id_cols: list of str
    return_cols: list of str
    features: list of str or list of int, if list of str it each
        element must start with 'feature_'
    """
    
    # initialise row date filters
    if date_range is not None:
        filters = [('date', '>=', date_range[0]), 
                   ('date', '<=', date_range[1])]
    else:
        filters = None
    
    # establish parquet connection
    pq_con = pq.ParquetDataset(pq_dir, filters=filters)
    
    # compile list of columns to query
    all_cols = pq_con.schema.names
    
    # select return columns (including weight)
    if  return_cols == "all":
        return_cols = [c for c in all_cols if (c not in ['date', 'ts_id'])
                       and (not c.startswith('feature_'))]
    else:
        pass
    
    # select features
    if features == "all":
        features = [c for c in all_cols if c.startswith('feature_')]
    elif type(features) == list:
        if all([type(f) == int for f in features]):
            features=['feature_' + str(f) for f in features] 
        else:
            pass
    else:
        pass

    # read data and convert to pandas
    columns = id_cols + features + return_cols
    df = pq_con.read(columns=columns).to_pandas()
    
    # convert date to int
    if 'date' in df.columns:
        df.loc[:, 'date'] = df['date'].astype(int)
    else:
        pass
    
    return df
