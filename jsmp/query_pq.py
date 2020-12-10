
## Functions to navigate Parquet version of JSMP training data ##

import pyarrow.parquet as pq

def query_train_pq(pq_dir, date_range=None, 
                   id_cols=['date', 'ts_id'], 
                   return_cols='all', features='all'):
    """
    Allows fast, seamless querying of training data parquet database.
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
    else:
        pass
 
    # read data and convert to pandas
    columns = id_cols + features + return_cols
    df = pq_con.read(columns=columns).to_pandas()
    return df

