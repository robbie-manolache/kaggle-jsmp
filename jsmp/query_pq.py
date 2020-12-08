
## Functions to navigate Parquet version of JSMP training data ##

import pyarrow.parquet as pq

def query_dates(pq_dir, dmin, dmax):
    """
    Future improvements:
    -> allow feature/column selection
    """
    
    # establish parquet connection
    pq_con = pq.ParquetDataset(pq_dir, filters=[('date', '>=', dmin), 
                                                ('date', '<=', dmax)])    
    # read data and convert to pandas
    df = pq_con.read().to_pandas()
    return df

