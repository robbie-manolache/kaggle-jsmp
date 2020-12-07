
## SCRIPT TO CONVERT DATA TO PARTITIONED PARQUET DB ##

# imports
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jsmp import env_config

# set read and write directories
env_config("config.json")
comp = "jane-street-market-prediction"
comp_dir = os.path.join(os.environ.get("DATA_DIR"), comp)
pq_dir = os.path.join(comp_dir, "train")

# iterate through CSV chunks and write to parquet
for chunk in pd.read_csv(os.path.join(comp_dir, "train.csv"), chunksize=50000):
    print("Processing dates %d to %d"%(chunk['date'].min(), chunk['date'].max()))
    table = pa.Table.from_pandas(chunk)
    pq.write_to_dataset(table, pq_dir, partition_cols=['date'])
