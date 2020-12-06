
## SCRIPT TO DOWNLOAD COMPETITION DATA LOCALLY ##

# imports
import os
from jsmp import env_config, competition_files, competition_download

# get file list
comp = "jane-street-market-prediction"
comp_files = competition_files(comp)

# get local directory
env_config("config.json")
comp_dir = os.path.join(os.environ.get("DATA_DIR"), comp)

# download each relevant file
for f in comp_files['name']:
    if f.endswith(".csv"):
        competition_download(comp, file_name=f, local_path=comp_dir)
