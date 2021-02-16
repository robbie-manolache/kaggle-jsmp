
## THE JSMP PACKAGE ##

# main functions
from jsmp.query_pq import query_train_pq, query_preds_pq
from jsmp.feat_eng import gen_tag_features, preproc_data
from jsmp.eval_tools import compute_utility, predict_return_bin, \
    confusion_matrix
from jsmp.train_prep import gen_return_bins, split_data, \
    convert_to_lgb_dataset
from jsmp.lgbm_trainer import train_lgb_classifier, gen_date_splits, \
    multi_date_lgbm_preds
from jsmp.NN_trainer import train_NN_action_model

# support functions
from jsmp.helpers.config import env_config
from jsmp.lazykaggler.competitions import competition_download, \
    competition_files, competition_list
from jsmp.lazykaggler.kernels import kernel_output_download
