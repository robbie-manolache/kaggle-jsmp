
## Functions to engineer features from JSMP training data ##

import numpy as np

def gen_tag_features(df, meta, verbose=False):
    """
    Adds Boolean columns for each tag depending on available features.
    """
    
    # copy input df and create Boolean table for non-missing features
    df_out = df.copy()
    has_feat = df.copy()[[c for c in df.columns
                          if c.startswith('feature_')]].notna()
        
    for t in range(meta.shape[1]):
        
        # matrix denoting which features must exist for this tag
        tag_mat = np.tile(meta['tag_'+str(t)], (df.shape[0], 1))
        
        # if a tag feature is missing it will equal 0
          # since F x T - T + 1 = 0
        has_tag = (has_feat * tag_mat).astype(int) - tag_mat + 1

        # if a single tag feature is missing, tag is false
        df_out.loc[:, 'tag_'+str(t)] = has_tag.min(axis=1).astype(bool)
        
        if verbose:
            print("Tag %d has been processed"%t)
            
    return df_out

def preproc_data(df, NA_fill=None, add_tags=False, tag_meta=None,
                 verbose=False):
    """
    * NA_fill = float, 'mean' or 'median'
    * If add_tags==True, must provide tag_meta
    """

    df_out = df.copy()

    # add feature tags
    if add_tags:
        df_out = gen_tag_features(df_out, tag_meta, verbose)
    else:
        pass

    # fill NA values
    if NA_fill is None:
        pass
    elif NA_fill == 'mean':
        df_out = df_out.fillna(df_out.mean())
    elif NA_fill == 'median':
        df_out = df_out.fillna(df_out.median())
    else:
        df_out = df_out.fillna(NA_fill)

    return df_out
