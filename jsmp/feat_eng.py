
## Functions to engineer features from JSMP training data ##

def assign_tag(row, tag, meta):
    """
    """
    row = row[[c for c in row.index if c.startswith('feature_')]]
    row = row.notna()
    return all((row * meta['tag_'+str(tag)]) == meta['tag_'+str(tag)])

def gen_tag_features(df, meta, verbose=False):
    """
    """
    df_out = df.copy()
   
    for t in range(meta.shape[1]):
        df_out.loc[:, 'tag_'+str(t)] = df_out.apply(assign_tag, 
                                                    axis=1, 
                                                    args=(t, meta))
        if verbose:
            print("Tag %d has been processed"%t)
            
    return df_out
