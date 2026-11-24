"""Full-corpus reciprocal-rank blend shared by cached and pixel inference."""
import numpy as np

def blended_scores(raw,adapted,weight=0,constant=60):
    if raw.shape!=adapted.shape or raw.ndim!=2 or not np.isfinite(raw).all() or not np.isfinite(adapted).all():raise ValueError('Invalid ranking matrices')
    if not 0<=weight<=1 or constant!=60:raise ValueError('Invalid rank blend')
    if weight==0:return adapted
    if weight==1:return raw
    result=np.zeros_like(raw,dtype=np.float32)
    values=np.asarray(1/(constant+np.arange(1,raw.shape[1]+1)),dtype=np.float32)[None,:]
    for matrix,factor in [(raw,weight),(adapted,1-weight)]:
        ranks=np.argsort(-matrix,axis=1,kind='stable');channel=np.empty_like(result)
        np.put_along_axis(channel,ranks,values,axis=1);result+=factor*channel
    return result
