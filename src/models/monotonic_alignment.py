"""Native monotonic dynamic programming; no recognizer or external alignment weights."""
import numpy as np
import torch

def monotonic_path(cost):
    """Min-cost path assigns every frame to one ordered unit, visits every unit."""
    values=cost.detach().float().cpu().numpy()
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Finite [frames,units] costs required')
    frames,units=values.shape
    if units<1 or frames<units:raise ValueError('At least one frame per unit required')
    previous=np.full(units,np.inf,np.float32);previous[0]=values[0,0]
    advance=np.zeros((frames,units),bool)
    for t in range(1,frames):
        shifted=np.r_[np.inf,previous[:-1]]
        advance[t]=shifted<previous
        current=values[t]+np.minimum(previous,shifted)
        positions=np.arange(units)
        current[(positions>t)|(positions<units-1-(frames-1-t))]=np.inf
        previous=current
    path=np.empty(frames,np.int64);unit=units-1
    for t in range(frames-1,-1,-1):
        path[t]=unit
        if t>0 and advance[t,unit]:unit-=1
    if unit!=0:raise RuntimeError('Invalid alignment path')
    return torch.from_numpy(path).to(cost.device)
