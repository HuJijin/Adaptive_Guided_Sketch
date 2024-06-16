import torch as th
import numpy as np


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)
    

def compute_gauss_percent(y, optim_scale, batchsize):
    # 纵坐标切换为概率
    y = y / np.sum(y)
    # 纵坐标切换为数量
    y = y * batchsize
    y = y.astype(np.int32)
    percent = y[:optim_scale].sum() / y.sum()
    return y, percent

def find_gauss(batchsize, percent, mu=1, small_scale=1, large_scale=25, optim_scale=4):
    scale = np.arange(small_scale, large_scale+1) 
    min_diff = 1
    count = None
    for sigma in np.arange(1, 10, 0.5):
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (scale - mu)**2 / (2 * sigma**2))
        y, now_percent = compute_gauss_percent(y, optim_scale, batchsize)
        
        percent_diff = abs(now_percent - percent)
        if percent_diff < min_diff:
            min_diff = percent_diff
            count = y
    
    return scale, count

def set_guidance_scale(batchsize, percent=0.4):
    scale, count =  find_gauss(batchsize, percent)
    res = batchsize - np.sum(count)
    # 数量总和不足 batchsize，多出来的加在最后几个 scale 上
    count[-res:] += 1
    sample_scale = None
    
    for i in range(len(scale)):
        if sample_scale is None:
            sample_scale = np.full(count[i], scale[i])
        else:
            sample_scale = np.concatenate((sample_scale, np.full(count[i], scale[i])))
    return sample_scale


        

