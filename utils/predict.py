import torch
import numpy as np
from main_folder.code_base.utils import CFG
import torch.nn.functional as F
from functools import reduce
from utils.knn import get_matches, th_matches

K = 51
di = 1792
dt = 1024
dc = di+dt
n_batch = 10

def filter_embeddings(feas, matches, sims):
    feas = feas.detach().cpu()
    new_feas = feas.clone()

    for i in range(feas.shape[0]):
        cur_feas = feas[matches[i]]
        weights = torch.unsqueeze(torch.Tensor(sims[i]), 1)
        new_feas[i] = weights.T@cur_feas
    new_feas = F.normalize(new_feas)
    return new_feas.to(CFG.device)

def filter_matches(matches, sims, th=1.0, k=3, dist=1e-2):
    top_matches = [row[:k] for row in matches]
    top_sims = [row[:k] for row in sims]
    for i in range(len(matches)):
        if len(matches[i]) < k+1:
            continue
        dist_1 = sims[i][k-2] - sims[i][k-1]
        dist_2 = sims[i][k-1] - sims[i][k]
        if dist_2 < dist:
            continue
        if th*dist_1 < dist_2:
            matches[i] = top_matches[i]
            sims[i] = top_sims[i]
    return matches, sims

def union_matches(*lists):
    matches = []
    for group in zip(*lists):
        matches.append(reduce(np.union1d, group).tolist())
    return matches

def predict(img_feas, txt_feas):

    img_feas, txt_feas = F.normalize(img_feas).to(CFG.device) , F.normalize(txt_feas).to(CFG.device)
    comb_feas = F.normalize(torch.cat([img_feas, txt_feas], dim=1)).to(CFG.device)

    bs  = len(comb_feas) // n_batch

    img_matches, img_sims = get_matches(bs, n_batch, img_feas, di, k=K)
    text_matches, text_sims = get_matches(bs, n_batch, txt_feas, dt, k=K)
    comb_matches, comb_sims = get_matches(bs, n_batch, comb_feas, dc, k=K)

    img_final, img_sims = th_matches(bs, n_batch, img_matches, img_sims, 0.704)
    text_final, text_sims = th_matches(bs, n_batch, text_matches, text_sims, 0.764)
    comb_final, comb_sims = th_matches(bs, n_batch, comb_matches, comb_sims, 0.52)

    comb_feas = filter_embeddings(comb_feas, comb_final, comb_sims)
    comb_matches, comb_sims = get_matches(bs, n_batch, comb_feas, dc, k=K)
    comb_final, comb_sims = th_matches(bs, n_batch, comb_matches, comb_sims, 0.9)

    img_final,_ = filter_matches(img_final, img_sims, 1.1, 4, 2e-2)
    text_final,_ = filter_matches(text_final, text_sims, 1.2, 4, 2e-2)
    comb_final,_ = filter_matches(comb_final, comb_sims, 1.0, 3, 2e-2)

    match_final = union_matches(img_final, text_final, comb_final)

    return match_final






