from main_folder.code_base.utils import CFG
import faiss
import torch

def build_faiss(feas, dim):
    if CFG.device.type == "cpu":
        index = faiss.IndexFlatIP(dim)
    else :
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
    index.add(feas)
    return index

def get_batches(bs, n_batch, feas):
    batches = []
    for i in range(n_batch):
        left = bs * i
        right = bs * (i+1)
        if i == n_batch - 1:
            right = feas.shape[0]
        batches.append(feas[left:right,:])
    return batches

def get_matches(bs, n_batch, feas, dim, k=51):
    index = build_faiss(feas, dim)
    m=[]
    s=[]
    for batch in get_batches(bs, n_batch, feas):
        batch = batch.to(CFG.device)
        sims, matches = index.search(batch, k)
        m.append(matches)
        s.append(sims)
    m = torch.cat(m, dim=0).to(torch.int32)
    s = torch.cat(s, dim=0)
    return m,s

def th_matches(bs, n_batch, matches, sims, th):
    matches = get_batches(bs, n_batch, matches)
    sims = get_batches(bs, n_batch, sims)
    m = []
    s=[]
    for (batch_m, batch_s) in zip(matches, sims):
        batch_m = batch_m.cpu().numpy()
        batch_s = batch_s.cpu().numpy()
        mask = (batch_s > th)
        for row in range(len(mask)):
            m.append(batch_m[row][mask[row]].tolist())
            s.append(batch_s[row][mask[row]].tolist())
    return m, s