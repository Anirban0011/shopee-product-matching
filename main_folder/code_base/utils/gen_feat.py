import torch
import tqdm.notebook as tqdm
import numpy as np
from main_folder.code_base.utils.cfg import CFG


def gen_img_feats(
    dataloader,
    model,
):
    model.eval()
    bar = tqdm(dataloader)
    preds = []
    with torch.no_grad():
        for _, (images) in enumerate(bar):
            images = images.to(CFG.device)
            logits = model(images)
        preds += [logits]
        preds = torch.cat(preds).numpy()
        return logits


def gen_text_feats(
    dataloader,
    model,
):
    model.eval()
    bar = tqdm(dataloader)
    preds = []
    with torch.no_grad():
        for _, (input_ids, attention_masks) in enumerate(bar):
            input_ids, attention_masks = (
                input_ids.to(CFG.device),
                attention_masks.to(CFG.device),
            )
            logits = model(input_ids, attention_masks)
            preds += [logits]
        preds = torch.cat(preds).numpy()
        return logits
