import torch
import tqdm.notebook as tqdm
import numpy as np
from main_folder.code_base.utils.cfg import CFG

# from code_base.utils.f1_score import row_wise_f1_score
from sklearn.metrics import f1_score


def valid_img_model(
    dataloader,
    model,
    loss_fn,
    quick_eval=False,
):
    model.eval()
    bar = tqdm.tqdm(dataloader)
    losses = []
    scores = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            if (batch_idx == 100) and quick_eval:
                return net_loss, net_score
            images, targets = images.to(CFG.device), targets.to(CFG.device).long()
            logits = model(
                images, targets
            )  # epoch=0 default, margin fixed during validation
            loss = loss_fn(logits, targets)
            targets = targets.detach().cpu().numpy()
            logits = torch.argmax(logits, 1).detach().cpu().numpy()
            score = f1_score(targets, logits, average="macro")
            print(
                f"loss : {loss.item():.4f} score : {score : .4f}",
                end="\r",
                flush=True,
            )
            losses.append(loss.item())
            scores.append(score)
        print("\n")
        net_loss = np.mean(losses)
        net_score = np.mean(scores)
        return net_loss, net_score


def valid_text_model(
    dataloader,
    model,
    loss_fn,
    quick_eval=False,
):
    model.eval()
    bar = tqdm.tqdm(dataloader)
    losses = []
    scores = []
    with torch.no_grad():
        for batch_idx, (input_ids, attention_masks, targets) in enumerate(bar):
            if (batch_idx == 100) and quick_eval:
                return net_loss, net_score
            input_ids, attention_masks, targets = (
                input_ids.to(CFG.device),
                attention_masks.to(CFG.device),
                targets.to(CFG.device).long(),
            )
            logits = model(
                input_ids, attention_masks, targets
            )  # epoch=0 default, margin fixed during validation
            loss = loss_fn(logits, targets)
            targets = targets.detach().cpu().numpy()
            logits = torch.argmax(logits, 1).detach().cpu().numpy()
            score = f1_score(targets, logits, average="macro")
            print(
                f"loss : {loss.item():.4f} score : {score : .4f}",
                end="\r",
                flush=True,
            )
            losses.append(loss.item())
            scores.append(score)
        print("\n")
        net_loss = np.mean(losses)
        net_score = np.mean(scores)
        return net_loss, net_score
