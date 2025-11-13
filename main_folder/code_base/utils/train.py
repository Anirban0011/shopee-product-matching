import torch
import tqdm.notebook as tqdm
import numpy as np
from main_folder.code_base.utils.cfg import CFG
from main_folder.code_base.utils.f1_score import row_wise_f1_score
from sklearn.metrics import f1_score


def train_img_model(epoch, dataloader, model, loss_fn, optimizer):
    model.train()
    bar = tqdm.tqdm(dataloader)
    losses = []
    scores = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.to(CFG.device), targets.to(CFG.device).long()
        optimizer.zero_grad()
        logits = model(images, targets, epoch)  # epoch needed if use dynamic margin
        loss = loss_fn(logits, targets)
        targets = targets.detach().cpu().numpy()
        logits = torch.argmax(logits, 1).detach().cpu().numpy()
        score = f1_score(targets, logits, average="macro")
        loss.backward()
        optimizer.step()
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


def train_text_model(epoch, dataloader, model, loss_fn, optimizer):
    model.train()
    bar = tqdm.tqdm(dataloader)
    losses = []
    scores = []
    for batch_idx, (input_ids, attention_masks, targets) in enumerate(bar):
        input_ids, attention_masks, targets = (
            input_ids.to(CFG.device),
            attention_masks.to(CFG.device),
            targets.to(CFG.device).long(),
        )
        logits = model(
            input_ids, attention_masks, targets, epoch
        )  # epoch needed if use dynamic margin
        loss = loss_fn(logits, targets)
        targets = targets.detach().cpu().numpy()
        logits = torch.argmax(logits, 1).detach().cpu().numpy()
        score = f1_score(targets, logits, average="macro")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
