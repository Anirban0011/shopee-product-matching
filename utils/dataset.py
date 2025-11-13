import io
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, li, transform=None):
        self.li = li
        self.transform = transform

    def __len__(self):
        return len(self.li)

    def __getitem__(self, index):
        img_byte = self.li[index]
        img = Image.open(io.BytesIO(img_byte)).convert("RGB")
        img = np.array(img)
        img = img.copy()

        if self.transform is not None:
            img = self.transform(image=img)
            img = img["image"]
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)

        return torch.tensor(img).float()

class TextDataset(Dataset):
    def __init__(self, li, tokenizer=None):
        self.li = li
        self.to = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.li)

    def __getitem__(self, index):
        text = self.li[index]
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        return input_ids, attention_mask


