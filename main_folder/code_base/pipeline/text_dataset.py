import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from main_folder.code_base.utils import clean_text

class SHOPEETextDataset(Dataset):
    def __init__(self, df, tokenizer=None, gen_feat_only=False, clean=False):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.only_feat = gen_feat_only
        self.clean = clean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        text = row.title
        if self.clean:
            text = clean_text(text)
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        if self.only_feat:
            return input_ids, attention_mask
        return input_ids, attention_mask, torch.tensor(row.label_group).float()
