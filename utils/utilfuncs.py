import torch
from pp.albu import transform
from main_folder.code_base.utils import CFG
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset, TextDataset
from main_folder.code_base.pipeline import ImgEncoder, TextEncoder

def gen_data(li,
                 lt,
                 IMG_SIZE,
                 BATCH_SIZE,
                 TKN_PATH,
                 num_workers):
    data_img = ImageDataset(li=li, transform=transform(size=IMG_SIZE))
    data_txt = TextDataset(li=lt, tokenizer=TKN_PATH)
    dataloader_img = DataLoader(data_img, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers)
    dataloader_txt = DataLoader(data_txt, batch_size=BATCH_SIZE, shuffle= False,
                            num_workers=num_workers)

    return dataloader_img, dataloader_txt

def load_model(backbone, ckpt_path, num_classes=11014, img = False):
    if img:
        model = ImgEncoder(num_classes, backbone = backbone, pretrained = False, p=4)
    else:
        model = TextEncoder(num_classes, backbone = backbone, eval_model=True)

    ckpt = torch.load(ckpt_path, weights_only=True, map_location = CFG.device)

    new_state_dict = {}

    for k, v in ckpt.items():
        new_key = k.replace("module.", "")  # remove module. prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(CFG.device)
    print(f"model {backbone} loaded successfully")
    return model

class gen_feas:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def gen_img_feas(self):

        self.model.eval()

        FEAS = []

        with torch.no_grad():
            for batch_idx, (images) in enumerate(self.dataloader):
                images = images.to(CFG.device)

                logits = self.model(images)
                FEAS += [logits.detach().cpu()]

        FEAS = torch.cat(FEAS).cpu().numpy()
        return FEAS

    def gen_txt_feas(self):

        self.model.eval()

        FEAS = []

        with torch.no_grad():
            for batch_idx, (inp_ids, att_masks) in enumerate(self.dataloader):
                inp_ids, att_masks = inp_ids.to(CFG.device), att_masks.to(CFG.device)

                logits = self.model(inp_ids, att_masks)
                FEAS += [logits.detach().cpu()]

        FEAS = torch.cat(FEAS).cpu().numpy()
        return FEAS


def return_feas(model, dataloader, img=False):
    if img:
        feas = gen_feas(model, dataloader).gen_img_feas()
    else:
        feas = gen_feas(model, dataloader).gen_txt_feas()
    feas = torch.tensor(feas).to(CFG.device)
    return feas