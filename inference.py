import gc
import torch
from utils.predict import predict
from utils.filterfunc import filter_match_titles
from utils.ckpts import img_ckpt, txt_ckpt
from utils.utilfuncs import gen_data, load_model, return_feas

img_backbone = ["timm/eca_nfnet_l1.ra2_in1k"]
txt_backbone = ["google-bert/bert-base-uncased"]

def clean():
    gc.collect()

def inference(li, lt, IMG_SIZE,
              TKN_PATH,
              BATCH_SIZE,
              num_workers = 4,
              ):
    dataloader_img, dataloader_txt = gen_data(li,
                                              lt,
                                              IMG_SIZE,
                                              BATCH_SIZE,
                                              TKN_PATH[0],
                                              num_workers)

    img_model = [load_model(backbone=img_backbone[i],
                            ckpt_path=img_ckpt[i],
                            img=True)
             for i in range(len(img_backbone))]

    img_feas = torch.cat([return_feas(
                img_model[i],
                dataloader_img, img=True)
            for i in range(len(img_backbone))], dim=1)

    txt_model = [load_model(backbone=TKN_PATH[i], ckpt_path=txt_ckpt[i])
             for i in range(len(txt_backbone))]

    txt_feas = torch.cat([return_feas(
                txt_model[i],
                dataloader_txt)
            for i in range(len(txt_backbone))], dim=1)

    match_final =  predict(img_feas=img_feas,
                   txt_feas=txt_feas)

    match_final = filter_match_titles(match_final, title_list=lt)

    assert len(match_final == 2)

    return set(match_final[0]) == set(match_final[1])





