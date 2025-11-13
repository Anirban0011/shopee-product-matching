from huggingface_hub import hf_hub_download

REPO_ID = "Anirban0011/multimodal-shopee-finetune"

def get_path(filename, repo):
    path = hf_hub_download(repo_id=repo, filename=filename)
    return path

img_path = get_path(repo=REPO_ID,
                            filename="img_model_eca_nfnet_l1.ra2_in1k.pth")
txt_path = get_path(repo=REPO_ID,
                            filename="txt_model_bert-base-uncased_35.pth")

img_ckpt = [img_path]
txt_ckpt = [txt_path]