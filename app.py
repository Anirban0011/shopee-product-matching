from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import torch
import nltk
nltk.download("stopwords")
import numpy as np
from typing import List
from inference import inference
from main_folder.code_base.utils import CFG

TKN_PATH= ["bert-base-uncased"]
IMG_SIZE = 256
BATCH_SIZE = 32
img = True

CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="shopee-test-app")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Space is running"}

@app.post("/predict")
async def predict_image(files: List[UploadFile] = File(...),
                          texts: List[str] = Form(...)):
    li, lt= [], []
    for file, text in zip(files, texts):
        contents = await file.read()
        li.append(contents)
        lt.append(text)
    res = inference(li=li,
                    lt=lt,
                    IMG_SIZE=IMG_SIZE,
                    TKN_PATH=TKN_PATH,
                    BATCH_SIZE=BATCH_SIZE
                    )
    msg = "products matched" if res else "products not matched"

    return {"message" : f"{msg}"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)








