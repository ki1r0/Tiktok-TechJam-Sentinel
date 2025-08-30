# -*- coding: utf-8 -*-
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM-Llama3-V-2_5"  # 显存不够时可换成："openbmb/MiniCPM-Llama3-V-2_5-int4"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 放一张测试图到当前目录，命名 test.jpg
    image = Image.open("ocr_outputs/a_ocr_vis.jpg").convert("RGB")
    msgs = [{"role": "user", "content": "point out all the index that may contain personal information in the image."}]

    out = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    print("\n=== 模型输出 ===\n", out)

if __name__ == "__main__":
    main()
