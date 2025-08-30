# -*- coding: utf-8 -*-
import os, re, json
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# 默认模型路径（可在调用时改）
DEFAULT_MODEL_PATH = "../model/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 加载模型 =====
_model, _processor = None, None
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _model, _processor
    if _model is None or _processor is None:
        _model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return _model, _processor


# ===== JSON 提取工具 =====
def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _remove_line_comments(s: str) -> str:
    return re.sub(r"//.*", "", s)

def extract_json_from_assistant(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    objs, stack, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    objs.append(s[start:i+1])

    candidates = [o for o in objs if '"boxes"' in o] or objs
    for obj in reversed(candidates):
        cleaned = _remove_line_comments(_strip_trailing_commas(obj))
        try:
            return json.loads(cleaned)
        except Exception:
            continue
    return None


# ===== 核心函数 =====
def run_grounding(image_path: str, model_path: str = DEFAULT_MODEL_PATH, device: str = DEVICE) -> Dict[str, Any]:
    """
    输入图片路径 -> 调用 MLLM -> 输出 JSON {boxes:[], image_size:{}}
    """
    model, processor = load_model(model_path)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    SYSTEM = (
        "You are a privacy redaction assistant.\n"
        "Find all regions that may contain personal/private information in the image.\n"
        "Examples include: student ID card (even if blurred), phone screen (even if blurred), "
        "Do NOT treat a whole person or human face as private information.\n"
        "street/block codes, people's names, license plates, credit cards, QR codes, etc.\n"
        "Be conservative: include all likely private regions.\n"
        "Output STRICT JSON ONLY in the following schema:\n"
        "{\n"
        '  "boxes": [\n'
        '    {"label": "name", "bbox": [10,20,50,60], "score": 0.9},\n'
        '    {"label": "phone", "bbox": [100,200,180,240], "score": 0.85}\n'
        "  ],\n"
        f'  "image_size": {{"width": {W}, "height": {H}}}\n'
        "}\n"
        "Rules:\n"
        "- Coordinates must be integers within [0,width]x[0,height].\n"
        "- Use the ORIGINAL image size provided.\n"
        "- Output JSON only, no explanation.\n"
    )

    USER = f"Image size: width={W}, height={H}. Detect bounding boxes for private information and return JSON only."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            output_scores=False
        )

    gen_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
    pred = extract_json_from_assistant(gen_text)
    if pred is None:
        print(f"[WARN] {os.path.basename(image_path)}: 无法解析模型输出")
        return {"boxes": [], "image_size": {"width": W, "height": H}}

    return pred


# ===== 自测 =====
if __name__ == "__main__":
    test_dir = "test_pic"
    out_dir = "annotated_pic"
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(test_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(test_dir, fname)
        result = run_grounding(path)

        image = Image.open(path).convert("RGB")
        W, H = image.size
        boxes = result.get("boxes", [])
        print(f"{fname}: 检测到 {len(boxes)} 个框")

        # 画框
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = None

        for b in boxes:
            label = str(b.get("label", "unknown"))
            score = b.get("score", 0.0)
            x1, y1, x2, y2 = b.get("bbox", [0,0,0,0])
            x1, y1, x2, y2 = map(int, [max(0,min(W,x1)), max(0,min(H,y1)), max(0,min(W,x2)), max(0,min(H,y2))])

            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            caption = f"{label}:{score:.2f}"
            if font:
                bbox = draw.textbbox((0,0), caption, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                tw, th = len(caption)*8, 16
            pad = 2
            draw.rectangle([x1, y1-th-2*pad, x1+tw+2*pad, y1], fill=(255, 0, 0))
            draw.text((x1+pad, y1-th-pad), caption, fill=(255,255,255), font=font)

        out_path = os.path.join(out_dir, fname)
        image.save(out_path)
        print(f"结果已保存 -> {out_path}")
