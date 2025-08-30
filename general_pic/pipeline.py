# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from grounding import run_grounding
from ocr import run_ocr
from utils_img import load_ocr_json, ocr_bbox_to_xyxy, apply_mosaic_boxes, save_image
from PIL import ImageDraw

# ========== 路径配置 ==========

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "Qwen2.5-VL-3B-Instruct"
# MODEL_PATH = str(MODEL_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# INPUT_IMG = Path("test_pic/0003.jpg")
OCR_DIR   = Path("ocr_outputs")
OUT_DIR   = Path("outputs")

# ========== 加载模型 ==========
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


def ask_qwen(image, prompt: str) -> str:
    """统一问模型，返回纯文本"""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.batch_decode(output, skip_special_tokens=True)[0]


def extract_json(text: str, default=None):
    text = text.strip()
    # 直接尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 如果失败，逐个提取大括号 {...}
    objs, stack, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    objs.append(text[start:i+1])

    for obj in reversed(objs):  # 从最后一个开始尝试
        try:
            return json.loads(obj)
        except Exception:
            continue

    return default if default is not None else {}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) OCR
    run_ocr(INPUT_IMG, OCR_DIR)  # 你现有的函数，生成 json
    ocr_json_path = OCR_DIR / f"{INPUT_IMG.stem}_ocr.json"
    items = load_ocr_json(ocr_json_path)

    img = Image.open(INPUT_IMG).convert("RGB")

    # 2) 第一次：判定哪些 index 是 PII
    ocr_lines = "\n".join([f"{it['index']}. {it['content']}" for it in items])
    prompt1 = (
        "Among these OCR results, which indices correspond to personal private information "
        "(such as name, phone number, email, ID card, license plate, home address, house number, unit number, community name, etc.)?\n"
        "Note: Text on signboards, door plates, unit plates, or residential/community name plates should usually be considered PII.\n"
        "Don't only pay attention to the ocr text, but also pay attention to the image which will help you better judge which text is pii\n"
        "Please output STRICTLY in JSON format: {\"pii_indices\": [..]}\n\n"
        + ocr_lines
    )


    out1 = ask_qwen(img, prompt1)
    print("Model output:", out1)

    pii_result = extract_json(out1, {"pii_indices":[]})
    pii_indices = set(pii_result.get("pii_indices", []))
    print("Detected PII indices:", pii_indices)

    # 3) 打码原图
    boxes = []
    for it in items:
        if it["index"] in pii_indices:
            x1,y1,x2,y2 = ocr_bbox_to_xyxy(it["bbox"])
            boxes.append([x1,y1,x2,y2])
    print("bboxes to redact:", boxes)
    redacted = apply_mosaic_boxes(img, boxes, mosaic_block=18)

    ext = INPUT_IMG.suffix.lower() if INPUT_IMG.suffix else ".png"
    redacted_path = OUT_DIR / f"{INPUT_IMG.stem}_redacted{ext}"
    save_image(redacted, redacted_path)

    # 4) 第二次：调用 grounding.py 复核
    grd = run_grounding(str(redacted_path), model_path=MODEL_PATH)

    boxes2 = grd.get("boxes", [])
    # grounding 返回的 bbox 已经是 [x1,y1,x2,y2]
    mosaic_boxes = [b.get("bbox", [0,0,0,0]) for b in boxes2]

    # 在 redacted 图的基础上再次打码
    image_final = apply_mosaic_boxes(redacted, mosaic_boxes, mosaic_block=18)

    ext = INPUT_IMG.suffix.lower() if INPUT_IMG.suffix else ".png"
    final_img_path = OUT_DIR / f"{INPUT_IMG.stem}_final{ext}"
    save_image(image_final, final_img_path)

if __name__ == "__main__":
    from pathlib import Path
    import shutil
    TEST_DIR = Path("inputs")
    ARCHIVE_DIR = Path("archive")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for i, img_path in enumerate(TEST_DIR.iterdir(), 1):
        if img_path.suffix.lower() in exts:
            INPUT_IMG = img_path
            print(f"[{i}] Processing {INPUT_IMG.name}...")
            main()
        try:
            target = ARCHIVE_DIR / img_path.name
            shutil.move(str(img_path), str(target))
            print(f"Archived {img_path.name} → {target}")
        except Exception as e:
            print(f"Failed to archive {img_path.name}: {e}")