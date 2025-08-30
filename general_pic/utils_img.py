# -*- coding: utf-8 -*-
from PIL import Image, ImageFilter
import json

def load_ocr_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ocr_bbox_to_xyxy(bbox):
    # 你的顺序: [x1, x2, y1, y2] -> 标准 xyxy
    x1, x2, y1, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)

def apply_mosaic_boxes(img, boxes_xyxy, mosaic_block=16):
    """
    对若干 xyxy 框执行像素化（简单稳妥）
    """
    out = img.copy()
    for (x1,y1,x2,y2) in boxes_xyxy:
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(out.width-1,x2), min(out.height-1,y2)
        region = out.crop((x1,y1,x2,y2))
        # 像素化：先缩小再放大
        small = region.resize(
            (max(1, (x2-x1)//mosaic_block), max(1, (y2-y1)//mosaic_block)),
            resample=Image.NEAREST
        )
        mosaic = small.resize(region.size, Image.NEAREST)
        out.paste(mosaic, (x1,y1,x2,y2))
    return out

def save_image(img, path):
    img.save(path)
    print(f"[OK] saved -> {path}")
