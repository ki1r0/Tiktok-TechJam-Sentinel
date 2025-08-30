import cv2
import json
import numpy as np
from pathlib import Path
import easyocr

def draw_poly(img, poly, color=(0, 255, 0), thickness=2):
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def poly_to_xyxy_xywh(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return [x1, y1, x2, y2], [x1, y1, x2 - x1, y2 - y1]

def draw_index_badge(img, anchor_xy, idx, radius=14, alpha=0.35):
    """画半透明圆形徽标，写序号 idx"""
    x, y = anchor_xy
    overlay = img.copy()
    cv2.circle(overlay, (x, y), radius, (0, 0, 0), thickness=-1)
    cv2.circle(overlay, (x, y), radius-2, (255, 255, 255), thickness=2)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    text = str(idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = x - tw // 2
    ty = y + th // 3
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def run_ocr(
    image_path,
    output_dir="ocr_outputs",
    langs=['en', 'ch_sim'],
    conf_thres=0.3,
    draw=True,
    use_gpu=False
):
    """
    对单张图片执行 OCR, 输出 JSON 和可视化图片.
    """
    reader = easyocr.Reader(langs, gpu=use_gpu)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"[ERR] File not found: {image_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERR] Cannot read image: {image_path}")
        return

    results = reader.readtext(str(image_path))
    items = []
    idx_counter = 1

    for (box, text, conf) in results:
        if conf < conf_thres or not str(text).strip():
            continue

        poly = [(int(x), int(y)) for (x, y) in box]
        xyxy, _ = poly_to_xyxy_xywh(poly)
        x1, y1, x2, y2 = xyxy

        # JSON 里顺序：x1, x2, y1, y2
        bbox_for_json = [x1, x2, y1, y2]

        items.append({
            "index": idx_counter,
            "bbox": bbox_for_json,
            "content": str(text),
            "confidence": float(conf)
        })

        if draw:
            draw_poly(img, poly, (0, 255, 0), 2)
            badge_x = max(0, x1 - 10)
            badge_y = max(0, y1 - 10)
            draw_index_badge(img, (badge_x, badge_y), idx_counter)

        idx_counter += 1

    stem = image_path.stem

    # 写 JSON
    json_path = output_dir / f"{stem}_ocr.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 写可视化图
    if draw:
        vis_path = output_dir / f"{stem}_ocr_vis.jpg"
        cv2.imwrite(str(vis_path), img)

    print(f"[OK] {image_path.name}: {len(items)} items -> {json_path.name}" + (", vis saved" if draw else ""))
    return json_path


# ========== 自测入口 ==========
if __name__ == "__main__":
    run_ocr(
        image_path="image_example/a.jpg",   # 单张图片
        output_dir="ocr_outputs",
        langs=['en', 'ch_sim'],
        conf_thres=0.1,
        draw=True,
        use_gpu=False
    )
