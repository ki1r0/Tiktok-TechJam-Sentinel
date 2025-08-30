# whisper_cli.py  —— 独立进程里跑 whisper，stdout 返回 JSON
import argparse, json, sys
import whisper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="仅测试可用性")
    ap.add_argument("--model", default="base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("audio_path", nargs="?")
    args = ap.parse_args()

    if args.ping:
        # 只验证能否 import，不加载模型，启动很快
        print(json.dumps({"ok": True}))
        return

    if not args.audio_path:
        print(json.dumps({"ok": False, "error": "no audio"}))
        sys.exit(2)

    # 真正转写：加载模型（CPU），fp16=False 保守
    model = whisper.load_model(args.model, device=args.device)
    result = model.transcribe(args.audio_path, fp16=False)
    text = (result.get("text") or "").strip()
    print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))

if __name__ == "__main__":
    main()
