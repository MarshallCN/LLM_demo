# convert_jsonl.py
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="input .jsonl file")
    ap.add_argument("out", help="output .json file")
    args = ap.parse_args()

    items = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"role": obj["role"], "content": obj["content"]})

    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(items, w, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
