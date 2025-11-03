import os
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from datasets import load_dataset


RAW_DIR = Path("data/raw").resolve()
RAW_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", None)
OFFLINE = os.getenv("OFFLINE", "0") == "1"

def export_hf_to_jsonl(ds, out_path: Path):
    """HuggingFace 데이터셋을 JSONL로 저장"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# PubMedQA Artificial 다운로드
print("[PubMedQA Official] 다운로드 시작")
name = "PubMedQA_official"
url = "https://codeload.github.com/pubmedqa/pubmedqa/zip/refs/heads/master"

dst = RAW_DIR / name / "pubmedqa.zip"
dst.parent.mkdir(parents=True, exist_ok=True)


print("[PubMedQA HF Artificial] 다운로드 시작")

name = "PubMedQA_hf_artificial"
repo_id = "qiaojin/pubmedqa"
config = "pqa_artificial"
split = "train"

target_dir = RAW_DIR / name
target_dir.mkdir(parents=True, exist_ok=True)

out_path = target_dir / f"pubmedqa_{config}_{split}.jsonl"

ds = load_dataset(
        repo_id,
        name=config,
        split=split,
        token=HF_TOKEN,
        trust_remote_code=True
    )
print(f"  저장: {out_path}")
export_hf_to_jsonl(ds, out_path)



# # HuggingFace unlabeled 
# print("[PubMedQA HF Unlabeled] 다운로드 시작")
# name = "PubMedQA_hf_unlabeled"
# repo_id = "qiaojin/pubmedqa"
# config = "pqa_unlabeled"
# split = "train"

# target_dir = RAW_DIR / name
# target_dir.mkdir(parents=True, exist_ok=True)

# out_path = target_dir / f"pubmedqa_{config}_{split}.jsonl"

# if out_path.exists():
#     print(f"  이미 존재: {out_path}")
# else:
#     print(f"  로드: {repo_id} | config={config} | split={split}")
#     ds = load_dataset(
#         repo_id,
#         name=config,
#         split=split,
#         token=HF_TOKEN,
#         trust_remote_code=True
#     )
#     print(f"  저장: {out_path}")
#     export_hf_to_jsonl(ds, out_path)
