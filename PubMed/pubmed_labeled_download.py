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


def http_download(url: str, dst: Path):
    """HTTP(S)로 파일 다운로드"""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def extract_zip(archive_path: Path, target_dir: Path):
    """ZIP 파일 압축 해제"""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        z.extractall(target_dir)


# PubMedQA 다운로드
print("[PubMedQA Official] 다운로드 시작")
name = "PubMedQA_official"
url = "https://codeload.github.com/pubmedqa/pubmedqa/zip/refs/heads/master"

dst = RAW_DIR / name / "pubmedqa.zip"
dst.parent.mkdir(parents=True, exist_ok=True)

http_download(url, dst)

outdir = RAW_DIR / name / "pubmedqa-master"
print(f"  압축 해제: {dst.name} → {outdir}")
extract_zip(dst, outdir)

