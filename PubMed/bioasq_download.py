#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_bioasq.py
"""
import os
import json
import zipfile
from pathlib import Path

# 데이터 저장 디렉토리
RAW_DIR = Path("data/raw").resolve()
BIOASQ_DIR = RAW_DIR / "BioASQ"

def setup_directories():
    """필요한 디렉토리 생성"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BIOASQ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] 디렉토리 생성: {BIOASQ_DIR}")

def handle_bioasq():
    
    print(f"[BIOASQ] 처리 시작: {BIOASQ_DIR}")
    
    # 1) ZIP 자동 해제
    zip_files = list(BIOASQ_DIR.rglob("*.zip"))
    if zip_files:
        print(f"[BIOASQ] {len(zip_files)}개의 ZIP 파일 발견")
        for z in zip_files:
            out_dir = z.with_suffix("")  # zip 이름과 같은 폴더
            out_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(out_dir)
                print(f"  ├─ 압축 해제: {z.name} → {out_dir.name}")
            except Exception as e:
                print(f"  ├─ 압축 해제 실패: {z.name} ({e})")
    else:
        print("[BIOASQ] ZIP 파일이 없습니다.")

    # 2) *_golden.json 수집 (없으면 *.json 전체로 fallback)
    json_files = list(BIOASQ_DIR.rglob("*_golden.json"))
    if not json_files:
        json_files = list(BIOASQ_DIR.rglob("*.json"))
    
    if not json_files:
        print(f"[BIOASQ] JSON 파일이 없음.")
        return

    print(f"[BIOASQ] {len(json_files)}개의 JSON 파일 발견")
    for jf in json_files:
        print(f"  ├─ {jf.relative_to(BIOASQ_DIR)}")

    # 3) JSONL 병합 (원본 필드 보존)
    out_dir = BIOASQ_DIR / "_processed"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bioasq_golden_all_raw.jsonl"

    total_questions = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for jf in sorted(json_files):
            try:
                text = jf.read_text(encoding="utf-8")
                data = json.loads(text)
                print(f"  ├─ 처리 중: {jf.name}")
            except Exception as e:
                print(f"  ├─ Parse 실패: {jf.name} ({e})")
                continue

            # 파일별 구조를 안전하게 처리
            def write_line(obj):
                nonlocal total_questions
                json.dump(obj, fout, ensure_ascii=False)
                fout.write("\n")
                total_questions += 1

            if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                # BioASQ 표준 포맷: {"questions": [...]}
                for x in data["questions"]:
                    write_line(x)
                print(f"    └─ {len(data['questions'])}개 질문 추가")
            elif isinstance(data, list):
                # 리스트 형태
                for x in data:
                    write_line(x)
                print(f"    └─ {len(data)}개 항목 추가")
            elif isinstance(data, dict):
                # 단일 객체
                write_line(data)
                print(f"    └─ 1개 항목 추가")
            else:
                print(f"    └─ 알 수 없는 구조 → 스킵")

    print(f"[BIOASQ] 병합 완료!")
    print(f"  ├─ 총 {total_questions}개 질문/항목")
    print(f"  └─ 출력: {out_path}")

    # 디렉토리 설정
    setup_directories()
    
    # BioASQ 처리
    result = handle_bioasq()
    
    if result:
        print()
        print(f"결과 파일: {result}")
    else:
        print()
        print("실패")

if __name__ == "__main__":
    main()