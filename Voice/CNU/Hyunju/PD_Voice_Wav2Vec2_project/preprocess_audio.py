# Analysis-grade 오디오 전처리 스크립트 (최소 전처리: DC 제거, HPF, 리샘플, 무음 트리밍)

import argparse, csv, os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== 설정 =====
TARGET_SR = 16000        # 목표 샘플링 레이트 (모노)
HP_HZ = 60               # 하이패스 컷오프 Hz (60–80 권장 범위에서 튜닝)
TRIM_TOP_DB = 35         # 앞/뒤 무음 트리밍 임계 (피크 대비 dB)
BITS = 'PCM_16'          # 저장 포맷 (연구 재현성 위해 그대로 둬도 무방)

# preprocess_audio.py 상단에 추가
import torch
import torchaudio
import torchaudio.transforms as T

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_file_gpu(in_path: Path, out_dir: Path, subfolder: str):
    try:
        # 1) GPU에서 로드 (torchaudio 사용)
        waveform, sample_rate = torchaudio.load(in_path)
        waveform = waveform.to(DEVICE)
        
        # 모노로 변환 (GPU에서)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 리샘플링 (GPU에서)
        if sample_rate != TARGET_SR:
            resampler = T.Resample(sample_rate, TARGET_SR).to(DEVICE)
            waveform = resampler(waveform)
        
        # CPU로 이동해서 나머지 처리
        y = waveform.squeeze().cpu().numpy()
        
        # 2) DC 제거
        y = remove_dc(y)

        # 3) 하이패스 필터 (scipy는 CPU에서)
        if HP_HZ and HP_HZ > 0:
            y = butter_filter(y, TARGET_SR, HP_HZ, 'highpass')

        # 4) 앞/뒤 무음 트리밍
        y = trim_silence(y, TARGET_SR, TRIM_TOP_DB)

        # 5) 저장
        out_subdir = out_dir / subfolder
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / (in_path.stem + ".wav")
        sf.write(out_path, y, TARGET_SR, subtype=BITS)

        dur = len(y) / TARGET_SR
        return (subfolder, str(in_path), str(out_path), dur, "ok", "")
    except Exception as e:
        return (subfolder, str(in_path), "", 0.0, "error", repr(e))

def remove_dc(x: np.ndarray) -> np.ndarray:
    """DC 오프셋 제거 (평균 0으로 정렬)"""
    m = float(np.mean(x)) if x.size else 0.0
    return x - m

def butter_filter(signal, sr, cutoff, btype, order=5):
    """Butterworth IIR + filtfilt(영위상)"""
    nyq = 0.5 * sr
    norm = min(cutoff / nyq, 0.999)  # 안정성 가드
    b, a = butter(order, norm, btype=btype)
    return filtfilt(b, a, signal)

def trim_silence(x, sr, top_db=35):
    """앞/뒤 저레벨 구간 트리밍 (librosa.effects.trim)"""
    xt, idx = librosa.effects.trim(x, top_db=top_db)
    # 전부 잘려나가면(극단 케이스) 원본 반환
    return xt if xt.size > 0 else x

def process_file(in_path: Path, out_dir: Path, subfolder: str):
    try:
        # 1) 로드 + 리샘플 + 모노
        y, _ = librosa.load(in_path, sr=TARGET_SR, mono=True)

        # 2) DC 제거
        y = remove_dc(y)

        # 3) 하이패스 필터 (저주파 험 제거)
        if HP_HZ and HP_HZ > 0:
            y = butter_filter(y, TARGET_SR, HP_HZ, 'highpass')

        # 4) 앞/뒤 무음 트리밍
        y = trim_silence(y, TARGET_SR, TRIM_TOP_DB)

        # 5) 저장 (16-bit PCM) - 하위 폴더 생성
        out_subdir = out_dir / subfolder
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / (in_path.stem + ".wav")
        sf.write(out_path, y, TARGET_SR, subtype=BITS)

        dur = len(y) / TARGET_SR
        return (subfolder, str(in_path), str(out_path), dur, "ok", "")
    except Exception as e:
        return (subfolder, str(in_path), "", 0.0, "error", repr(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for processing")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(in_dir.rglob("*.wav"))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for f in files:
            rel = f.relative_to(in_dir)
            subfolder = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            futures[ex.submit(process_file, f, out_dir, subfolder)] = (f, subfolder)
        for fut in tqdm(as_completed(futures), total=len(files)):
            rows.append(fut.result())

    # 폴더별로 결과 분리 및 저장
    result_by_folder = {}
    for row in rows:
        subfolder = row[0]
        result_by_folder.setdefault(subfolder, []).append(row[1:])

    for subfolder, sub_rows in result_by_folder.items():
        csv_path = out_dir / subfolder / f"report_{subfolder}.csv"
        (out_dir / subfolder).mkdir(parents=True, exist_ok=True)  # 폴더가 없을 경우 생성
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src", "dst", "duration_sec", "status", "error"])
            w.writerows(sub_rows)

if __name__ == "__main__":
    main()
