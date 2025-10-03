import os
import glob
import numpy as np


def load_audio_data(healthy_path, parkinson_path):
    """오디오 데이터 로드 및 라벨 생성"""
    print("파킨슨 병 음성 분류 앙상블 모델")
    print("=" * 50)
    
    # 오디오 파일 경로 수집 (WAV와 MP3 모두 지원)
    audio_extensions = ["*.wav", "*.mp3"]
    
    healthy_files = []
    parkinson_files = []
    
    # 건강한 사람 파일 수집
    for ext in audio_extensions:
        healthy_files.extend(glob.glob(os.path.join(healthy_path, ext)))
    
    # 파킨슨 환자 파일 수집
    for ext in audio_extensions:
        parkinson_files.extend(glob.glob(os.path.join(parkinson_path, ext)))
    
    # 라벨 생성 (HC=0, PD=1)
    audio_paths = healthy_files + parkinson_files
    labels = [0] * len(healthy_files) + [1] * len(parkinson_files)  # HC=0, PD=1
    
    print(f"건강한 사람 데이터: {len(healthy_files)}개")
    print(f"파킨슨 환자 데이터: {len(parkinson_files)}개")
    print(f"총 데이터: {len(audio_paths)}개")
    
    # 파일 형식별 통계 출력
    wav_count = sum(1 for path in audio_paths if path.lower().endswith('.wav'))
    mp3_count = sum(1 for path in audio_paths if path.lower().endswith('.mp3'))
    print(f"파일 형식별 통계: WAV {wav_count}개, MP3 {mp3_count}개")
    
    # 데이터를 numpy 배열로 변환
    X = np.array(audio_paths)
    y = np.array(labels)
    
    return X, y