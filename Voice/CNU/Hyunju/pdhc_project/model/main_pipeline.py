"""
main_pipeline.py
멀티모달(오디오 + 이미지) 파킨슨병 분류 전체 실행 스크립트
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# === 사용자 정의 모듈 불러오기 ===
from modules.dataset_utils import load_audio_dataset, load_mfcc_image_dataset
from modules.embedding_utils import extract_audio_embeddings, extract_mfcc_embeddings
from modules.fusion_model import MultiModalClassifier
from modules.train_utils import train_model
from modules.eval_utils import evaluate_model

# === 기본 설정 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] DEVICE: {DEVICE}")

# === 경로 설정 ===
audio_root = "data/vowel_data_preprocessed"
img_root = "data/vowel_data_preprocessed_img"
SAVE_PATH = "model/best_multimodal_model.pth"

# === 모델 로드 ===
MODEL_PATH = "model/wav2vec2_finetuning_model/checkpoint-1690"
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
audio_model.eval()

# === 캐시된 임베딩 확인 ===
audio_feature_path = "model/wav2vec_features.npy"
img_feature_path = "model/mfcc_features.npy"
label_path = "model/wav2vec_labels.npy"

if all(os.path.exists(p) for p in [audio_feature_path, img_feature_path, label_path]):
    print("[INFO] 기존 npy 파일 존재. 임베딩 로드 중...")
    X_audio = np.load(audio_feature_path)
    X_img = np.load(img_feature_path)
    y = np.load(label_path)
else:
    print("[INFO] npy 파일이 없어 임베딩 새로 추출 중...")

    # 오디오 데이터 로드 및 임베딩
    audio_paths, audio_labels = load_audio_dataset(audio_root)
    X_audio, y_audio = extract_audio_embeddings(audio_paths, audio_labels, audio_model, processor, DEVICE)

    # 이미지 데이터 로드 및 임베딩
    img_paths, img_labels = load_mfcc_image_dataset(img_root)
    X_img, y_img = extract_mfcc_embeddings(img_paths, img_labels, DEVICE)

    assert np.array_equal(y_audio, y_img), "오디오와 이미지 라벨 불일치"
    np.save(audio_feature_path, X_audio)
    np.save(img_feature_path, X_img)
    np.save(label_path, y_audio)
    y = y_audio
    print("[INFO] 임베딩 및 라벨 저장 완료")

# === 데이터 분할 ===
Xa_tr, Xa_te, Xi_tr, Xi_te, y_tr, y_te = train_test_split(
    X_audio, X_img, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(Xa_tr, dtype=torch.float32),
                  torch.tensor(Xi_tr, dtype=torch.float32),
                  torch.tensor(y_tr, dtype=torch.long)),
    batch_size=8, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(Xa_te, dtype=torch.float32),
                  torch.tensor(Xi_te, dtype=torch.float32),
                  torch.tensor(y_te, dtype=torch.long)),
    batch_size=8, shuffle=False
)

# === 모델 학습 ===
print("[INFO] LLM 기반 멀티모달 분류기 학습 시작...")
model = MultiModalClassifier().to(DEVICE)
best_auc = train_model(model, train_loader, test_loader, DEVICE, SAVE_PATH, epochs=10)

# === 외부 데이터 평가 ===
print("[INFO] 외부 데이터셋 평가 시작...")

ext_audio_root = "data/italian_voice_pdhc_split"
ext_img_root = "data/italian_voice_pdhc_split_img"

ext_audio_paths, ext_audio_labels = load_audio_dataset(ext_audio_root)
ext_img_paths, ext_img_labels = load_mfcc_image_dataset(ext_img_root)

print(f"[INFO] 외부 데이터 로드 완료: 오디오 {len(ext_audio_paths)}개, 이미지 {len(ext_img_paths)}개")

print("[INFO] 외부 데이터 임베딩 추출 중...")
X_audio_ext, y_audio_ext = extract_audio_embeddings(ext_audio_paths, ext_audio_labels, audio_model, processor, DEVICE)
X_img_ext, y_img_ext = extract_mfcc_embeddings(ext_img_paths, ext_img_labels, DEVICE)
print("[INFO] 임베딩 추출 완료")

assert np.array_equal(y_audio_ext, y_img_ext)

ext_loader = DataLoader(
    TensorDataset(torch.tensor(X_audio_ext, dtype=torch.float32),
                  torch.tensor(X_img_ext, dtype=torch.float32),
                  torch.tensor(y_audio_ext, dtype=torch.long)),
    batch_size=8, shuffle=False
)

print("[INFO] 저장된 모델 로드 및 평가 시작...")
state_dict = torch.load(SAVE_PATH, weights_only=True)
model.load_state_dict(state_dict)
evaluate_model(model, ext_loader, DEVICE)

