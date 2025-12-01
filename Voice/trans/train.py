"""
Transformer Cross-Attention 모델 학습 스크립트

학습 프로세스:
1. CNN 모델에서 특징 추출
2. MFCC 특징 추출
3. Transformer Cross-Attention 모델 학습
4. 성능 평가 및 모델 저장
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import CombinedFeatureExtractor
from model import TransformerCrossAttentionModel, SimpleTransformerModel


# ==================== 설정 ====================
# 데이터 경로
HEALTHY_PATH = "../data/EN/healthy"
PARKINSON_PATH = "../data/EN/parkinson"

# CNN 모델 경로
CNN_MODEL_PATH = "../cnn/cnn_model.pth"

# 모델 저장 경로
MODEL_SAVE_PATH = "./transformer_model.pth"

# 학습 파라미터
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 20  # Early stopping patience

# 모델 하이퍼파라미터
MODEL_TYPE = 'cross_attention'  # 'cross_attention' 또는 'simple'
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.25

# 장치 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =============================================


def load_audio_paths(healthy_path, parkinson_path):
    """오디오 파일 경로 로드"""
    print("="*60)
    print("데이터 로딩")
    print("="*60)

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
    labels = [0] * len(healthy_files) + [1] * len(parkinson_files)

    print(f"건강한 사람: {len(healthy_files)}개")
    print(f"파킨슨 환자: {len(parkinson_files)}개")
    print(f"총 데이터: {len(audio_paths)}개")

    return np.array(audio_paths), np.array(labels)


def extract_features(audio_paths, feature_extractor):
    """오디오 경로들로부터 특징 추출"""
    print("\n=== 특징 추출 ===")
    cnn_features, mfcc_features = feature_extractor.extract(audio_paths)

    print(f"CNN 특징: {cnn_features.shape}")
    print(f"MFCC 특징: {mfcc_features.shape}")

    return cnn_features, mfcc_features


def create_model(model_type, device):
    """모델 생성"""
    print(f"\n=== 모델 생성: {model_type} ===")

    if model_type == 'cross_attention':
        model = TransformerCrossAttentionModel(
            cnn_feature_dim=2048,
            mfcc_feature_dim=12,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            num_classes=2
        )
    elif model_type == 'simple':
        model = SimpleTransformerModel(
            cnn_feature_dim=2048,
            mfcc_feature_dim=12,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            num_classes=2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")

    return model


def train_epoch(model, cnn_train, mfcc_train, labels_train, optimizer, criterion, device, batch_size):
    """한 epoch 학습"""
    model.train()
    epoch_loss = 0
    num_batches = 0

    dataset_size = len(labels_train)
    indices = np.random.permutation(dataset_size)

    for i in range(0, dataset_size, batch_size):
        end_idx = min(i + batch_size, dataset_size)
        batch_indices = indices[i:end_idx]

        # 배치 데이터 준비
        batch_cnn = torch.FloatTensor(cnn_train[batch_indices]).to(device)
        batch_mfcc = torch.FloatTensor(mfcc_train[batch_indices]).to(device)
        batch_labels = torch.LongTensor(labels_train[batch_indices]).to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(batch_cnn, batch_mfcc)
        loss = criterion(outputs, batch_labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        del batch_cnn, batch_mfcc, batch_labels, outputs, loss
        if device == 'cuda':
            torch.cuda.empty_cache()

    return epoch_loss / num_batches


def evaluate(model, cnn_data, mfcc_data, labels, device, batch_size):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_probs = []
    eval_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            end_idx = min(i + batch_size, len(labels))

            batch_cnn = torch.FloatTensor(cnn_data[i:end_idx]).to(device)
            batch_mfcc = torch.FloatTensor(mfcc_data[i:end_idx]).to(device)
            batch_labels = torch.LongTensor(labels[i:end_idx]).to(device)

            outputs = model(batch_cnn, batch_mfcc)
            loss = criterion(outputs, batch_labels)

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_predictions.extend(preds)
            all_probs.extend(probs)
            eval_loss += loss.item()
            num_batches += 1

            del batch_cnn, batch_mfcc, batch_labels, outputs
            if device == 'cuda':
                torch.cuda.empty_cache()

    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    avg_loss = eval_loss / num_batches

    return all_predictions, all_probs, avg_loss


def save_training_curves(train_losses, val_losses):
    """학습 곡선 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"transformer_training_curve_{timestamp}.png"

    try:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_losses) + 1)

        plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Transformer Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 최소 validation loss 표시
        min_val_loss = min(val_losses)
        min_epoch = val_losses.index(min_val_loss) + 1
        plt.axvline(x=min_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Best Model (Epoch {min_epoch})')
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n학습 곡선 저장 완료: {plot_filename}")
    except Exception as e:
        print(f"학습 곡선 저장 실패: {e}")


def train():
    """메인 학습 함수"""
    print("\n" + "="*60)
    print("Transformer Cross-Attention 모델 학습 시작")
    print("="*60)

    # 경로 검증
    if not os.path.exists(HEALTHY_PATH):
        print(f"오류: 건강한 사람 데이터 경로 없음: {HEALTHY_PATH}")
        return

    if not os.path.exists(PARKINSON_PATH):
        print(f"오류: 파킨슨 환자 데이터 경로 없음: {PARKINSON_PATH}")
        return

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"오류: CNN 모델 없음: {CNN_MODEL_PATH}")
        print("먼저 CNN 모델을 학습하세요.")
        return

    # 오디오 경로 로드
    audio_paths, labels = load_audio_paths(HEALTHY_PATH, PARKINSON_PATH)

    if len(audio_paths) == 0:
        print("데이터를 찾을 수 없습니다.")
        return

    # 데이터 분할
    print("\n=== 데이터 분할 ===")
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.125, random_state=42,
        stratify=train_val_labels
    )

    print(f"훈련: {len(train_paths)}개 (HC: {np.sum(train_labels == 0)}, PD: {np.sum(train_labels == 1)})")
    print(f"검증: {len(val_paths)}개 (HC: {np.sum(val_labels == 0)}, PD: {np.sum(val_labels == 1)})")
    print(f"테스트: {len(test_paths)}개 (HC: {np.sum(test_labels == 0)}, PD: {np.sum(test_labels == 1)})")

    # 특징 추출기 생성
    print(f"\n=== 특징 추출기 초기화 ===")
    print(f"CNN 모델 경로: {CNN_MODEL_PATH}")
    feature_extractor = CombinedFeatureExtractor(
        cnn_model_path=CNN_MODEL_PATH,
        n_mfcc=12,
        device=DEVICE
    )

    # 특징 추출
    cnn_train, mfcc_train = extract_features(train_paths, feature_extractor)
    cnn_val, mfcc_val = extract_features(val_paths, feature_extractor)
    cnn_test, mfcc_test = extract_features(test_paths, feature_extractor)

    # 모델 생성
    model = create_model(MODEL_TYPE, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 학습
    print(f"\n=== 모델 훈련 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}) ===")
    print(f"Device: {DEVICE}")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # 학습
        train_loss = train_epoch(
            model, cnn_train, mfcc_train, train_labels,
            optimizer, criterion, DEVICE, BATCH_SIZE
        )

        # 검증
        _, _, val_loss = evaluate(
            model, cnn_val, mfcc_val, val_labels, DEVICE, BATCH_SIZE
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate 조정
        scheduler.step(val_loss)

        # 로그 출력
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
            if (epoch + 1) % 5 != 0:
                print(f"  -> New best model (Val Loss: {best_val_loss:.4f})")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    print("\n훈련 완료!" if early_stopping_counter < PATIENCE else "\nEarly stopping으로 훈련 중단!")

    # 학습 곡선 저장
    save_training_curves(train_losses, val_losses)

    # 최적 모델 로드
    print("\n=== 성능 평가 (Test Set) ===")
    if best_model_state:
        print("최적 성능의 모델을 로드하여 평가합니다.")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: 최적 모델이 없습니다. 마지막 모델로 평가합니다.")
        best_model_state = model.state_dict()

    # 테스트 평가
    test_predictions, test_probs, _ = evaluate(
        model, cnn_test, mfcc_test, test_labels, DEVICE, BATCH_SIZE
    )

    # 메트릭 계산
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_predictions, average='weighted', zero_division=0)

    # AUC 계산
    auc = 0.0
    if len(np.unique(test_labels)) > 1:
        try:
            auc = roc_auc_score(test_labels, test_probs[:, 1])
        except ValueError:
            print("Warning: AUC 계산 중 오류 발생.")
            auc = 0.0

    cm = confusion_matrix(test_labels, test_predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 모델 저장
    print("\n=== 모델 저장 ===")
    try:
        torch.save({
            'model_state_dict': best_model_state,
            'model_type': MODEL_TYPE,
            'model_config': {
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS,
                'dim_feedforward': DIM_FEEDFORWARD,
                'dropout': DROPOUT
            },
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm
        }, MODEL_SAVE_PATH)
        print(f"저장 완료: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"저장 오류: {e}")

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)


if __name__ == "__main__":
    train()
