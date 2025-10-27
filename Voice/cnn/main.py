import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import glob
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# 현재 디렉터리를 패스에 추가
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import CNNAudioPreprocessor
    from model import CNNModel
except ImportError:
    print("Warning: 'preprocessing.py' 또는 'model.py'를 찾을 수 없습니다.")
    # 임시 클래스 정의 (코드 실행을 위해)
    class CNNAudioPreprocessor:
        def load_and_preprocess_audio(self, paths):
            print(f"[임시] {len(paths)}개 파일 전처리 중...")
            # 실제 전처리 로직 대신 더미 데이터 반환
            # (실제 크기(예: [128, 128])에 맞게 조정 필요)
            return np.random.rand(len(paths), 128, 128) 

    class CNNModel(nn.Module):
        def __init__(self, num_classes=2):
            super(CNNModel, self).__init__()
            # (실제 모델 구조와 입력 크기가 맞아야 함)
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # 입력 크기가 [128, 128]이라고 가정
            self.fc1 = nn.Linear(16 * 64 * 64, 128) 
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = x.view(x.size(0), -1) # Flatten
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x


# 학습용 데이터 경로
HEALTHY_PATH = "../data/EN/healthy"
PARKINSON_PATH = "../data/EN/parkinson"

# 예측용 데이터 경로
PREDICT_DATA_PATH = "../data/testdata_KO/healthy"
# PREDICT_DATA_PATH = ["audio1.wav", "audio2.wav"]  # 파일 리스트
# PREDICT_DATA_PATH = "../data/audio.wav"  # 단일 파일

# 모델 저장 경로
MODEL_SAVE_PATH = "./cnn_model.pth"

# 훈련 파라미터
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 실행 모드: 'train' 또는 'predict'
MODE = 'train'



def setup_device():
    """GPU 설정"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    return device


def load_audio_data(healthy_path, parkinson_path):
    """오디오 데이터 경로 수집"""
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

    X = np.array(audio_paths)
    y = np.array(labels)

    return X, y


def train():
    """CNN 모델 학습"""
    print("\n" + "="*60)
    print("CNN 모델 학습 시작 (노이즈 제거 강화)")
    print("="*60)

    # --- [추가] Early Stopping 파라미터 ---
    PATIENCE = 40  # N 에포크 동안 성능 향상이 없으면 중지
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    # ------------------------------------

    # 경로 검증
    if not os.path.exists(HEALTHY_PATH):
        print(f"오류: 건강한 사람 데이터 경로 없음: {HEALTHY_PATH}")
        return

    if not os.path.exists(PARKINSON_PATH):
        print(f"오류: 파킨슨 환자 데이터 경로 없음: {PARKINSON_PATH}")
        return

    # 데이터 로드
    X, y = load_audio_data(HEALTHY_PATH, PARKINSON_PATH)

    if len(X) == 0:
        print("데이터를 찾을 수 없습니다.")
        return

    # 장치 설정
    device = setup_device()

    # 데이터 분할
    print("\n=== 데이터 분할 ===")
    
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.125, random_state=42, stratify=train_val_labels
    )
    # ---------------------------------------------------------

    print(f"훈련: {len(train_paths)}개 (HC: {np.sum(train_labels == 0)}, PD: {np.sum(train_labels == 1)})")
    # --- [추가] Validation set 출력 ---
    print(f"검증: {len(val_paths)}개 (HC: {np.sum(val_labels == 0)}, PD: {np.sum(val_labels == 1)})")
    # -----------------------------------
    print(f"테스트: {len(test_paths)}개 (HC: {np.sum(test_labels == 0)}, PD: {np.sum(test_labels == 1)})")

    # 전처리
    print("\n=== 전처리 (Wiener + Hamming + Wavelet) ===")
    preprocessor = CNNAudioPreprocessor()

    train_data = preprocessor.load_and_preprocess_audio(train_paths)
    test_data = preprocessor.load_and_preprocess_audio(test_paths)
    # --- [추가] Validation 데이터 전처리 ---
    val_data = preprocessor.load_and_preprocess_audio(val_paths)
    # -------------------------------------

    # 모델 생성
    model = CNNModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 학습
    print(f"\n=== 모델 훈련 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}) ===")
    
    dataset_size = len(train_labels)

    for epoch in range(EPOCHS):
        model.train() # 훈련 모드
        epoch_loss = 0
        num_batches = 0

        # --- 훈련 단계 ---
        for i in range(0, dataset_size, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, dataset_size)

            # 배치 데이터 준비
            batch_data = torch.FloatTensor(train_data[i:end_idx]).unsqueeze(1).to(device)
            batch_labels = torch.LongTensor(train_labels[i:end_idx]).to(device)

            # 순전파
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            del batch_data, batch_labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        # --- [추가] Validation 단계 ---
        model.eval()  # 평가 모드
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, len(val_data))
                batch_data = torch.FloatTensor(val_data[i:end_idx]).unsqueeze(1).to(device)
                batch_labels = torch.LongTensor(val_labels[i:end_idx]).to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                num_val_batches += 1

                del batch_data, batch_labels, outputs, loss
                torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / (num_val_batches if num_val_batches > 0 else 1)
        # --------------------------------

        # [수정] 출력문 (Train Loss와 Val Loss 함께)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- [추가] Early Stopping 로직 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # 최고 성능 모델의 상태 저장
            best_model_state = model.state_dict() 
            if (epoch + 1) % 5 != 0: # 5 epoch 단위 출력이 아닐 때만 갱신 메시지 출력
                 print(f"  -> New best model (Val Loss: {best_val_loss:.4f})")
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break  # 학습 루프 탈출
        # ----------------------------------

    # [수정] 훈련 완료 메시지
    print("\n훈련 완료!" if early_stopping_counter < PATIENCE else "\nEarly stopping으로 훈련 중단!")


    # --- [수정] 평가 섹션 (기존 코드 오류 수정) ---
    print("\n=== 성능 평가 ===")
    
    # Validation에서 가장 좋았던 모델 상태로 복원
    if best_model_state:
        print("최적 성능의 모델(Best Model)을 로드하여 평가합니다.")
        model.load_state_dict(best_model_state)
    else:
        # (Early stopping이 발생하지 않았거나, 1 epoch만 돌린 경우)
        print("Warning: 최적 모델이 없습니다. 마지막 모델로 평가합니다.")
        best_model_state = model.state_dict() # 마지막 모델 상태 저장

    model.eval()
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        # Test Set (test_data)으로 평가
        for i in range(0, len(test_data), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(test_data))
            batch_data = torch.FloatTensor(test_data[i:end_idx]).unsqueeze(1).to(device)

            outputs = model(batch_data)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # 예측 결과 및 확률 저장
            all_predictions.extend(preds)
            all_probs.extend(probs)

            del batch_data, outputs, probs, preds
            torch.cuda.empty_cache()

    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    # --------------------------------------------

    # 메트릭 계산
    accuracy = accuracy_score(test_labels, all_predictions)
    precision = precision_score(test_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, all_predictions, average='weighted', zero_division=0)
    
    # --- [수정] AUC 계산 (Test Set에 2개 클래스가 모두 있어야 함) ---
    auc = 0.0
    if len(np.unique(test_labels)) > 1:
        try:
            auc = roc_auc_score(test_labels, all_probs[:, 1])
        except ValueError:
            print("Warning: AUC 계산 중 오류 발생. (단일 클래스만 존재)")
            auc = 0.0
    else:
        print("Warning: Test Set에 클래스가 하나뿐이어서 AUC를 계산할 수 없습니다.")
    # --------------------------------------------------------

    cm = confusion_matrix(test_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 모델 저장
    print("\n=== 모델 저장 ===")
    try:
        torch.save({
            # --- [수정] 최고 성능 모델(best_model_state) 저장 ---
            'model_state_dict': best_model_state,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }, MODEL_SAVE_PATH)
        print(f"저장 완료: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"저장 오류: {e}")

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)


def predict():
    """CNN 모델 예측"""
    print("\n" + "="*60)
    print("CNN 모델 예측 시작")
    print("="*60)

    # 모델 존재 확인
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"오류: 모델 파일이 없습니다: {MODEL_SAVE_PATH}")
        print("먼저 train 모드를 실행하세요.")
        return

    # 장치 설정
    device = setup_device()

    # 모델 로드
    print("\n=== 모델 로딩 ===")
    model = CNNModel(num_classes=2).to(device)

    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("모델 로드 완료!")

        if 'accuracy' in checkpoint:
            print(f"저장된 모델의 Test Accuracy: {checkpoint['accuracy']:.4f}")
        if 'auc' in checkpoint:
            print(f"저장된 모델의 Test AUC: {checkpoint['auc']:.4f}")
            
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return

    # 데이터 경로 파싱
    audio_paths = []

    if isinstance(PREDICT_DATA_PATH, list):
        audio_paths = PREDICT_DATA_PATH
    elif os.path.isdir(PREDICT_DATA_PATH):
        for ext in ["*.wav", "*.mp3"]:
            audio_paths.extend(glob.glob(os.path.join(PREDICT_DATA_PATH, ext)))
    elif os.path.isfile(PREDICT_DATA_PATH):
        audio_paths = [PREDICT_DATA_PATH]
    else:
        print(f"오류: 유효하지 않은 데이터 경로: {PREDICT_DATA_PATH}")
        return

    # 파일 존재 확인
    valid_paths = [p for p in audio_paths if os.path.exists(p)]

    if not valid_paths:
        print("유효한 오디오 파일이 없습니다.")
        return

    print(f"\n총 {len(valid_paths)}개 파일")

    # 전처리
    print("\n=== 전처리 (Wiener + Hamming + Wavelet) ===")
    preprocessor = CNNAudioPreprocessor()
    processed_data = preprocessor.load_and_preprocess_audio(valid_paths)

    # 예측 수행
    print("\n=== 예측 수행 ===")
    try:
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(processed_data), BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, len(processed_data))
                batch_data = torch.FloatTensor(processed_data[i:end_idx]).unsqueeze(1).to(device)

                outputs = model(batch_data)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_predictions.extend(preds)
                all_probs.extend(probs)

                del batch_data, outputs
                torch.cuda.empty_cache()

        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        # 결과 출력
        print("\n" + "="*60)
        print("예측 결과")
        print("="*60)

        hc_count = 0
        pd_count = 0

        csv_data = [['파일명', '예측', 'HC 확률', 'PD 확률', '신뢰도']]

        for i, path in enumerate(valid_paths):
            filename = os.path.basename(path)
            pred_label = "PD" if all_predictions[i] == 1 else "HC"
            hc_prob = all_probs[i][0]
            pd_prob = all_probs[i][1]
            confidence = max(hc_prob, pd_prob)

            if pred_label == "PD":
                pd_count += 1
            else:
                hc_count += 1

            print(f"\n[{i+1}/{len(valid_paths)}] {filename}")
            print(f"  예측: {pred_label}")
            print(f"  신뢰도: {confidence:.1%}")
            print(f"  확률: HC={hc_prob:.1%}, PD={pd_prob:.1%}")

            csv_data.append([filename, pred_label, f"{hc_prob:.4f}", f"{pd_prob:.4f}", f"{confidence:.4f}"])

        # 통계
        print("\n" + "="*60)
        print("=== 예측 통계 ===")
        total_files = len(valid_paths)
        print(f"총 파일 수: {total_files}개")
        if total_files > 0:
            print(f"HC 예측: {hc_count}개 ({hc_count/total_files*100:.1f}%)")
            print(f"PD 예측: {pd_count}개 ({pd_count/total_files*100:.1f}%)")

        # 결과 저장
        save_results(csv_data, valid_paths, all_predictions, all_probs)

    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n예측 완료!")


def save_results(csv_data, valid_paths, predictions, probs):
    """예측 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_filename = f"cnn_prediction_{timestamp}.csv"
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print(f"\n✓ CSV 저장: {csv_filename}")
    except Exception as e:
        print(f"\n✗ CSV 저장 실패: {e}")

    # TXT 저장
    txt_filename = f"cnn_prediction_{timestamp}.txt"
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CNN 모델 예측 결과 (노이즈 제거 강화)\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            hc_count = 0
            pd_count = 0
            total_files = len(valid_paths)

            for i, path in enumerate(valid_paths):
                filename = os.path.basename(path)
                pred_label = "PD" if predictions[i] == 1 else "HC"
                hc_prob = probs[i][0]
                pd_prob = probs[i][1]
                confidence = max(hc_prob, pd_prob)

                if pred_label == "PD":
                    pd_count += 1
                else:
                    hc_count += 1

                f.write(f"[{i+1}/{total_files}] {filename}\n")
                f.write(f"  예측: {pred_label}\n")
                f.write(f"  신뢰도: {confidence:.1%}\n")
                f.write(f"  확률: HC={hc_prob:.1%}, PD={pd_prob:.1%}\n\n")

            f.write("="*60 + "\n")
            f.write("예측 통계\n")
            f.write("="*60 + "\n")
            f.write(f"총 파일 수: {total_files}개\n")
            if total_files > 0:
                f.write(f"HC 예측: {hc_count}개 ({hc_count/total_files*100:.1f}%)\n")
                f.write(f"PD 예측: {pd_count}개 ({pd_count/total_files*100:.1f}%)\n")

        print(f"✓ TXT 저장: {txt_filename}")
    except Exception as e:
        print(f"✗ TXT 저장 실패: {e}")


if __name__ == "__main__":
    if MODE == 'train':
        train()
    elif MODE == 'predict':
        predict()
    else:
        print(f"오류: MODE는 'train' 또는 'predict'여야 합니다. 현재: {MODE}")