"""
Transformer Cross-Attention 모델 예측 스크립트

예측 프로세스:
1. 학습된 CNN 모델로 특징 추출
2. MFCC 특징 추출
3. 학습된 Transformer 모델로 파킨슨병 예측
4. 결과 저장 및 출력
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import csv
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import CombinedFeatureExtractor
from model import TransformerCrossAttentionModel, SimpleTransformerModel


# ==================== 설정 ====================
# 예측 데이터 경로 (폴더, 파일 리스트, 또는 단일 파일)
PREDICT_DATA_PATH = "../data/testdata_KO/healthy"
# PREDICT_DATA_PATH = ["audio1.wav", "audio2.wav"]  # 파일 리스트
# PREDICT_DATA_PATH = "../data/audio.wav"  # 단일 파일

# CNN 모델 경로
CNN_MODEL_PATH = "../cnn/cnn_model.pth"

# Transformer 모델 경로
TRANSFORMER_MODEL_PATH = "./transformer_model.pth"

# 배치 크기
BATCH_SIZE = 8

# 장치 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =============================================


def load_model(model_path, device):
    """학습된 Transformer 모델 로드"""
    print("\n=== 모델 로딩 ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 모델 설정 로드
    model_type = checkpoint.get('model_type', 'cross_attention')
    config = checkpoint.get('model_config', {})

    print(f"모델 타입: {model_type}")
    print(f"모델 설정: {config}")

    # 모델 생성
    if model_type == 'cross_attention':
        model = TransformerCrossAttentionModel(
            cnn_feature_dim=2048,
            mfcc_feature_dim=12,
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.3),
            num_classes=2
        )
    elif model_type == 'simple':
        model = SimpleTransformerModel(
            cnn_feature_dim=2048,
            mfcc_feature_dim=12,
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.3),
            num_classes=2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("모델 로드 완료!")

    # 성능 정보 출력
    if 'accuracy' in checkpoint:
        print(f"저장된 모델의 Test Accuracy: {checkpoint['accuracy']:.4f}")
    if 'auc' in checkpoint:
        print(f"저장된 모델의 Test AUC: {checkpoint['auc']:.4f}")
    if 'f1_score' in checkpoint:
        print(f"저장된 모델의 Test F1 Score: {checkpoint['f1_score']:.4f}")

    return model


def parse_audio_paths(data_path):
    """데이터 경로 파싱"""
    audio_paths = []

    if isinstance(data_path, list):
        # 파일 리스트
        audio_paths = data_path
    elif os.path.isdir(data_path):
        # 폴더
        for ext in ["*.wav", "*.mp3"]:
            audio_paths.extend(glob.glob(os.path.join(data_path, ext)))
    elif os.path.isfile(data_path):
        # 단일 파일
        audio_paths = [data_path]
    else:
        raise ValueError(f"유효하지 않은 데이터 경로: {data_path}")

    # 파일 존재 확인
    valid_paths = [p for p in audio_paths if os.path.exists(p)]

    if not valid_paths:
        raise FileNotFoundError("유효한 오디오 파일이 없습니다.")

    print(f"\n총 {len(valid_paths)}개 파일")

    return valid_paths


def predict_batch(model, cnn_features, mfcc_features, device, batch_size):
    """배치 단위 예측"""
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(cnn_features), batch_size):
            end_idx = min(i + batch_size, len(cnn_features))

            batch_cnn = torch.FloatTensor(cnn_features[i:end_idx]).to(device)
            batch_mfcc = torch.FloatTensor(mfcc_features[i:end_idx]).to(device)

            outputs = model(batch_cnn, batch_mfcc)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_predictions.extend(preds)
            all_probs.extend(probs)

            del batch_cnn, batch_mfcc, outputs
            if device == 'cuda':
                torch.cuda.empty_cache()

    return np.array(all_predictions), np.array(all_probs)


def save_results(csv_data, valid_paths, predictions, probs):
    """예측 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_filename = f"transformer_prediction_{timestamp}.csv"
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print(f"\n✓ CSV 저장: {csv_filename}")
    except Exception as e:
        print(f"\n✗ CSV 저장 실패: {e}")

    # TXT 저장
    txt_filename = f"transformer_prediction_{timestamp}.txt"
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Transformer Cross-Attention 모델 예측 결과\n")
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


def predict():
    """메인 예측 함수"""
    print("\n" + "="*60)
    print("Transformer Cross-Attention 모델 예측 시작")
    print("="*60)

    # 모델 파일 확인
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"오류: CNN 모델 파일이 없습니다: {CNN_MODEL_PATH}")
        return

    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        print(f"오류: Transformer 모델 파일이 없습니다: {TRANSFORMER_MODEL_PATH}")
        print("먼저 train.py를 실행하여 모델을 학습하세요.")
        return

    # 오디오 경로 파싱
    try:
        audio_paths = parse_audio_paths(PREDICT_DATA_PATH)
    except Exception as e:
        print(f"오류: {e}")
        return

    # 특징 추출
    print("\n=== 특징 추출 ===")
    try:
        feature_extractor = CombinedFeatureExtractor(
            cnn_model_path=CNN_MODEL_PATH,
            n_mfcc=12,
            device=DEVICE
        )

        cnn_features, mfcc_features = feature_extractor.extract(audio_paths)

        print(f"CNN 특징: {cnn_features.shape}")
        print(f"MFCC 특징: {mfcc_features.shape}")

    except Exception as e:
        print(f"특징 추출 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    # 모델 로드
    try:
        model = load_model(TRANSFORMER_MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    # 예측 수행
    print("\n=== 예측 수행 ===")
    try:
        predictions, probs = predict_batch(
            model, cnn_features, mfcc_features, DEVICE, BATCH_SIZE
        )

        # 결과 출력
        print("\n" + "="*60)
        print("예측 결과")
        print("="*60)

        hc_count = 0
        pd_count = 0

        csv_data = [['파일명', '예측', 'HC 확률', 'PD 확률', '신뢰도']]

        for i, path in enumerate(audio_paths):
            filename = os.path.basename(path)
            pred_label = "PD" if predictions[i] == 1 else "HC"
            hc_prob = probs[i][0]
            pd_prob = probs[i][1]
            confidence = max(hc_prob, pd_prob)

            if pred_label == "PD":
                pd_count += 1
            else:
                hc_count += 1

            print(f"\n[{i+1}/{len(audio_paths)}] {filename}")
            print(f"  예측: {pred_label}")
            print(f"  신뢰도: {confidence:.1%}")
            print(f"  확률: HC={hc_prob:.1%}, PD={pd_prob:.1%}")

            csv_data.append([filename, pred_label, f"{hc_prob:.4f}",
                           f"{pd_prob:.4f}", f"{confidence:.4f}"])

        # 통계
        print("\n" + "="*60)
        print("=== 예측 통계 ===")
        total_files = len(audio_paths)
        print(f"총 파일 수: {total_files}개")
        if total_files > 0:
            print(f"HC 예측: {hc_count}개 ({hc_count/total_files*100:.1f}%)")
            print(f"PD 예측: {pd_count}개 ({pd_count/total_files*100:.1f}%)")

        # 결과 저장
        save_results(csv_data, audio_paths, predictions, probs)

    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n예측 완료!")


if __name__ == "__main__":
    predict()
