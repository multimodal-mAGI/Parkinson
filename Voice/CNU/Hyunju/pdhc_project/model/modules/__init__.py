"""
modules 패키지 초기화 스크립트
오디오/이미지 임베딩, 멀티모달 모델, 학습 및 평가 유틸리티 모듈을 포함
"""

from .dataset_utils import load_audio_dataset, load_mfcc_image_dataset
from .embedding_utils import extract_audio_embeddings, extract_mfcc_embeddings
from .fusion_model import MultiModalClassifier, CrossAttentionFusion
from .train_utils import train_model
from .eval_utils import evaluate_model

__all__ = [
    "load_audio_dataset", "load_mfcc_image_dataset",
    "extract_audio_embeddings", "extract_mfcc_embeddings",
    "MultiModalClassifier", "CrossAttentionFusion",
    "train_model", "evaluate_model",
]