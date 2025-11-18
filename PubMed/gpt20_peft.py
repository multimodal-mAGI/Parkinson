# CoT 기반 커리큘럼 학습

from gpt20_cot_answer_focus_model import load_tokenizer, load_lora_model
from gpt20_cot_answer_focus_train_logic import CurriculumSampler, TARGET_WITH_LABELED, DATASET_PATHS, train_with_curriculum
from gpt20_cot_answer_focus_train import train_answer_focused
from torch.optim import AdamW
import torch

MODEL_NAME = "openai/gpt-oss-20b"
BATCH_SIZE = 1
NUM_EPOCHS = 5
LR = 1e-5

# LoRA 설정
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# CoT 사용 여부
USE_COT = True  # False로 설정하면 기존 방식

# 1. 설정 출력
print("=" * 30)
print("CONFIGURATION (CoT-based)")
print("=" * 30)
print(f"Model: {MODEL_NAME}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LR}")
print(f"LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"CoT Enabled: {USE_COT}")
print("=" * 30)

# 2. 토크나이저/모델 로딩
print("\nLoading tokenizer and model...")
tokenizer = load_tokenizer(MODEL_NAME)
model = load_lora_model(
    MODEL_NAME, 
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
)

# 3. 옵티마이저 준비
print("\nCreating optimizer...")
trainable_params = [p for p in model.parameters() if p.requires_grad]
if not trainable_params:
    raise RuntimeError("No trainable parameters found for optimizer!")

print(f"Found {len(trainable_params)} parameter groups for optimizer")
optimizer = AdamW(trainable_params, lr=LR)

# 4. 커리큘럼 샘플러 (CoT 적용)
print("\nInitializing curriculum sampler with CoT...")
sampler = CurriculumSampler(
    DATASET_PATHS, 
    TARGET_WITH_LABELED, 
    batch_size=BATCH_SIZE,
    use_cot=USE_COT  # CoT 활성화
)


# 5. 학습 실행
SAVE_DIR = "./cp_gptoss20b_cot3"
PUBMEDQA_DEV_PATH = "data/processed/pubmedqa_dev_cv.jsonl"

print("\n" + "="*50)
print("STARTING COT-BASED TRAINING")
print("="*50)

# train_with_curriculum 대신
train_answer_focused(
    model, optimizer, sampler,
    tokenizer=tokenizer,
    num_epochs=NUM_EPOCHS,
    save_dir=SAVE_DIR,
    eval_path=PUBMEDQA_DEV_PATH,
    use_reasoning=False,  # CoT 없이 시작
    class_weights=True    # 클래스 불균형 해결
)

print("\n" + "="*30)
print("TRAINING COMPLETED!")
print(f"Best model saved in: {SAVE_DIR}/best_model")
print("="*30)