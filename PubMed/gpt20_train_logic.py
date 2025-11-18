# gpt20_cot_answer_focus_train_logic
"""
CoT(Chain-of-Thought)를 적용한 커리큘럼 학습 로직
간소화 버전 CoT 포맷 사용:
Key Evidence -> Synthesis -> Answer
"""

import json
import random
import numpy as np
import torch
import csv
import os
from typing import List, Dict, Any

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def generate_cot_reasoning(question: str, abstract: str, answer: str) -> str:
    # 입력 검증 및 타입 변환
    if not isinstance(question, str):
        question = str(question) if question else ""
    if not isinstance(abstract, str):
        abstract = str(abstract) if abstract else ""
    if not isinstance(answer, str):
        answer = str(answer) if answer else ""
    
    # Abstract 처리 (안전한 슬라이싱)
    key_finding = abstract[:150] if len(abstract) > 0 else f"Question focuses on: {question[:100]}"
    
    # Study design 추정
    study_design = "Clinical study"
    if abstract:
        abstract_lower = abstract.lower()
        if any(word in abstract_lower for word in ['randomized', 'rct', 'trial']):
            study_design = "Randomized controlled trial"
        elif any(word in abstract_lower for word in ['cohort', 'prospective']):
            study_design = "Cohort study"
        elif any(word in abstract_lower for word in ['meta-analysis', 'systematic review']):
            study_design = "Systematic review/Meta-analysis"
        elif any(word in abstract_lower for word in ['case', 'retrospective']):
            study_design = "Case study/Retrospective analysis"
    
    # 통계적 유의성
    statistical_evidence = "Statistical significance not specified"
    if abstract:
        if any(word in abstract.lower() for word in ['p<0.05', 'p < 0.05', 'significant', 'p<0.01']):
            statistical_evidence = "Statistically significant results reported"
        elif any(word in abstract.lower() for word in ['not significant', 'ns', 'p>0.05']):
            statistical_evidence = "No statistical significance found"
    
    # Answer에 따른 connection (소문자 변환)
    answer_lower = answer.lower()
    connection_map = {
        "yes": "The evidence supports a positive relationship or effect.",
        "no": "The evidence does not support the relationship or shows negative results.",
        "maybe": "The evidence is inconclusive or shows mixed results.",
        "supports": "The evidence supports the claim.",
        "refutes": "The evidence refutes the claim.",
        "nei": "Not enough information to determine."
    }
    connection = connection_map.get(answer_lower, "Based on the evidence, this answer is supported.")
    
    cot = f"""Reasoning:
1. Key finding: {key_finding}
2. Study design: {study_design}
3. Statistical evidence: {statistical_evidence}
4. Connection: {connection}

Final Answer: {answer}"""
    
    return cot




def add_cot_to_sample(sample: dict) -> dict:
    """
    개별 샘플에 CoT 추가

    Args:
        sample: 원본 데이터 샘플
    Returns:
        CoT가 추가된 샘플
    """
    # Question 추출
    question = sample.get("question", sample.get("claim", sample.get("QUESTION", "")))
    
    # Abstract 추출 (다양한 포맷 지원)
    abstract = sample.get("abstract", sample.get("context", sample.get("contexts", "")))
    
    # abstract가 리스트면 문자열로 합치기
    if isinstance(abstract, list):
        abstract = " ".join(str(x) for x in abstract if x)
    elif isinstance(abstract, dict):
        abstract = " ".join(abstract.get("contexts", abstract.get("context", [])))
    
    if not isinstance(abstract, str):
        abstract = ""
    
    # Answer 추출 (리스트 처리 추가)
    answer = sample.get("answer", sample.get("final_decision", sample.get("label", "")))
    
    # answer가 리스트면 첫 번째 요소 사용
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    
    # 문자열로 변환
    if not isinstance(answer, str):
        answer = str(answer) if answer else ""
    
    # yes/no/maybe로 정규화 (소문자 변환)
    if isinstance(answer, str):
        answer = answer.lower().strip()
    
    # CoT 생성
    reasoning = generate_cot_reasoning(question, abstract, answer)
    
    # 원본 샘플에 CoT 추가
    sample_with_cot = sample.copy()
    sample_with_cot["cot_reasoning"] = reasoning
    
    # 표준 필드로 정규화
    sample_with_cot["question"] = question
    sample_with_cot["abstract"] = abstract
    sample_with_cot["answer"] = answer
    
    return sample_with_cot





# PubMedQA 평가 함수
def evaluate_on_pubmedqa(model, tokenizer, pubmedqa_path, max_samples=1000):
    """
    PubMedQA 평가 (CoT 생성 없이 answer만 예측)
    학습은 CoT로, 평가는 direct answer prediction
    """
    import json
    import torch
    import os

    model.eval()
    
    file_ext = os.path.splitext(pubmedqa_path)[1].lower()
    
    try:
        if file_ext == '.jsonl':
            print(f"Reading JSONL file: {pubmedqa_path}")
            data = []
            with open(pubmedqa_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            print(f"Reading JSON file: {pubmedqa_path}")
            with open(pubmedqa_path, 'r') as f:
                data = json.load(f)
                
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
            
    except Exception as e:
        print(f"Error reading file {pubmedqa_path}: {e}")
        return 0.0
    
    questions, answers = [], []
    for ex in data[:max_samples]:
        q = ex.get("question", ex.get("claim", ""))
        a = ex.get("final_decision", ex.get("answer", None))
        
        if q and a in {"yes", "no", "maybe"}:
            questions.append(q)
            answers.append(a)

    if not questions:
        print("No valid questions found!")
        return 0.0

    print(f"Evaluating {len(questions)} questions...")
    
    verbalizers = ["yes", "no", "maybe"]
    try:
        verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in verbalizers]
    except:
        verbalizer_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)[0]) for v in verbalizers]
    
    correct = 0

    for i, (q, gt) in enumerate(zip(questions, answers)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(questions)} samples...")
            
        inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
        
        scores = [logits[vid].item() for vid in verbalizer_ids]
        pred_idx = int(torch.tensor(scores).argmax())
        pred = verbalizers[pred_idx]
        
        if gt == pred:
            correct += 1

    acc = correct / len(questions) if questions else 0.0
    print(f"[PubMedQA] Accuracy: {acc*100:.2f}% ({correct}/{len(questions)})")
    model.train()
    return acc


def evaluate_on_validation(model, tokenizer, val_path, max_samples=1000):
    import json
    import torch
    import os
    
    model.eval()
    
    file_ext = os.path.splitext(val_path)[1].lower()
    
    try:
        if file_ext == '.jsonl':
            print(f"Reading JSONL file: {val_path}")
            data = []
            with open(val_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            print(f"Reading JSON file: {val_path}")
            with open(val_path, 'r') as f:
                data = json.load(f)
                
        data = data[:max_samples]
    except Exception as e:
        print(f"Error reading validation file {val_path}: {e}")
        return 0.0, 0.0
    
    questions, answers = [], []
    for ex in data:
        q = ex.get("claim", ex.get("question", ""))
        a = ex.get("answer", None)
        if q and a is not None:
            questions.append(q)
            answers.append(a)

    if not questions:
        print("No valid validation questions found!")
        return 0.0, 0.0

    print(f"Validating {len(questions)} questions...")

    verbalizers = ["yes", "no", "maybe"]
    try:
        verbalizer_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in verbalizers]
    except:
        verbalizer_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)[0]) for v in verbalizers]
    
    tokenizer.padding_side = "left"
    correct = 0
    total_loss = 0.0
    
    for i, (q, gt) in enumerate(zip(questions, answers)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(questions)} samples...")
            
        inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            logits = outputs.logits[0, -1]
            loss = outputs.loss.item() if hasattr(outputs, 'loss') and outputs.loss is not None else 0.0
        
        total_loss += loss
        scores = [logits[vid].item() for vid in verbalizer_ids]
        pred_idx = int(torch.tensor(scores).argmax())
        pred = verbalizers[pred_idx]
        
        if gt == pred:
            correct += 1
            
    acc = correct / len(questions) if questions else 0.0
    avg_loss = total_loss / len(questions) if questions else 0.0
    print(f"[Validation] Accuracy: {acc*100:.2f}% ({correct}/{len(questions)}) | Avg Loss: {avg_loss:.4f}")
    model.train()
    return acc, avg_loss


def train_with_curriculum(
    model, optimizer, sampler, tokenizer=None, num_epochs=5, 
    save_dir="./checkpoints", eval_path=None, val_path=None
):
    """
    CoT 기반 커리큘럼 학습
    """
    import csv
    import os

    model_name = getattr(model, 'name_or_path', None)
    if model_name is None and hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
        model_name = model.config.name_or_path
    if model_name is None:
        model_name = 'model'
    model_name = os.path.basename(model_name).replace('/', '_')

    log_path = os.path.join(save_dir, f"{model_name}_cot_log.csv")
    log_fields = ["epoch", "train_loss", "pubmedqa_acc", "val_acc", "val_loss", "is_best", "epoch_samples"]
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs} start (CoT-based)")
        print(f"Best so far: {best_acc*100:.2f}% (epoch {best_epoch})")
        print(f"{'='*50}")
        
        batches = sampler.get_epoch_batches(epoch)
        epoch_loss = 0.0
        total = 0
        
        print(f"Total batches for epoch {epoch}: {len(batches)}")

        for batch_idx, batch in enumerate(batches):
            model.train()
            
            # CoT 포맷으로 변환 (안전한 필드 추출)
            texts = []
            for ex in batch:
                question = ex.get("question", "")
                answer = ex.get("answer", "")
                abstract = ex.get("abstract", "")
                
                if "cot_reasoning" in ex:
                    cot = ex["cot_reasoning"]
                else:
                    cot = generate_cot_reasoning(question, abstract, answer)
                
                # 학습 텍스트 구성
                text = f"Question: {question}\nContext: {abstract[:200] if abstract else 'N/A'}\n\n{cot}"
                texts.append(text)
            
            # tokenize
            tokenizer.padding_side = "left"
            enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=768).to(model.device)
            
            optimizer.zero_grad()
            out = model(**enc, labels=enc['input_ids'])
            loss = out.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += float(loss.detach().cpu())
            total += 1
            
            if batch_idx % 200 == 0:
                current_loss = epoch_loss / max(1, batch_idx + 1)
                print(f"  Batch {batch_idx}/{len(batches)} | Current Avg Loss: {current_loss:.4f} | Last Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / total if total else 0.0
        print(f"\nEpoch {epoch} finished | Final Avg Loss: {avg_loss:.4f}")

        # 모델 저장
        save_path = os.path.join(save_dir, f"epoch{epoch}")
        model.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

        # 평가
        pubmedqa_acc = None
        val_acc = None
        val_loss = None
        is_best = False

        if eval_path and tokenizer is not None:
            print(f"\n[Eval] PubMedQA on {eval_path}")
            pubmedqa_acc = evaluate_on_pubmedqa(model, tokenizer, eval_path, max_samples=500)
            
            if pubmedqa_acc > best_acc:
                best_acc = pubmedqa_acc
                best_epoch = epoch
                is_best = True
                
                best_save_path = os.path.join(save_dir, "best_model")
                model.save_pretrained(best_save_path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(best_save_path)
                print(f"NEW BEST! Accuracy: {best_acc*100:.2f}% - Saved to {best_save_path}")
            else:
                print(f"Current: {pubmedqa_acc*100:.2f}%, Best: {best_acc*100:.2f}%")

        if val_path and tokenizer is not None:
            print(f"\n[Validation] on {val_path}")
            val_acc, val_loss = evaluate_on_validation(model, tokenizer, val_path, max_samples=300)

        # 로그 저장
        current_samples = sampler.get_epoch_samples_count(epoch)
        log_row = {
            "epoch": epoch, 
            "train_loss": avg_loss, 
            "pubmedqa_acc": pubmedqa_acc, 
            "val_acc": val_acc, 
            "val_loss": val_loss,
            "is_best": is_best,
            "epoch_samples": current_samples
        }
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)
        print(f"[Log] Saved epoch {epoch} metrics to {log_path}")

    print(f"\nTraining completed!")
    print(f"Final best accuracy: {best_acc*100:.2f}% (epoch {best_epoch})")
    return best_acc, best_epoch


# 클래스 균형 샘플링
def balanced_sampling(data, target_size, balance_key='answer'):

    # 유효한 답변만 필터링
    valid_data = [
        ex for ex in data 
        if ex.get(balance_key, "").lower() in {'yes', 'no', 'maybe'}
    ]
    
    if len(valid_data) < target_size * 0.5:
        return []

    from collections import defaultdict
    import random
    
    # 클래스별로 분리
    class_data = defaultdict(list)
    other_data = []  # yes/no/maybe가 아닌 데이터
    
    for sample in data:
        label = sample.get(balance_key)
        if label in {'yes', 'no', 'maybe'}:
            class_data[label].append(sample)
        else:
            other_data.append(sample)
    
    # yes/no/maybe 데이터가 없으면 그냥 랜덤 샘플링
    if not class_data:
        print(f"    No yes/no/maybe answers found, using random sampling")
        if len(data) >= target_size:
            return random.sample(data, target_size)
        else:
            return data + random.choices(data, k=target_size - len(data))
    
    # 클래스별 샘플 수 계산
    n_classes = len(class_data)
    samples_per_class = target_size // n_classes
    
    balanced_data = []
    for label in ['yes', 'no', 'maybe']:
        if label not in class_data:
            continue
            
        samples = class_data[label]
        if len(samples) >= samples_per_class:
            balanced_data.extend(random.sample(samples, samples_per_class))
        else:
            # 부족하면 중복 샘플링
            balanced_data.extend(samples)
            balanced_data.extend(random.choices(samples, k=samples_per_class - len(samples)))
    
    # 목표 크기에 못 미치면 other_data에서 보충
    if len(balanced_data) < target_size and other_data:
        remaining = target_size - len(balanced_data)
        if len(other_data) >= remaining:
            balanced_data.extend(random.sample(other_data, remaining))
        else:
            balanced_data.extend(other_data)
            balanced_data.extend(random.choices(other_data, k=remaining - len(other_data)))
    
    random.shuffle(balanced_data)
    return balanced_data[:target_size]  # 정확한 크기 보장




class CurriculumSampler:
    """
    CoT를 지원하는 커리큘럼 샘플러
    """
    def __init__(self, dataset_paths: Dict[str, Any], curriculum_schedule: List[Dict], batch_size: int = 8, use_cot: bool = True):
        self.dataset_paths = dataset_paths
        self.curriculum_schedule = curriculum_schedule
        self.batch_size = batch_size
        self.use_cot = use_cot
        self.datasets = self._load_all_datasets()
        self._print_dataset_stats()

    def _load_jsonl(self, path: str) -> List[dict]:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def _load_all_datasets(self) -> Dict[str, List[dict]]:
        datasets = {}
        for key, paths in self.dataset_paths.items():
            if isinstance(paths, list):
                data = []
                for p in paths:
                    data.extend(self._load_jsonl(p))
                datasets[key] = data
            else:
                datasets[key] = self._load_jsonl(paths)
        
        # CoT 추가 (옵션)
        if self.use_cot:
            print("\nAdding CoT to datasets (rule-based, no API needed)...")
            for key in datasets:
                print(f"  Processing {key}...")
                datasets[key] = [add_cot_to_sample(ex) for ex in datasets[key]]
            print("CoT addition completed.\n")
        
        return datasets

    def _print_dataset_stats(self):
        print("\nDataset Statistics (with CoT):")
    
        for key, data in self.datasets.items():
            print(f"{key}: {len(data):,} samples")
            
            if data and 'answer' in data[0]:
                from collections import Counter
                
                # answer 추출 (리스트면 첫 번째 요소 또는 문자열로 변환)
                answers = []
                for d in data:
                    ans = d.get('answer')
                    if ans:
                        # 리스트면 첫 번째 요소 사용
                        if isinstance(ans, list):
                            if len(ans) > 0:
                                answers.append(str(ans[0]))
                        else:
                            answers.append(str(ans))
                
                if answers:
                    answer_dist = Counter(answers)
                    
                    # yes/no/maybe만 출력
                    key_answers = {k: v for k, v in answer_dist.items() if k in {'yes', 'no', 'maybe'}}
                    other_count = sum(v for k, v in answer_dist.items() if k not in {'yes', 'no', 'maybe'})

                    if key_answers:
                        print(f"  Class distribution (yes/no/maybe): {key_answers}")
                    if other_count > 0:
                        print(f"  Other answers: {other_count} samples ({len([k for k in answer_dist.keys() if k not in {'yes', 'no', 'maybe'}])} unique)")
        print()


    def get_ratios_for_epoch(self, epoch: int) -> Dict[str, float]:
        for sched in self.curriculum_schedule:
            if epoch in sched["epochs"]:
                return sched["ratios"]
        raise ValueError(f"No curriculum schedule found for epoch {epoch}")

    def get_epoch_samples_count(self, epoch: int) -> int:
        if epoch <= 2:
            return 1000
        elif epoch ==5:
            return 2000
        else:
            return 1500

    def get_epoch_batches(self, epoch: int) -> List[List[dict]]:
        ratios = self.get_ratios_for_epoch(epoch)
        
        epoch_samples = 3000
        
        samples_per_dataset = {k: int(ratios[k] * epoch_samples) for k in ratios}
        
        epoch_data = []
        for k, n in samples_per_dataset.items():
            data = self.datasets[k]
            print(f"  {k}: requesting {n:,} from {len(data):,} available")
            
            # 클래스 균형 샘플링 적용
            sampled = balanced_sampling(data, n, balance_key='answer')
            epoch_data.extend(sampled)
        
        random.shuffle(epoch_data)
        batches = [epoch_data[i:i+self.batch_size] for i in range(0, len(epoch_data), self.batch_size)]
        
        print(f"  Final: {len(epoch_data):,} samples in {len(batches):,} batches")
        
        # 클래스 분포 출력
        from collections import Counter
        answer_dist = Counter([ex.get('answer') for ex in epoch_data])
        print(f"  Class distribution: {dict(answer_dist)}")
        
        return batches


# 커리큘럼 스케줄
TARGET_WITH_LABELED = [
    {"epochs": [1], "ratios": {
        "pubmedqa_artificial": 0.50,
        "bioasq": 0.30, 
        "medqa_medmcqa": 0.20
    }},
    {"epochs": [2], "ratios": {
        "pubmedqa_artificial": 0.40,
        "pubmedqa_labeled_fold0_train": 0.05,
        "bioasq": 0.35,
        "medqa_medmcqa": 0.20,
    }},
    {"epochs": [3, 4], "ratios": {
        "pubmedqa_artificial": 0.45,
        "pubmedqa_labeled_fold0_train": 0.05,
        "bioasq": 0.30,
        "medqa_medmcqa": 0.20
    }},
    {"epochs": [5], "ratios": {
        "pubmedqa_labeled_fold0_train": 0.20,
        "bioasq": 0.30,
        "pubmedqa_artificial": 0.50
    }},
]


# 데이터셋 경로
DATASET_PATHS = {
    "bioasq": "/home/jovyan/sy/gptoss/project/data/raw/BioASQ/_processed/bioasq_golden_all_raw.jsonl",
    "pubmedqa_labeled_fold0_train": "/home/jovyan/sy/gptoss/project/data/raw/PubMedQA_official/pubmedqa-master/pubmedqa-master/data/pqal_fold0/train_set.jsonl",
    "pubmedqa_artificial": "/home/jovyan/sy/gptoss/project/data/raw/PubMedQA_hf_artificial/pubmedqa_pqa_artificial_train.jsonl",
    "medqa_medmcqa": [
        "/home/jovyan/sy/gptoss/project/data/raw/MedQA_hf_bigbio_4opt/medqa4_med_qa_en_4options_bigbio_qa_train.jsonl",
        "/home/jovyan/sy/gptoss/project/data/raw/MedMCQA_hf/medmcqa_train.jsonl"
    ],
}