import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_trained_model(base_model_name, checkpoint_path):
    """학습된 PEFT 모델 로딩"""
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    print(f"Loading PEFT model from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    return model


def load_tokenizer(model_name):
    """토크나이저 로딩"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def predict_answer(model, tokenizer, question):

    # 토크나이징
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    # 예측
    verbalizers = ["yes", "no", "maybe"]

    def get_verbalizer_ids_safe(tokenizer, verbalizers):

        ids = []
        
        for word in verbalizers:
            # 1: 공백 없이 시도
            tokens = tokenizer.encode(word, add_special_tokens=False)
            
            if len(tokens) == 1:
                ids.append(tokens[0])
            else:
                # 2: 공백 포함해서 시도
                tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
                
                if len(tokens_with_space) == 1:
                    ids.append(tokens_with_space[0])
                elif len(tokens_with_space) > 1:
                    ids.append(tokens_with_space[-1])
                else:
                    # 3: vocab에서 직접 찾기
                    if word in tokenizer.get_vocab():
                        ids.append(tokenizer.get_vocab()[word])
                    else:
                        # 4: UNK 토큰
                        ids.append(tokenizer.unk_token_id)
                        print(f"⚠️ Warning: '{word}' not found, using UNK token")
            
        return ids

    verbalizer_ids = get_verbalizer_ids_safe(tokenizer, verbalizers)

    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]
    
    # 각 답변의 점수
    scores = [logits[vid].item() for vid in verbalizer_ids]
    pred_idx = int(torch.tensor(scores).argmax())
    predicted = verbalizers[pred_idx]
    
    scores_dict = {v: s for v, s in zip(verbalizers, scores)}
    
    return predicted, scores_dict




def evaluate_pubmedqa_full(model, tokenizer, test_file, max_samples=None):
    """전체 PubMedQA 테스트셋 평가"""
    print(f"\nEvaluating on {test_file}")
    
    # JSONL 파일 읽기
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Total samples: {len(data)}")
    
    correct = 0
    results = []
    
    for i, item in enumerate(data):
        question = item['question']
        correct_answer = item['answer']
        
        # 예측
        predicted, scores = predict_answer(model, tokenizer, question)
        is_correct = (predicted == correct_answer)
        
        if is_correct:
            correct += 1
        
        results.append({
            'pmid': item.get('pmid', f'item_{i}'),
            'question': question,
            'correct_answer': correct_answer,
            'predicted': predicted,
            'correct': is_correct,
            'scores': scores
        })
        
        if (i + 1) % 50 == 0:
            current_acc = correct / (i + 1)
            print(f"Progress: {i+1}/{len(data)} | Current Accuracy: {current_acc*100:.2f}%")
    
    # 최종 결과
    final_accuracy = correct / len(data)
    print(f"\nFinal Results:")
    print(f"Correct: {correct}/{len(data)}")
    print(f"Accuracy: {final_accuracy*100:.2f}%")
    
    return final_accuracy, results





def main():
    # 설정
    BASE_MODEL = "openai/gpt-oss-20b"
    CHECKPOINT_PATH = "best_model"     # 경로
    TEST_FILE = "pubmedqa_test.jsonl"

    print("Starting PubMedQA Test")
    
    # 모델과 토크나이저 로딩
    tokenizer = load_tokenizer(BASE_MODEL)
    model = load_trained_model(BASE_MODEL, CHECKPOINT_PATH)
    
    # 전체 평가
    print("FULL EVALUATION")
    print("="*60)
    
    if os.path.exists(TEST_FILE):
        response = input(f"Run full evaluation on {TEST_FILE}? (y/n): ")
        if response.lower() == 'y':
            max_samples = input("Max samples (Enter for all): ")
            max_samples = int(max_samples) if max_samples.strip() else None
            
            accuracy, results = evaluate_pubmedqa_full(model, tokenizer, TEST_FILE, max_samples)
            
            # 결과 저장
            output_file = f"pubmedqa_results_{os.path.dirname(CHECKPOINT_PATH)}_{os.path.basename(CHECKPOINT_PATH)}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'checkpoint': CHECKPOINT_PATH,
                    'accuracy': accuracy,
                    'total_samples': len(results),
                    'correct_count': sum(r['correct'] for r in results),
                    'results': results
                }, f, indent=2)
            print(f"\nResults saved to {output_file}")
    else:
        print(f"Test file not found: {TEST_FILE}")

if __name__ == "__main__":
    main()