import torch
import torch.nn.functional as F
from typing import List, Dict
import random


def compute_answer_focused_loss(
    model, 
    tokenizer, 
    batch: List[Dict],
    device: torch.device,
    use_reasoning: bool = False
):
    
    # 1. Answer verbalizer 토큰 ID
    verbalizers = ["yes", "no", "maybe"]
    try:
        # " yes", " no", " maybe" (공백 포함)
        verb_ids = [tokenizer.encode(" " + v, add_special_tokens=False)[0] for v in verbalizers]
    except:
        verb_ids = [tokenizer.convert_tokens_to_ids(v) for v in verbalizers]
    
    # 2. 배치별 input
    texts = []
    answer_labels = []
    
    for ex in batch:
        question = ex.get("question", "")
        abstract = ex.get("abstract", "")
        answer = ex.get("answer", "").lower()
        
        # Answer index 확인
        if answer in verbalizers:
            answer_idx = verbalizers.index(answer)
        else:
            continue  # 유효하지 않은 answer는 스킵
        
        # Input 텍스트 구성
        if use_reasoning and "cot_reasoning" in ex:
            # CoT reasoning
            text = f"""Question: {question}
Context: {abstract[:200] if abstract else 'N/A'}

Reasoning: {ex['cot_reasoning']}

Answer the question with yes, no, or maybe.
Answer:"""
        else:
            # 간단한 포맷
            text = f"""Question: {question}
Context: {abstract[:300] if abstract else 'N/A'}

Answer the question with yes, no, or maybe.
Answer:"""
        
        texts.append(text)
        answer_labels.append(answer_idx)
    
    if not texts:
        return None
    
    # 3. tokenizer
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)
    
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    
    # 4. Answer 토큰에 대한 logits만 추출
    answer_logits = logits[:, verb_ids]
    
    answer_labels_tensor = torch.tensor(answer_labels, dtype=torch.long).to(device)
    loss = F.cross_entropy(answer_logits, answer_labels_tensor)
    
    return loss


def train_answer_focused(
    model, 
    optimizer, 
    sampler, 
    tokenizer,
    num_epochs: int = 5,
    save_dir: str = "./checkpoints",
    eval_path: str = None,
    use_reasoning: bool = False,
    class_weights: bool = True
):
    """
    Answer-focused training loop
    
    Args:
        class_weights: yes/no/maybe에 대한 class weight 사용 여부
    """
    import os
    import csv
    from collections import Counter
    
    os.makedirs(save_dir, exist_ok=True)
    
    log_path = os.path.join(save_dir, "answer_focused_log.csv")
    log_fields = ["epoch", "train_loss", "pubmedqa_acc", "best"]
    
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    
    best_acc = 0.0
    

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs} - Answer-Focused Training")
        print(f"{'='*50}")
        
        batches = sampler.get_epoch_batches(epoch)
        epoch_loss = 0.0
        total_batches = 0
        
        model.train()
        
        for batch_idx, batch in enumerate(batches):
            # Answer-focused loss
            loss = compute_answer_focused_loss(
                model, tokenizer, batch, model.device, use_reasoning=use_reasoning
            )
            
            if loss is None:
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            epoch_loss += loss.item()
            total_batches += 1
            
            if batch_idx % 100 == 0:
                avg_loss = epoch_loss / max(1, total_batches)
                print(f"  Batch {batch_idx}/{len(batches)} | Avg Loss: {avg_loss:.4f}")
        
        avg_loss = epoch_loss / max(1, total_batches)
        print(f"Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}")
        
        # 평가
        if eval_path and tokenizer:
            from gpt20_cot_train_logic import evaluate_on_pubmedqa
            
            acc = evaluate_on_pubmedqa(model, tokenizer, eval_path, max_samples=500)
            
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_path = os.path.join(save_dir, "best_model")
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                print(f"🎉 NEW BEST: {best_acc*100:.2f}%")
            
            # Log
            with open(log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "pubmedqa_acc": acc,
                    "best": is_best
                })
        
        # 체크포인트 저장
        save_path = os.path.join(save_dir, f"epoch{epoch}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    print(f"\nTraining completed! Best accuracy: {best_acc*100:.2f}%")
    return best_acc
