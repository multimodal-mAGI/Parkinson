# gpt20_model.py

# GPT-OSS-20B + LoRA/QLoRA 모델 및 토크나이저 로딩 (A100 최적화)로딩 유틸리티

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import gc

# A100 호환성 체크
def check_gpu_compatibility():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device_name = torch.cuda.get_device_name()
    print(f"GPU: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # A100 확인
    if "A100" in device_name:
        print("✅ A100 detected - enabling optimizations")
        return True
    else:
        print(f"⚠️  Not A100, but proceeding with {device_name}")
        return False



def load_tokenizer(model_name_or_path: str):
    print(f"Loading tokenizer: {model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast=True,
        )
    except Exception as e:
        print(f"Fast tokenizer failed: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast=False,
        )
     
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def load_lora_model(
    model_name_or_path: str, 
    lora_r=8, 
    lora_alpha=16, 
    lora_dropout=0.05, 
    use_qlora=False,
    use_mxfp4=False,
    device_map="auto"
):
    is_a100 = check_gpu_compatibility()
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Loading model: {model_name_or_path}")
    print(f"Model may already be quantized, loading without additional quantization...")


    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        dtype=torch.bfloat16 if is_a100 else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # 양자화 상태 확인
    if hasattr(model.config, 'quantization_config') and hasattr(model.config.quantization_config, 'quant_method'):
        print(f"✅ Model loaded with {model.config.quantization_config.quant_method} quantization")
    else:
        print("✅ Model loaded")

    


    # A100에 최적화된 LoRA 설정
    if is_a100:
        # A100은 더 큰 LoRA rank 처리 가능
        lora_r = min(lora_r * 2, 32)  # rank를 2배로, 최대 32
        lora_alpha = lora_r * 2  # alpha도 비례적으로 증가
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none", 
        task_type="CAUSAL_LM",
        # A100에서는 더 많은 모듈 타겟 가능
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj"      # MLP (if available)
        ]
    )
    
    print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
    

    try:
        model = get_peft_model(model, lora_config)
        print("✅ LoRA applied successfully")
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        # 타겟 모듈이 맞지 않을 수 있으므로 기본 설정으로 재시도
        print("Retrying with basic target modules...")
        lora_config.target_modules = ["q_proj", "v_proj"]  # 기본적인 attention 모듈만
        model = get_peft_model(model, lora_config)
        print("✅ LoRA applied with basic target modules")
    

    # A100 최적화 설정
    if is_a100:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TensorFloat-32 enabled for A100")

    # LoRA 파라미터 gradient 활성화 강제 확인
    print("🔧 Ensuring LoRA parameters are trainable...")
    lora_param_count = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            if not param.requires_grad:
                param.requires_grad = True
                print(f"  🔧 Fixed gradient for: {name}")
            lora_param_count += 1
            
    model.train()
    
    # 최종 gradient 상태 확인
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    
    print(f"✅ LoRA parameters found: {lora_param_count}")
    print(f"✅ Trainable parameters: {trainable_count:,}/{total_count:,}")
    
    if trainable_count == 0:
        raise RuntimeError("❌ CRITICAL: No trainable parameters found! LoRA may have failed.")
    
    
    # 최종 메모리 정리
    torch.cuda.empty_cache()
    
    try:
        # PEFT 모델의 경우 get_nb_trainable_parameters()가 튜플을 반환할 수 있음
        trainable_info = model.get_nb_trainable_parameters()
        if isinstance(trainable_info, tuple):
            trainable_params = trainable_info[0] if len(trainable_info) > 0 else 0
        else:
            trainable_params = trainable_info
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        if total_params > 0:
            print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    except Exception as e:
        print(f"Could not calculate parameter statistics: {e}")
        # 수동으로 계산
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (manual): {trainable_params:,}")
        print(f"Total parameters (manual): {total_params:,}")
        if total_params > 0:
            print(f"Trainable % (manual): {100 * trainable_params / total_params:.2f}%")
    
    return model


