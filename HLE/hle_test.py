from transformers import AutoModelForCausalLM, AutoTokenizer

# finetuing 모델 불러오기
from peft import PeftModel

base_model_path = "/home/jovyan/FLPD/magi/model"       # 원래 llama 모델
adapter_path    = "/home/jovyan/FLPD/magi/result/finetune/1.multiple"  # LoRA adapter 폴더

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype="auto"
)

# model = PeftModel.from_pretrained(base_model, adapter_path)
import pandas as pd

valid = pd.read_json("/home/jovyan/FLPD/magi/data/valid.json", orient="records", lines=True)
tmp_list = []

no_instruction = ''
base_instruction = """Read the question and select one of the multiple-choice answers listed under "Answer Chioces. Select the answer first and explain it later. The first letter should be one of the multiple-choice choices listed in the question. If you don't know the answer, choose it first and explain why.
<Example>
Question : What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
Answer : B
</Example> """

import torch

for c, i in enumerate(range(len(valid))):
    print(c)
    prompt = f"### Instruction:\n{base_instruction}\n\n### Input:\n{valid.iloc[i]['question']}\n\n### Response:\n"
    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 생성
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=2000,          
    )

    # 출력
    tmp_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
pd.Series(tmp_list).to_json(f"/home/jovyan/FLPD/magi/result/llama3.1_resulttest.json", orient="records", lines=True, force_ascii=False)