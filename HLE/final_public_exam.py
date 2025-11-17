from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

local_path = "/home/jovyan/FLPD/magi/model"
print('******************************')
print('추론 모델 불러오기')
print('******************************')
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path,
    device_map="auto", 
    dtype="auto" 
)

import pandas as pd
valid = pd.read_json("/home/jovyan/FLPD/magi/CertifiedTest/valid.json", orient="records", lines=True)
tmp_list = []

base_gpt_instruction = """Read the question and select one of the multiple-choice answers listed under "Answer Choices."
Select the answer first and explain it later.
The first letter should be one of the multiple-choice choices listed in the question.
If you don't know the answer, choose it first and explain why.

<Example 1>
Question: What is 1 plus 1? 
Answer Choices: 
A. 1 
B. 2 
C. 3 
D. 4
Answer: B
Explanation: 1 plus 1 equals 2.

<Example 2>
Question: Which element has the chemical symbol 'O'? 
Answer Choices: 
A. Gold 
B. Oxygen 
C. Hydrogen 
D. Carbon
Answer: B
Explanation: The symbol 'O' corresponds to Oxygen.

<Example 3>
Question: What is the capital of France? 
Answer Choices: 
A. Rome 
B. Berlin 
C. Paris 
D. Madrid
Answer: C
Explanation: Paris is the capital city of France.

<Example 4>
Question: Which planet is known as the Red Planet?
Answer Choices: 
A. Earth 
B. Mars 
C. Jupiter 
D. Venus
Answer: B
Explanation: Mars appears reddish due to iron oxide on its surface, so it's called the Red Planet.

<Example 5>
Question: If you are not sure about the answer, what should you do? 
Answer Choices: 
A. Skip it 
B. Guess without explaining 
C. Choose one and explain why you're unsure 
D. Leave blank
Answer: C
Explanation: When uncertain, still choose one option and explain your reasoning."""

for c, i in enumerate(range(len(valid))):
    print('******************************')
    print(c+1,"번 문항 추론 중")
    print('******************************')
    prompt = f"### Instruction:\n{base_gpt_instruction}\n\n### Input:\n{valid.iloc[i]['question']}\n\n### Response:\n"
    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 생성
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=2000,        
        temperature=1.1,           
        top_p=0.9,                  
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

    )
    # 출력
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
    tmp_list.append(result)

result_path = f"/home/jovyan/FLPD/magi/CertifiedTest/final_result1.json"
pd.Series(tmp_list).to_json(result_path, orient="records", lines=True, force_ascii=False)

print('**************************************')
print('추론 완료')
print('**************************************')

import re
def select_single_char(text_origin):
    text = text_origin
    text = text.replace('.',"")
    words = re.split(r"[ \n*]+", text)
    # 1. 첫 단어가 한 글자 & 대문자 영어
    if words and len(words[0]) == 1 and words[0].isalpha() and words[0].isupper():

        return words[0]
        
    # 2. 'Answer' 또는 'ANSWER' 다음의 한 문장 안에서 한 글자 대문자 영어 찾기
    text = text_origin
    text = text.split('\n')[0]
    match = re.search(r'(?:Answer:|ANSWER:)\s*:?\s*([A-Za-z])\b', text)
    if match:

        return match.group(1)

    # 3.  Answer 또는 ANSWER 뒤의 문장에서 첫 '.' 이전 한 글자 대문자 영어 찾기
    text = text_origin
    match = re.search(r'(?:correct answer|Answer|ANSWER|final answer)\s*:?\s*(.*)', text, re.DOTALL)
    if match:
        after_answer = match.group(1).strip()
        first_sentence = re.split(r'[.]', after_answer)[0]
        single_upper = re.search(r'\b([A-Z])\b', first_sentence)
        if single_upper:

            return single_upper.group(1)
            
    # 4. '.' 기준 첫 문장에서 처음 등장하는 한 글자 대문자 영어 찾기
    text = text_origin
    first_sentence = text.split('.')[0]
    single_upper = re.search(r"(?<![-,])(?<!,\s)\b(?:I(?!['’]ll\b)|[A-HJ-Z])\b(?![-])", first_sentence)
    if single_upper:

        return single_upper.group(0)



    # 5. 전체에서 처음 나오는 한 글자짜리 대문자 영어 찾기
    for w in words[1:]:
        if len(w) == 1 and w.isalpha() and w.isupper():

            return w

    return words[0][0]


result = pd.read_json(result_path, orient="records", lines=True)
result_list = []
for r in result[0]:
    result_list.append(select_single_char(r.split('### Response:\n')[1])) # 첫번째줄에 띄어쓰기 기준으로 나눴을때 한글자인거

result_anla_path = f"/home/jovyan/FLPD/magi/CertifiedTest/final_result_anal1.json"
pd.DataFrame(result_list).to_json(result_anla_path, orient="records", lines=True, force_ascii=False)
result_anla = pd.read_json(result_anla_path, orient="records", lines=True)
correct_answer_num = sum(valid['answer'] == result_anla[0])
sm =  round(correct_answer_num/118,3)
sc = 0.100
sh = 0.700
hle_acc = round(((sm-sc)/(sh-sc))*100,3)

print(f"118개 문항 중 맞춘 갯수 : {correct_answer_num}")
# print(f'Sc(우연에 의한 성능) : 0.100')
# print(f'Sh(인간 평가자의 평균 성능) : 0.700')
print(f'Sm(모델 정확도) : {sm}')
print(f'HLE 응답정확도: {hle_acc}%')
print('**************************************')