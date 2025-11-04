from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
for num in range(3,10):

    local_path = "/home/jovyan/FLPD/magi/model"
    # local_path = "/home/jovyan/FLPD/magi/deepseekmodel"
    # hle finetuning
    # local_path = "/home/jovyan/FLPD/magi/result/finetune/1.multiple"
    # deepseek
    # local_path = "/home/jovyan/FLPD/magi/deepseekmodel"
    # gpt oss
    # local_path = '/home/jovyan/FLPD/magi/gemma3_27b'
    #일반 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path,
        device_map="auto",  # GPU 자동 할당
        torch_dtype="auto" # bfloat16/float16 자동 설정 (A100/V100에서 권장)
    )
    # # llama 70b에서 할것
    # model_id = 'meta-llama/Llama-3.3-70B-Instruct'
    # local_path = '/home/jovyan/FLPD/magi/llama70b'
    # bnb = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # tok = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    # max_mem = {
    #     0: "38GiB",      # GPU 여유에 맞게
    #     "cpu": "120GiB"  # RAM 넉넉히 (가능하면 크게)
    # }

    # model = AutoModelForCausalLM.from_pretrained(
    #     local_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    #      max_memory = max_mem, offload_folder = '/home/jovyan/FLPD/tmp'

    # )
    # model.eval()
    # from transformers import BitsAndBytesConfig

    # bnb8 = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_enable_fp32_cpu_offload=True  # ★ 8bit 전용
    # )
    # tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     local_path,
    #     quantization_config=bnb8,
    #     device_map='auto',  # 또는 "auto"
    #     max_memory={0: "38GiB", "cpu": "120GiB"},
    #     offload_folder='/home/jovyan/FLPD/tmp',
    #     torch_dtype=torch.bfloat16
    # )


    #finetuing 모델 불러오기
    # from peft import PeftModel

    # base_model_path = "/home/jovyan/FLPD/magi/model"       # 원래 llama 모델
    # adapter_path    = "/home/jovyan/FLPD/magi/result/finetune/1.multiple"  # LoRA adapter 폴더

    # tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     device_map="auto",
    #     torch_dtype="auto"
    # )

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
    base_gpt_instruction = """Read the question and select one of the multiple-choice answers listed under "Answer Choices."
    Select the answer first and explain it later.
    The first letter should be one of the multiple-choice choices listed in the question.
    If you don't know the answer, choose it first and explain why.

    <Example 1>
    Question : What is 1 plus 1? 
    Answer Choices: A. 1 B. 2 C. 3 D. 4
    Answer : B
    Explanation : 1 plus 1 equals 2.

    <Example 2>
    Question : Which element has the chemical symbol 'O'? 
    Answer Choices: A. Gold B. Oxygen C. Hydrogen D. Carbon
    Answer : B
    Explanation : The symbol 'O' corresponds to Oxygen.

    <Example 3>
    Question : What is the capital of France? 
    Answer Choices: A. Rome B. Berlin C. Paris D. Madrid
    Answer : C
    Explanation : Paris is the capital city of France.

    <Example 4>
    Question : Which planet is known as the Red Planet?
    Answer Choices: A. Earth B. Mars C. Jupiter D. Venus
    Answer : B
    Explanation : Mars appears reddish due to iron oxide on its surface, so it's called the Red Planet.

    <Example 5>
    Question : If you are not sure about the answer, what should you do? 
    Answer Choices: A. Skip it B. Guess without explaining C. Choose one and explain why you're unsure D. Leave blank
    Answer : C
    Explanation : When uncertain, still choose one option and explain your reasoning."""
    base1_instruction = """Read the question and select one of the multiple-choice answers listed under "Answer Chioces. Select the answer first and explain it later. The first letter should be one of the multiple-choice choices listed in the question. If you don't know the answer, choose it first and explain why.
    <Example>
    Input: What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
    Adding 1 to 1 makes 2, so the correct answer is B.
    Answer : B
    </Example> """
    base_instruction_cot = """Read the question and select one of the multiple-choice answers listed under "Answer Chioces. Select the answer first and explain it later. The first letter should be one of the multiple-choice choices listed in the question. If you don't know the answer, choose it first and explain why.
    <Example>
    Question : What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
    Answer : B
    </Example> """
    base1_instruction_cot = """Read the question and select one of the multiple-choice answers listed under "Answer Chioces. Select the answer first and explain it later. The first letter should be one of the multiple-choice choices listed in the question. If you don't know the answer, choose it first and explain why.
    <Example>
    Input: What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
    Adding 1 to 1 makes 2, so the correct answer is B.
    Answer : B
    </Example> """

    base2_instruction = """Read the question and select one of the multiple-choice answers listed below the “Answer Choices.” Explain your reasoning and select your answer. If you don't know the answer, choose the option you believe is closest to the correct one. Select the correct answer within 2000 characters.
    <Example>
    Input: What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
    Response : Adding 1 to 1 makes 2, so the correct answer is B.
    Answer : B
    </Example> """


    gpt_instruction = '''You are a logical thinking assistant. Break down the given problem step by step and write down your solution process.
    After writing your reasoning process, be sure to write the correct answer in the following format at the end:

    Answer : <Choice>

    Even if you are unsure, make your best choice and be sure to select one.

    <Example>
    Input: What is 1 plus 1? A. 1 B. 2 C. 3 D. 4
    Adding 1 to 1 makes 2, so the correct answer is B.
    Answer : B
    </Example>
    '''

    instructions = [base_instruction,base1_instruction,base_instruction_cot,base1_instruction_cot]
    import torch

    for c, i in enumerate(range(len(valid))):
        print(num, c)
        prompt = f"### Instruction:\n{base_gpt_instruction}\n\n### Input:\n{valid.iloc[i]['question']}\n\n### Response:\n"
        # 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # 생성
        with torch.no_grad():
            outputs = model.generate(
            **inputs,
            max_new_tokens=5000,          # 너무 크게 주지 말고, 예상 답변 길이에 맞게 조정
            # temperature=0.5,              # 답변 다양성 조절
            # top_p=0.9,                    # nucleus sampling
            # repetition_penalty=1.15,       # 같은 문장 반복 억제
            # no_repeat_ngram_size=4,       # n-gram 반복 금지 (3 이상 권장)
            # eos_token_id=tokenizer.eos_token_id,  # 문장 끝나면 멈추도록 >> DeepSeek-R1-Distill-Qwen-7B 에서 적용
            # pad_token_id=tokenizer.eos_token_id,   # pad 토큰도 eos로 처리 (에러 방지) >> DeepSeek-R1-Distill-Qwen-7B >> 에서 적용
            # max_time=1800, # DeepSeek-R1-Distill-Qwen-7B >> 에서 적용 3:37
            # use_cache=True,
            # output_scores = False,
            # do_sample = False
        )

        # 출력
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(result)
        tmp_list.append(result)
    pd.Series(tmp_list).to_json(f"/home/jovyan/FLPD/magi/result/final_deepseek3_{num}.json", orient="records", lines=True, force_ascii=False)
