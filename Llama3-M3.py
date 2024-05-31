import transformers
import torch
import csv

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
 
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)   


file_name = "documents-train.tsv"

doc_id_list = []
snt_id_list = []
snt_source_list = []

with open(file_name, "r", encoding="utf-8") as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        doc_id_list.append(row["doc_id"])
        snt_id_list.append(row["snt_id"])
        snt_source_list.append(row["snt_source"])
      

import csv

file_name2 = "terms-train.tsv"
term_list = []
snt_id_list2 = []
difficulty_list = []
exp_id_list = []

with open(file_name2, "r", encoding="utf-8") as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        snt_id_list2.append(row["snt_id"])
        term_list.append(row["term"])  
        difficulty_list.append(row["difficulty"])
        exp_id_list.append(row["exp_id"])

       

for j in range(1,len(snt_id_list)):    
        answer_list=[]
        answer_list2=[]
        Human_answer=""
        for i in range (0,len(snt_id_list2)):
            if snt_id_list2[i]==snt_id_list[j-1]:
                Human_answer=term_list[i]
    
        instruction = f"Extract complex words from sentence, Do not generate long tesxt, "
        messages = [ {"role": "system", "content": f"Human answered to find complex term is {Human_answer}"},
            {"role": "user", "content": f"{snt_source_list[j-1]}{instruction}"},]       
        prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)    
        answer_list.append(outputs[0]["generated_text"][len(prompt):])
        
        
        messages = [ {"role": "user", "content": f"{snt_source_list[j]}{instruction}"},]
        prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs2 = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        answer_list2.append(outputs2[0]["generated_text"][len(prompt):])
        
        combined_answers = "".join(answer_list2)  
        file_path = f"result_{snt_id_list[j]}.txt"
        with open(file_path, 'w') as txtfile:
            txtfile.write(combined_answers) 

   