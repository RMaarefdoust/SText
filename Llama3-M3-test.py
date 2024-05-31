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

#------------------------------------------------------
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
      

#------------------------------------------------------
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
#------------------------------------------------------
file_name = "documents-test.tsv"

doc_t_id_list = []
snt_t_id_list = []
snt_t_source_list = []

with open(file_name, "r", encoding="utf-8") as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        doc_t_id_list.append(row["doc_id"])
        snt_t_id_list.append(row["snt_id"])
        snt_t_source_list.append(row["source_snt"])

#----------------------------------------------------
file_name = "definitions_generated.tsv"
def_snt_id=[]
def_term=[]	
definition=[]	
positive=[]	
negative=[]

with open(file_name, "r", encoding="utf-8") as file:
    tsv_reader = csv.DictReader(file, delimiter='\t')
    for row in tsv_reader:
        def_snt_id.append(row["snt_id"])
        def_term.append(row["term"])
        definition.append(row["definition"])
        positive.append(row["positive"])
        negative.append(row["negative"])
        
    

#----------------------------------------------------------   

for j in range(400,len(snt_t_id_list)):    
        answer_list=[]
        answer_list2=[]
        answer_list3=[]
        Human_answer=""
        for i in range (0,len(snt_id_list2)):
            if snt_id_list2[i]==snt_id_list[j]:
                Human_answer=term_list[i]
                                
        Human_def=""
        Human_pos=""
        Human_neg=""
        for i in range (0,len(def_snt_id)):
            if def_snt_id[i]==snt_id_list[j]:
                Human_def=def_term[i]
                Human_pos=positive[i]
                Human_neg=negative[i] 
        
        
        instruction = f"Extract complex words from sentence, and labeled difficulty of word with one of 'e' means easy, m means medium, d: means difficult,and then generate a definition for each complex words based on sentence, generate an explanation for each complex word"    
        messages = [ {"role": "system", "content": f"Human answered to find complex term is {Human_answer}, and difficulty {difficulty_list},Human definition{Human_def}, Human good definition{Human_pos}, Human wrong definition{Human_neg}"},
            {"role": "user", "content": f"{snt_source_list[j]}{instruction}"},] 
        prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)    
        answer_list.append(outputs[0]["generated_text"][len(prompt):])
        
        instruction = f"Extract complex words from sentence, and labeed difficulty of word with one of 'e' means easy, m means medium, d: means difficult,and then generate a definition for each complex words based on sentence, generate an explanation for each complex word"    

        messages = [ {"role": "user", "content": f"{snt_t_source_list[j]}{instruction}"},]
        prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs2 = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        answer_list2.append(outputs2[0]["generated_text"][len(prompt):])
        answer=outputs2[0]["generated_text"][len(prompt):]
        answer_list2.append(answer)

        combined_answers = "".join(answer_list2)  
        file_path = f"result_all_{snt_t_id_list[j]}.txt"
        with open(file_path, 'w') as txtfile:
            txtfile.write(combined_answers) 
