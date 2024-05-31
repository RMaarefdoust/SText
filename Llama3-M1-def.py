import transformers
import torch
import json

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
list=[
    """Sentence: “With network and small screen device improvements, such as wireless abilities, increased memory and CPU speeds, users are no longer limited by location when accessing on-line information.”
Word: "network"    Difficulty: “easy”
Word :"CPU"      Difficulty: “difficult”
Sentences:” We are interested in studying the effect of users switching from a large screen device, such as a desktop or laptop to use the same web page on a small device, in this case a PDA (Personal Digital Assistant).”
Word: “desktop” 	Difficulty:” easy”
Word:”PDA”	Difficulty: “medium”	
Word:”personal digital assistant”	Difficulty: “easy”	
Word:”laptop”  Difficulty: “easy”	
Sentence:”We discuss three common transformation approaches for display of web pages on the small screen: Direct Migration, Linear and Overview.”
Word:”direct migration”		Difficulty: “difficult”
Word:”linear”		Difficulty: “difficult”
Word:”overview”	Difficulty: “difficult”
Sentence:”We introduce a new Overview method, called the Gateway, for use on the small screen that exploits a user’s familiarity of a web page.”
Word:”gateway” 	Difficulty: “difficult”
"""

]


import csv

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
      
#-------------------------------------------------------

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
#-------------------------------------------------------      

for j in range(0,5):#len(snt_id_list)):    
        answer_list=[]
        answer_list2=[]
        instruction = f"Extract complex words from sentence, Do not generate long tesxt "
        messages = [ {"role": "user", "content": f"{snt_source_list[j]}"},]
        prompt = pipeline.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)    
        answer_list.append(outputs[0]["generated_text"][len(prompt):])
        
        Human_answer=""
        for i in range (0,len(snt_id_list2)):
            if snt_id_list2[i]==snt_id_list[j]:
                Human_answer=term_list[i]
        
        Human_def=""
        Human_pos=""
        Human_neg=""
        for i in range (0,len(snt_id_list2)):
            if def_snt_id[i]==snt_id_list[j]:
                Human_def=def_term[i]
                Human_pos=positive[i]
                Human_neg=negative[i]       
        instruction2= f"Extract complex words from sentence, generate only one definition for each complex words"
        messages1 = [ {"role": "system", "content": f"Human answered to find complex term is {Human_answer},Human definition{Human_def}"},
            {"role": "user", "content": f"{snt_source_list[j]}{instruction2}"},]
        prompt = pipeline.tokenizer.apply_chat_template( messages1, tokenize=False, add_generation_prompt=True)
        terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs2 = pipeline( prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        answer_list2.append(outputs2[0]["generated_text"][len(prompt):])
        
        combined_answers = "".join(answer_list)  
        file_path = f"result{j}.txt"
        with open(file_path, 'w') as txtfile:
            txtfile.write(combined_answers) 

   