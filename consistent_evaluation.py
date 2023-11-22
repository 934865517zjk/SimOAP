from itertools import count
import json
import pickle
from sys import argv
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUDA_AVAILABLE = False
if torch.cuda.is_available():
    CUDA_AVAILABLE = True
    print("CUDA IS AVAILABLE")
else:
    print("CUDA NOT AVAILABLE")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NLI_MODEL_PATH = '/consistent_model/'


text_list = []
personas = []
dialogs = []
query = []
gold = []
with open(os.path.join('/data/','candidate_subset.tsv')) as file:
    for line in file:
        text_list.append(line.strip())
    for i in range(int(len(text_list)/103)):
        personas.append(text_list[i*103][8:])
        query.append(text_list[i*103+1][6:])
        gold.append(text_list[i*103+2][5:])
        dialogs.append(text_list[i*103+3:i*103+100+3])

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

def get_dataloader(input_examples, tokenizer, device):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=100)
    return dataloader

pred_results = []
pro_entailment = []
pro_neutral = []
pro_contradiction = []
text_pro = []
text_pro_sorted = []
with torch.no_grad():
    for i in tqdm(range(len(personas))):
        cur_persona = personas[i]
        cur_dialogs = dialogs[i]
        cnt = 0
        cur_pred_results = []
        input_examples = []
        for dialog in cur_dialogs:
            input_examples.append(InputExample(str(cnt), cur_persona, dialog, '0'))
            cnt += 1
        train_dataloader = get_dataloader(input_examples, tokenizer, device)
        all_logits = None
        for batch in train_dataloader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].detach()
            else:
                all_logits = torch.cat((all_logits, outputs[0].detach()), dim=0)
        results = torch.argmax(all_logits, dim=1)
        
        sorces = torch.nn.functional.softmax(all_logits,dim=1)
        sorces = sorces.tolist()
        for j in range(len(sorces)):
            pro_entailment.append(sorces[j][2])
            pro_neutral.append(sorces[j][1])
            pro_contradiction.append(sorces[j][0])


for i in range(len(query)):
    for j in range(100):
        text_pro.append([dialogs[i][j],pro_entailment[i*100+j],pro_neutral[i*100+j],pro_contradiction[i*100+j]])
        
    k = sorted(text_pro[i*100:i*100+100],key=lambda x:x[1], reverse=True)
    text_pro_sorted = text_pro_sorted+k


with open('final.tsv', "w", encoding="utf-8") as outf:
    for i in range(len(query)):
        outf.write(f"persona:{personas[i]}\tquery:{query[i]}\tgold:{gold[i]}\n")
        for j in range(1):
            outf.write(f"{text_pro_sorted[i*100+j][0]}\n")
