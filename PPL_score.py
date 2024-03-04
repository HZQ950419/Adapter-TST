import sys
sys.path.append('/home/zqhu/home/adapter-transformers-tst/src/')
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

data_path =  'trained_models/adapter-tst-tense-pp-front-back-t5/generated_predictions_comp_1.txt'
# data_path = "trained_models/yelp_outputs/StyleTransformer/generated_predictions.txt"
# data_path =  '../style-transformer/save/tense_pp_removal/generated_predictions_comp.txt'
# data_path =  '../style-transformer/save/tense_pp_removal/generated_predictions.txt'
# data_path =  '../style-transformer-3-attri/save/tense_pp_removal/generated_predictions.txt'
# data_path = "trained_models/yelp_outputs/Human3/generated_predictions.txt"

def reader(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip('\n'))

    return data

data = reader(data_path)

encodings = tokenizer("\n\n".join(data), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512

nlls = []
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)
