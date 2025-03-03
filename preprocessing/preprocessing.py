import pandas as pd
import torch
from transformers import AutoTokenizer

class Preprocessing:
    def __init__(self):
        self.tag_mapping = {
            'B_from': 0,
            'I_from': 1,
            'B_gender': 2,
            'I_gender': 3,
            'B_brand': 4,
            'I_brand': 5,
            'B_item_name': 6,
            'I_item_name': 7
        }
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def padding(self, x, y, max_len=21):
        for i in range(len(x)): 
            if len(x[i]) > max_len:
                x[i] = x[i][:max_len - 1] + [2]
                y[i] = y[i][:max_len - 1] + [-100]
            else:
                x[i] = x[i] + [1] * (max_len - len(x[i]))
                y[i] = y[i] + [-100] * (max_len - len(y[i]))

        return x, y

    def identify(self, x, max_len=21):
        return self.tokenizer(x, truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    def reverse_ids_to_token(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def process_data(self, x):
        tokens = self.identify(x)
        input_ids, attention_mask = tokens['input_ids'][:, 1:], tokens['attention_mask'][:, 1:]
        word_ids = []

        for i in range(len(input_ids)):
            sep_mask = input_ids[i] == 2
            input_ids[i][sep_mask] = 1
            attention_mask[i][sep_mask] = 0
            word_ids.append(tokens.word_ids(i)[1:])

        return {'x': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}, word_ids
