from OCR import ocr_clothing
from entity_recognition import FashionNER
from transformers import AutoTokenizer
import pandas as pd
import torch

IMG_PATH = 'test2.png'

# text = ocr_clothing(IMG_PATH)
# print(text)

data = pd.read_csv('full_dataset.csv')
words = data['word'].tolist()
tags = data['tag'].tolist()
num_tag = len(data['tag'].unique())

max_len = 512
chunk_size = max_len - 2
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

word_chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
tag_chunks = [tags[i:i + chunk_size] for i in range(0, len(tags), chunk_size)]
if len(tag_chunks[-1]) != len(tag_chunks[-2]):
    for i in range(chunk_size - len(tag_chunks[-1])):
        tag_chunks[-1].append('O')

tokens = tokenizer(word_chunks, is_split_into_words=True, truncation=True, padding=True, return_tensors='pt')
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

word_ids = []
for i in range(len(tokens['input_ids'])):
    word_ids.append(tokens.word_ids(i))


tag_mapping = {
    'B_from': 0,
    'I_from': 1,
    'B_gender': 2,
    'I_gender': 3,
    'B_brand': 4,
    'I_brand': 5,
    'B_item_name': 6,
    'I_item_name': 7,
    'O': 8
}

tag_ids = []
for i in range(len(word_ids)):
    temp = []
    for j in range(1, len(word_ids[i]) - 1):
        if word_ids[i][j] is None:
            temp.append(tag_mapping['O'])
        else:
            temp.append(tag_mapping[tag_chunks[i][j - 1]])
    tag_ids.append(temp)

tag_ids = torch.tensor(tag_ids)
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)

model = FashionNER(num_tag=len(tags))
model.fit(input_ids=input_ids, attention_mask=attention_mask, tags=tags_ids, epochs=10)
