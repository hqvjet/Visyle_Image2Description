from OCR import ocr_clothing
import pandas as pd
import torch
from preprocessing import PrepareTrainData
from entity_recognition import FashionNER
from torch.utils.data import TensorDataset, DataLoader

preprocessing = PrepareTrainData('full_dataset.csv')
data = preprocessing.data
print(data['x'].size())
print(data['y'].size())
print(data['attention_mask'].size())

train = TensorDataset(data['x'][:10000], data['attention_mask'][:10000], data['y'][:10000])
val = TensorDataset(data['x'][10000:12000], data['attention_mask'][10000:12000], data['y'][10000:12000])
test = TensorDataset(data['x'][12000:], data['attention_mask'][12000:], data['y'][12000:])

train_loader = DataLoader(train, shuffle=False, batch_size=512)
val_loader = DataLoader(val, shuffle=False, batch_size=512)
test_loader = DataLoader(test, shuffle=False, batch_size=512)

trainer = FashionNER(num_tag=len(preprocessing.tag_mapping))
trainer.fit(train_loader, val_loader, test_loader, epochs=10)
