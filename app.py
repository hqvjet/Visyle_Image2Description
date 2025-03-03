from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from OCR import ocr_clothing
from preprocessing import Preprocessing
from entity_recognition import NER
from schema import DescriptionForm
import torch

preprocessing = Preprocessing()
model = NER(128, len(preprocessing.tag_mapping))
model.load_state_dict(torch.load("model.pt"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/api/v1/description")
def get_description_form(image: UploadFile = File(...)):
    file_path = f'temp_{image.filename}'
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    text = ocr_clothing(file_path)
    
    input, word_ids = preprocessing.process_data(text)
    output = model(input['x'], input['attention_mask'])

    form = {
        'brand': [],
        'gender': [],
        'from': [],
        'item_name': []
    }

    for i in range(word_ids.__len__()):
        words = text[i].split(' ')
        prev_ids = -1

        for j in range(word_ids[i].__len__()):
            if word_ids[i][j] != None and word_ids[i][j] != prev_ids:
                key = [key for key, value in preprocessing.tag_mapping.items() if value == output[i][j]][0]
                prefix = key[:2]
                key = key[2:]

                if prefix == 'B_':
                    form[key].append([words[word_ids[i][j]]])
                elif prefix == 'I_':
                    form[key][-1].append(words[word_ids[i][j]])

            prev_ids = word_ids[i][j]

    os.remove(file_path)
    print(text)

    return {"text": form}
