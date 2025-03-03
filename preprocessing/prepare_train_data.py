import pandas as pd
import torch
from preprocessing.preprocessing import Preprocessing

class PrepareTrainData(Preprocessing):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = self.get_data()

    def get_data(self):
        data = pd.read_csv(self.path)
        words = data['word'].tolist()
        tags = data['tag'].tolist()

        sentence = []
        sentence_tag = []
        temp_s = [words[0]]
        temp_t = [self.tag_mapping[tags[0]]]
        for word, tag in zip(words, tags):
            if tag[0] == 'I':
                temp_s.append(word)
                temp_t.append(self.tag_mapping[tag])
            else:
                sentence.append(' '.join(temp_s))
                sentence_tag.append(temp_t)
                temp_t = [self.tag_mapping[tag]]
                temp_s = [word]

        tokens = self.identify(sentence)
        inputs_ids = tokens['input_ids'][:,1:]
        attention_mask = tokens['attention_mask'][:,1:]
        tags = sentence_tag

        for i in range(len(inputs_ids)):
            sep_mask = inputs_ids[i] == 2
            inputs_ids[i][sep_mask] = 1
            attention_mask[i][sep_mask] = 0
            temp = []
            word_ids = tokens.word_ids(i)[1:]
            for j in word_ids:
                if j is not None:
                    temp.append(tags[i][j])
            temp = temp + [0] * (len(word_ids) - len(temp))
            tags[i] = temp

        return {'x': torch.tensor(inputs_ids), 'attention_mask': torch.tensor(attention_mask), 'y': torch.tensor(tags)}
