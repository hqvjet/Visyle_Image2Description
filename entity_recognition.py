import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLMRobertaModel
from torchcrf import CRF

class NER(nn.Module):
    def __init__(self, hidden_dim, num_tag):
        super(NER, self).__init__()

        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tag)
        self.crf = CRF(num_tag, batch_first=True)

    def forward(self, input_ids, attention_mask, tags=None):
        with torch.no_grad():
            output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)

        lstm_out, _ = self.lstm(output.last_hidden_state)
        emissions = self.fc(lstm_out)

        if tags is not None:
            loss = self.crf(emissions, tags, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, attention_mask.bool())

class FashionNER:
    def __init__(self, num_tag):
        self.model = NER(256, num_tag)
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)

    def fit(self, input_ids, attention_mask, tags, epochs):
        for epoch in range(epochs):
            self.opt.zero_grad()
            loss = self.model(input_ids, attention_mask, tags)
            loss.backward()
            self.opt.step()

            print(f'Epoch: {epoch}, Loss: {loss.item()}')
