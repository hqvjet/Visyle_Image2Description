import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLMRobertaModel
from torchcrf import CRF
from tqdm import tqdm

class NER(nn.Module):
    def __init__(self, hidden_dim, num_tag):
        super(NER, self).__init__()

        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tag)
        self.crf = CRF(num_tag, batch_first=True)

    def forward(self, input_ids, attention_mask=None, tags=None):
        with torch.no_grad():
            output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)

        lstm_out, _ = self.lstm(output.last_hidden_state)
        emissions = self.fc(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, attention_mask.bool())

class FashionNER:
    def __init__(self, num_tag):
        self.model = NER(128, num_tag)
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda')

    def fit(self, train=None, val=None, test=None, epochs=10):
        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for input, attention_mask, tag in tqdm(train, desc=f'Epoch {epoch+1}/{epochs}'):
                input = input.to(self.device)
                attention_mask = attention_mask.to(self.device)
                tag = tag.to(self.device)

                self.opt.zero_grad()
                loss = self.model(input, attention_mask, tag)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()

            print(f'Loss: {total_loss/len(train)}')

            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for input, attention_mask, tag in val:
                    input = input.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    tag = tag.to(self.device)
                    loss = self.model(input, attention_mask, tag)
                    total_loss += loss.item()

            print(f'Val Loss: {total_loss/len(val)}')


        torch.save(self.model.state_dict(), 'model.pt')
