import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F    
import pickle as pkl


class DGACLassifier1(nn.Module):
    def __init__(self, feat_1):
        super(DGACLassifier1, self).__init__()
        self.emb = nn.Embedding(feat_1, 100)
        self.CNN = nn.Conv1d(in_channels=100, out_channels=256, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = LSTM(9216, 128, bidirectional=True, batch_first=True)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.emb(x).float()
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.CNN(x))
        x = self.pool1(x)
        x = self.flat(x)
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pth', map_location=torch.device('cpu'))
model.to(device)

test_df = pd.read_csv('test.csv')
res_domain = test_df['domain']
test_df['domain'] = test_df['domain'].apply(lambda x: list(x.split('.')[0]))

chars = ['-', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']

char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
int_to_char = dict((i + 1, c) for i, c in enumerate(chars))

NUM_VOCAB = len(chars)
NUM_CHARS = 75

x = np.zeros((len(test_df), NUM_CHARS))

for i, row in test_df.iterrows():
    x[i, :len(row['domain'])] = np.array([char_to_int[c] for c in row['domain']])

model.eval()
predictions = []

with torch.no_grad():
    for i in range(len(x)):
        data = torch.tensor(x[i], dtype=torch.long).to(device).unsqueeze(0)
        output = model(data)
        pred = torch.round(output).cpu().item()
        predictions.append(int(pred))


result = pd.DataFrame({'domain': res_domain, 'is_dga': predictions})
result.to_csv('prediction.csv', header=True, index=False)
