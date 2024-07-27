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


# аналогично обрабатываем вылидационную выборку
val_df = pd.read_csv('val.csv')
val_df['domain'] = val_df['domain'].apply(lambda x: list(x.split('.')[0]))

chars = ['-', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']

char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
int_to_char = dict((i + 1, c) for i, c in enumerate(chars))

NUM_VOCAB = len(chars)
NUM_CHARS = 75

x = np.zeros((len(val_df), NUM_CHARS))
y = val_df['is_dga'].to_numpy()

for i, row in val_df.iterrows():
    x[i, :len(row['domain'])] = np.array([char_to_int[c] for c in row['domain']])


model.eval()

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

with torch.no_grad():
    for i in range(len(x)):
        data = torch.tensor(x[i], dtype=torch.long).to(device).unsqueeze(0)
        output = model(data)
        
        # порог отсечения для бинарной классификации 0.5
        pred = torch.round(output).cpu().item()
        true_label = y[i].item()
        
        if pred == 1 and true_label == 1:
            true_positive += 1
        elif pred == 1 and true_label == 0:
            false_positive += 1
        elif pred == 0 and true_label == 1:
            false_negative += 1
        elif pred == 0 and true_label == 0:
            true_negative += 1

# вычисление метрик
accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# вывод результатов
print(f'True positive: {true_positive}')
print(f'False positive: {false_positive}')
print(f'False negative: {false_negative}')
print(f'True negative: {true_negative}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')


# сохраняем в файл
with open('validation.txt', 'w') as f:
    f.write(f'True positive: {true_positive}\n')
    f.write(f'False positive: {false_positive}\n')
    f.write(f'False negative: {false_negative}\n')
    f.write(f'True negative: {true_negative}\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1: {f1:.4f}\n')
