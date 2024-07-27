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
    
# считаем зараннее подготовленный датасет
df = pd.read_csv('proccessed2.csv')

df.rename(columns={'Domain': 'domain', 'Type': 'is_dga'}, inplace=True)
df = df.head(100000)

df['domain'] = df['domain'].apply(lambda x: list(x))
chars = ['-', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']

char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
int_to_char = dict((i + 1, c) for i, c in enumerate(chars))

NUM_VOCAB = len(chars)
NUM_CHARS = 75

# преобразуем каждый символ домена в число
x = np.zeros((len(df), NUM_CHARS))
y = df['is_dga'].to_numpy()

for i, row in df.iterrows():
    x[i, :len(row['domain'])] = np.array([char_to_int[c] for c in row['domain']])


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# гиперпаметры
epoch = 2
batch_size = 100
feat_1 = 75

ln = int(len(x) * 0.7)
train = x[:ln]
train_y = y[:ln]

test = x[ln:]
test_y = y[ln:]

# Формируем обучающую выборку
train_loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=False)

# обучение
model = DGACLassifier1(feat_1).to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.unsqueeze(1).to(device)
        target = target.to(device)
        
        output = model(data)
        output = output.squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print('Train Epoch: {} \tAverage Loss: {:.6f}'.format(i, average_loss))

# сохраняем модель
# model
torch.save(model, 'saved_model.pth')

# Тестирование
model.eval()

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

with torch.no_grad():
    for i, (data, y) in enumerate(test_loader):
        data = data.to(device)
        y = y.to(device)
        output = model(data)
        pred = (output > 0.5).float()
        for j in range(pred.size(0)):
            if pred[j] == 1.0 and y[j] == 1.0:
                true_positive += 1
            elif pred[j] == 1.0 and y[j] == 0.0:
                false_positive += 1
            elif pred[j] == 0.0 and y[j] == 1.0:
                false_negative += 1
            elif pred[j] == 0.0 and y[j] == 0.0:
                true_negative += 1


# Вычисление метрик
accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Вывод результатов
print(f'True positive: {true_positive}')
print(f'False positive: {false_positive}')
print(f'False negative: {false_negative}')
print(f'True negative: {true_negative}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')