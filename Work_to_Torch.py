import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt


def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text


url_1 = 'https://www.gutenberg.org/cache/epub/67979/pg67979.txt'  
url_2 = 'https://www.gutenberg.org/cache/epub/75342/pg75342.txt' 

text_1 = get_text_from_url(url_1)
text_2 = get_text_from_url(url_2)


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower() 
    return text.split()

tokens_1 = preprocess_text(text_1)
tokens_2 = preprocess_text(text_2)

vocab = list(set(tokens_1 + tokens_2))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Создание датасета
class TextDataset(Dataset):
    '''Класс для создания датасета из текста.'''
    def __init__(self, tokens, label):
        self.tokens = tokens
        self.label = label

    def __len__(self):
        '''Возвращает количество биграмм в тексте минус 1.'''
        return len(self.tokens) - 2 

    def __getitem__(self, idx):
        '''Возвращает биграмму и метку.'''
        x = [word_to_idx[self.tokens[idx]], word_to_idx[self.tokens[idx + 1]]]
        y = self.label
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Метки: 0 для текста 1, 1 для текста 2
dataset_1 = TextDataset(tokens_1, label=0)
dataset_2 = TextDataset(tokens_2, label=1)

# Объединяем датасеты
dataset = dataset_1 + dataset_2
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Используется для итерации по датасету с заданным размером батча (32) и перемешиванием данных.

class SimpleNLPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleNLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * 2, num_classes)  # Умножаем на 2 для биграмм

    def forward(self, x):
        embeds = self.embedding(x)  # Преобразуем слова в эмбеддинги
        embeds = embeds.view(embeds.size(0), -1)  # Разворачиваем в один вектор
        out = self.fc(embeds)
        return out


vocab_size = len(vocab)
embed_dim = 10
num_classes = 2

model = SimpleNLPModel(vocab_size, embed_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()