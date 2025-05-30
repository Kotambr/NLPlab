import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re
import string
import requests
from bs4 import BeautifulSoup
import os

# Функция для загрузки текста с сайта Project Gutenberg
def get_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        start = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
        end = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
        if start != -1 and end != -1:
            text = text[start:end]
        return text
    except requests.RequestException as e:
        print(f"Ошибка при загрузке текста: {e}")
        return ""

# Загрузка текстов с нескольких URL
urls = [
    "https://www.gutenberg.org/cache/epub/84/pg84.txt"       # Frankenstein
]

text = ""
for url in urls:
    text += get_text_from_url(url)

# Предобработка текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    return words

words = preprocess_text(text)

# Создание словаря и обратного словаря
def build_vocab(words):
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    vocab["<unk>"] = len(vocab)  # Для неизвестных слов
    vocab["<eos>"] = len(vocab)  # Для обозначения конца предложения
    return vocab

def build_inverse_vocab(vocab):
    return {idx: word for word, idx in vocab.items()}

vocab = build_vocab(words)
inverse_vocab = build_inverse_vocab(vocab)

# Датасет
class TextDataset(Dataset):
    def __init__(self, words, vocab, context_size=3):
        self.data = []
        for i in range(len(words) - context_size):
            context = words[i:i+context_size]
            target = words[i+context_size]
            self.data.append((context, target))
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.vocab.get(w, self.vocab["<unk>"]) for w in context], dtype=torch.long)
        target_idx = torch.tensor(self.vocab.get(target, self.vocab["<unk>"]), dtype=torch.long)
        return context_idxs, target_idx

# Модель
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Функция обучения
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Функция предсказания
def predict_next_word(model, context, vocab, inverse_vocab, device, temperature=1.0):
    model.eval()
    context_idxs = torch.tensor([vocab.get(w, vocab["<unk>"]) for w in context], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(context_idxs)
        probabilities = F.softmax(output / temperature, dim=-1)
        predicted_idx = torch.multinomial(probabilities, 1).item()
    return inverse_vocab.get(predicted_idx, "<unk>")

# Параметры модели
context_size = 5
embedding_dim = 256
hidden_dim = 512
num_layers = 2
batch_size = 64
epochs = 5
learning_rate = 0.001

dataset = TextDataset(words, vocab, context_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NextWordPredictor(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Сохранение и загрузка модели
model_path = "next_word_model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    if checkpoint["vocab_size"] != len(vocab) or checkpoint["embedding_dim"] != embedding_dim:
        print("Параметры модели изменились. Переобучение модели...")
        train_model(model, dataloader, criterion, optimizer, epochs)
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "embedding_dim": embedding_dim
        }, model_path)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Модель загружена.")
else:
    print("Обучение модели...")
    train_model(model, dataloader, criterion, optimizer, epochs)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim
    }, model_path)

context = ["it", "is", "a", "truth"]
predicted_word = predict_next_word(model, context, vocab, inverse_vocab, device)
print(f"Следующее слово: {predicted_word}")

# https://jamesmalcolm.me/posts/next-word-prediction-pytorch/
