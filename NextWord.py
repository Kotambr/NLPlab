import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
from bs4 import BeautifulSoup
import re
import os

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
        print(f"Ошибка при получении текста: {e}")
        return ""

url_1 = 'https://www.gutenberg.org/cache/epub/67979/pg67979.txt'  # Текст Ницше
url_2 = 'https://www.gutenberg.org/cache/epub/75342/pg75342.txt'  # Текст Рассела

text_1 = get_text_from_url(url_1)
text_2 = get_text_from_url(url_2)

text = text_1 + text_2  # Объединяем тексты

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()
    return text.split()  

tokens = preprocess_text(text)

# Создаем словарь
vocab = list(set(tokens))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in word_to_idx.items()}

print(f"Размер словаря: {len(word_to_idx)}")

class WordPredictionDataset(Dataset):
    def __init__(self, tokens, context_size=2):
        self.tokens = tokens
        self.context_size = context_size

    def __len__(self):
        return len(self.tokens) - self.context_size

    def __getitem__(self, idx):
        context = self.tokens[idx:idx + self.context_size]
        target = self.tokens[idx + self.context_size]
        context_idxs = [word_to_idx[word] for word in context]
        target_idx = word_to_idx[target]
        return torch.tensor(context_idxs, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

context_size = 2  # Количество слов в контексте
dataset = WordPredictionDataset(tokens, context_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class WordPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(WordPredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * context_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)  # Преобразуем слова в эмбеддинги
        embeds = embeds.view(embeds.size(0), -1)  # Разворачиваем в один вектор
        out = self.fc(embeds)  # Прогоняем через линейный слой
        return out

model_path = "word_prediction_model.pth"

vocab_size = len(vocab)
embed_dim = 10

if os.path.exists(model_path):
    # Проверяем размер словаря
    checkpoint = torch.load(model_path)
    saved_vocab_size = checkpoint['embedding.weight'].size(0)
    if saved_vocab_size != len(vocab):
        print("Размер словаря изменился. Удаляем старую модель.")
        os.remove(model_path)
        # Обучение модели
        model = WordPredictionModel(vocab_size, embed_dim, context_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20
        for epoch in range(num_epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

        # Сохранение модели
        torch.save(model.state_dict(), model_path)
        print("Модель сохранена в файл.")
    else:
        # Загружаем модель
        model = WordPredictionModel(vocab_size, embed_dim, context_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Модель загружена из файла.")
else:
    # Обучение модели
    model = WordPredictionModel(vocab_size, embed_dim, context_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), model_path)
    print("Модель сохранена в файл.")

def predict_next_word(model, context):
    model.eval()
    try:
        context_idxs = [word_to_idx[word] for word in context]
        print(f"Контекстные слова: {context}")
        print(f"Индексы: {context_idxs}")
        context_tensor = torch.tensor(context_idxs, dtype=torch.long).unsqueeze(0)
    except KeyError as e:
        print(f"Ошибка: слово '{e.args[0]}' отсутствует в словаре.")
        return "<unknown>"

    with torch.no_grad():
        outputs = model(context_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        print(f"Предсказанный индекс: {predicted_idx}")

        if predicted_idx in idx_to_word:
            return idx_to_word[predicted_idx]
        else:
            print(f"Предсказанный индекс {predicted_idx} отсутствует в словаре.")
            return "<unknown>"


while True:
    print("\nВведите два слова для предсказания следующего (или 'exit' для выхода):")
    # print(f"Пример словаря (первые 10 слов): {list(word_to_idx.items())[:10]}")
    user_input = input().strip().lower()
    if user_input == "exit":
        break
    context = user_input.split()
    if len(context) != context_size:
        print(f"Пожалуйста, введите ровно {context_size} слова.")
        continue
    if any(word not in word_to_idx for word in context):
        print("Некоторые слова отсутствуют в словаре.")
        continue
    predicted_word = predict_next_word(model, context)
    print(f"Предсказанное следующее слово: {predicted_word}")