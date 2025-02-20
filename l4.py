import random
import re
import requests
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_trigram_model(text):
    tokens = tokenize(text)
    trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens) - 2)]
    model = defaultdict(Counter)
    for w1, w2, w3 in trigrams:
        model[(w1, w2)][w3] += 1
    return model

def generate_sentence(model, seed, max_length=50):
    sentence = list(seed)
    for _ in range(max_length - 2):
        context = tuple(sentence[-2:])
        if context in model:
            next_word = random.choices(list(model[context].keys()), 
                                       list(model[context].values()))[0]
        else:
            next_word = random.choice(list(model.keys()))[0]
        sentence.append(next_word)
        if next_word in '.!?':
            break
    return ' '.join(sentence)

def laplace_smoothing(model, vocab_size):
    for context in model:
        total_count = sum(model[context].values())
        for word in model[context]:
            model[context][word] = (model[context][word] + 1) / (total_count + vocab_size)
    return model

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    start = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[start:end]
    return text

#  Ницше и Рассел
nietzsche_url = 'https://www.gutenberg.org/cache/epub/67979/pg67979.txt'
russell_url = 'https://www.gutenberg.org/cache/epub/75342/pg75342.txt'

nietzsche_text = get_text_from_url(nietzsche_url)
russell_text = get_text_from_url(russell_url)

nietzsche_model = build_trigram_model(nietzsche_text)
russell_model = build_trigram_model(russell_text)

# Применение сглаживания Лапласа
vocab_size = len(set(tokenize(nietzsche_text + russell_text)))
nietzsche_model = laplace_smoothing(nietzsche_model, vocab_size)
russell_model = laplace_smoothing(russell_model, vocab_size)

# Генерация диалога
seed = ('the', 'first', 'trigram')
for _ in range(2):
    nietzsche_sentence = generate_sentence(nietzsche_model, seed)
    print("Nietzsche:", nietzsche_sentence)
    seed = tuple(nietzsche_sentence.split()[-3:])
    
    russell_sentence = generate_sentence(russell_model, seed)
    print("Russell:", russell_sentence)
    seed = tuple(russell_sentence.split()[-3:])