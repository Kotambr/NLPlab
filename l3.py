import numpy as np
import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    start = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[start:end]
    return text

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1-1] == token2[t2-1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                cost = 1
                if token1[t1-1].isspace() or token2[t2-1].isspace():
                    cost = 0.5
                
                distances[t1][t2] = min(a + cost, b + cost, c + cost)

    return distances[len(token1)][len(token2)]
text = get_text_from_url('https://www.gutenberg.org/cache/epub/14741/pg14741-images.html')
result = levenshteinDistanceDP(text[100:200], text[300:400])
print(result)
