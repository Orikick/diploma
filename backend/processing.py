import re
import numpy as np
import pandas as pd
from collections import defaultdict

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()

def generate_ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

def build_markov_chain(ngrams):
    model = defaultdict(lambda: defaultdict(int))

    for gram in ngrams:
        prefix, next_word = gram[:-1], gram[-1]
        prefix_str = " ".join(prefix)  # Перетворюємо tuple у рядок
        model[prefix_str][next_word] += 1

    return {k: dict(v) for k, v in model.items()}

def process_text(filepath, n=2):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    words = clean_text(text)
    ngrams = generate_ngrams(words, n)
    markov_chain = build_markov_chain(ngrams)
    
    return {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'n_grams': ngrams,
        'markov_chain': markov_chain
    }
