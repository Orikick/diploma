import numpy as np
from numba import jit, njit
import pandas as pd
from time import time
from scipy.optimize import curve_fit
from string import punctuation
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from sklearn.metrics import r2_score
import networkx as nx
import json
import traceback
import os
import tempfile
import shutil
import zipfile
import io
import string
import matplotlib.pyplot as plt
import random
import math
from scipy.fft import fft, fftfreq
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
length_updated = False
model = None
L = 0
V = 0
df = None

# Ukrainian text processing globals
VOWELS = set('аеєиіїоуюяёэы')
CONSONANTS = set('бвгґджзйклмнпрстфхцчшщь')
VOWELS_CV = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")

# Text preprocessing functions
def replace_letters(text):
    """Replace specific Ukrainian letters with phonetic equivalents"""
    result = []
    words = text.split()
    
    for word in words:
        new_word = ''
        i = 0
        while i < len(word):
            # Check current letter
            if word[i] in 'яюєї':
                if i > 0 and word[i-1] in CONSONANTS:
                    replacement = {'я': 'ьа', 'ю': 'ьу', 'є': 'ье', 'ї': 'ьі'}[word[i]]
                else:
                    replacement = {'я': 'йа', 'ю': 'йу', 'є': 'йе', 'ї': 'йі'}[word[i]]
                new_word += replacement
            else:
                new_word += word[i]
            i += 1
        # Replace "щ" with "шч"
        new_word = new_word.replace("щ", "шч")
        result.append(new_word)
    
    return ' '.join(result)

def preprocess_text(text):
    """Process text: lowercase, replace letters, remove punctuation and digits"""
    # Convert to lowercase
    text = text.lower()
    # Replace special Ukrainian letters
    text = replace_letters(text)
    
    # Replace hyphens and dashes with spaces
    text = text.replace("-", " ")
    text = text.replace("—", " ")
    
    # Remove punctuation and digits
    translator = str.maketrans("", "", string.punctuation + string.digits + "…" + "→")
    text = text.translate(translator)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    return text

# Syllable splitting functions
def is_vowel(char):
    """Check if character is a Ukrainian vowel"""
    vowels = 'аеєиіїоуюя'
    return char.lower() in vowels

def is_special_consonant_pair(text, index):
    """Check if consonants at index form дж or дз"""
    if index + 1 >= len(text):
        return False
    pair = text[index:index+2].lower()
    return pair in ['дж', 'дз']

def has_special_suffix(word, index):
    """Check if word has special suffix starting at index"""
    suffixes = ['ский', 'цкий', 'зкий']
    for suffix in suffixes:
        if index + len(suffix) <= len(word) and word[index:index+len(suffix)].lower() == suffix:
            return len(suffix)
    return 0

def remove_soft_sign(word):
    """Remove all soft signs (ь) from the word"""
    return word.replace('ь', '')

def should_keep_consonants_together(text, index):
    """Check if consonants at index should stay in the same syllable"""
    if index + 1 >= len(text):
        return False
    
    consonant_group = ""
    i = index
    while i < len(text) and not is_vowel(text[i]):
        consonant_group += text[i].lower()
        i += 1
        if len(consonant_group) >= 4:
            break
    
    inseparable_pairs = [
        'кл', 'шк', 'шн', 'дз', 'мй', 'кр', 'чк', 'зм', 'сн', 'гл', 'тл',
        'вл', 'бл', 'см', 'сл', 'зн', 'бн', 'тн', 'хн', 'сп', 'ст', 'нт',
        'бр', 'тр', 'шн', 'жн', 'хн', 'пл'
    ]
    
    three_consonants = [
        'стр', 'внт', 'шнь'
    ]
    
    # Check three-letter combinations
    if len(consonant_group) >= 3:
        for group in three_consonants:
            if consonant_group.startswith(group):
                return True
    
    # Check all possible pairs
    for i in range(len(consonant_group) - 1):
        pair = consonant_group[i:i+2]
        if pair in inseparable_pairs:
            return True
            
    return False

def split_into_syllables(word):
    """Split a Ukrainian word into syllables"""
    if not word:
        return []
    
    word = remove_soft_sign(word)
    syllables = []
    current_syllable = ""
    i = 0
    
    word = ''.join(c for c in word if c.isalpha() or c == "'")
    
    if not word:
        return []
    
    while i < len(word):
        if has_special_suffix(word, i) > 0:
            if current_syllable:
                syllables.append(current_syllable)
            syllables.append(word[i:i+has_special_suffix(word, i)])
            break
            
        while i < len(word) and not is_vowel(word[i]):
            if i < len(word) - 1 and (is_special_consonant_pair(word, i) or should_keep_consonants_together(word, i)):
                if current_syllable and any(is_vowel(c) for c in current_syllable):
                    syllables.append(current_syllable)
                    current_syllable = ""
                if should_keep_consonants_together(word, i):
                    group_end = i + 1
                    while group_end < len(word) and not is_vowel(word[group_end]):
                        if group_end + 1 < len(word) and should_keep_consonants_together(word, group_end):
                            group_end += 1
                        else:
                            break
                        group_end += 1
                    current_syllable += word[i:group_end+1]
                    i = group_end + 1
                else:
                    current_syllable += word[i:i+2]
                    i += 2
            else:
                current_syllable += word[i]
                i += 1
        
        if i < len(word):
            current_syllable += word[i]
            i += 1
            
            if i < len(word):
                next_vowel = i
                while next_vowel < len(word) and not is_vowel(word[next_vowel]):
                    next_vowel += 1
                
                if next_vowel < len(word):
                    consonants = word[i:next_vowel]
                    if should_keep_consonants_together(word, i) or len(consonants) == 1:
                        syllables.append(current_syllable)
                        current_syllable = consonants
                    else:
                        mid = i + (len(consonants) // 2)
                        current_syllable += word[i:mid]
                        syllables.append(current_syllable)
                        current_syllable = word[mid:next_vowel]
                    i = next_vowel
                    continue
            
            if current_syllable:
                syllables.append(current_syllable)
                current_syllable = ""
    
    if current_syllable:
        if syllables and not any(is_vowel(c) for c in current_syllable):
            syllables[-1] += current_syllable
        else:
            syllables.append(current_syllable)
    
    return syllables

def split_text_into_syllables(text):
    """Split all words in text into syllables"""
    words = text.split()
    result = []
    
    for word in words:
        # Separate punctuation from the word
        prefix = ''
        suffix = ''
        while word and not (word[0].isalpha() or word[0] == "'"):
            prefix += word[0]
            word = word[1:]
        while word and not (word[-1].isalpha() or word[-1] == "'"):
            suffix = word[-1] + suffix
            word = word[:-1]
        
        # Split the word into syllables
        if word:
            syllables = split_into_syllables(word)
            if syllables:
                processed_word = prefix + ' '.join(syllables) + suffix
            else:
                processed_word = prefix + word + suffix
        else:
            processed_word = prefix + suffix
            
        result.append(processed_word)
    
    return ' '.join(result)

def convert_to_cv(text):
    """Convert text to Consonant-Vowel (CV) sequence representation
    
    This function replaces vowels with 'v' and consonants with 'c', 
    while keeping other characters (spaces, punctuation) unchanged.
    """
    if not text:
        return ""
    
    # Remove soft signs
    text = text.replace("ь", "").replace("Ь", "")
    
    # Convert characters to CV representation
    result = []
    for char in text:
        if char in VOWELS_CV:
            result.append("v")
        elif re.match(r"[\u0410-\u044f]", char):  # Cyrillic letters range
            result.append("c")
        else:
            result.append(char)  # Keep non-letter characters as they are
    
    return "".join(result)

def remove_punctuation_for_words(data):
    # Split the text into words using regular expression
    words = re.findall(r'\b\w+(?:[-\']\w+)*\b', data)

    # Further process the words to handle special characters
    processed_words = []
    for word in words:
        # Handle special characters and dashes within words
        processed_word = re.split(r'[^a-zA-Z0-9\']', word)
        processed_words.extend(processed_word)

    # Filter out empty strings and lowercase each word
    processed_words = [word.lower() for word in processed_words if word]

    return processed_words


def remove_punctuation(data):
    temp = []
    start_time = time()
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    result = "".join(temp)
    end_time = time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
    return result


class Ngram(dict):
    def __init__(self, iterable=None):
        super(Ngram, self).__init__()
        self.fa = {}
        self.counts = {}
        self.sums = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1


def make_dataframe(model, fmin=0):
    filtered_data = list(
        filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))
    if 'new_ngram' not in filtered_data:
        filtered_data.append("new_ngram")
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram":
            data['F'][i] = sum(model[ngram].bool) if hasattr(model[ngram], "bool") else 0
            continue
            
        # Check if model[ngram] has a 'pos' attribute, if not, handle the error
        try:
            if hasattr(model[ngram], 'pos'):
                data["F"][i] = len(model[ngram].pos)
            else:
                # If model[ngram] is a dictionary without 'pos', try to get count another way
                data["F"][i] = sum(1 for _ in range(L) if model[ngram].get('bool', np.zeros(L, dtype=np.uint8))[_])
        except Exception as e:
            print(f"Error processing ngram {ngram}: {str(e)}")
            # Set a default value
            data["F"][i] = 0

    df = pd.DataFrame(data=data)
    return df


def make_markov_chain(data, order=1):
    global model, L, V
    model = dict()
    L = len(data) - order
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)
    model['new_ngram'].pos = []
    
    try:
        if order > 1:
            for i in range(L - 1):
                window = tuple(data[i: i + order])
                if window in model:
                    model[window].update([data[i + order]])
                    model[window].pos.append(i + 1)
                    model[window].bool[i] = 1
                else:
                    model[window] = Ngram([data[i + order]])
                    model[window].pos = []
                    model[window].pos.append(i + 1)
                    model[window].bool = np.zeros(L, dtype=np.uint8)
                    model[window].bool[i] = 1
                    model['new_ngram'].bool[i] = 1
                    model['new_ngram'].pos.append(i + 1)
        else:
            for i in range(L):
                if data[i] in model:
                    model[data[i]].update([data[i + order]])
                    model[data[i]].pos.append(i + order)
                    try:
                        model[data[i]].bool[i] = 1
                    except Exception:
                        print('Adding bool attribute to ngram')
                        model[data[i]].bool = np.zeros(L, dtype=np.uint8)
                        model[data[i]].bool[i] = 1
                else:
                    model[data[i]] = Ngram([data[i + order]])
                    model[data[i]].pos = []
                    model[data[i]].pos.append(i + order)
                    model[data[i]].bool = np.zeros(L, dtype=np.uint8)
                    model[data[i]].bool[i] = 1

                    model['new_ngram'].bool[i] = 1
                    model['new_ngram'].pos.append(i + order)

            # Connect the last word with the first one
            if data[L] in model:
                if not hasattr(model[data[L]], 'update'):
                    # Convert to Ngram object if it's a plain dictionary
                    old_dict = model[data[L]]
                    model[data[L]] = Ngram()
                    for k, v in old_dict.items():
                        model[data[L]][k] = v
                model[data[L]].update({data[0]: 1})
            else:
                model[data[L]] = Ngram({data[0]: 1})
                model[data[L]].pos = []
                model[data[L]].bool = np.zeros(L, dtype=np.uint8)

            # Connect the first word with the last one
            if data[0] in model:
                if not hasattr(model[data[0]], 'update'):
                    # Convert to Ngram object if it's a plain dictionary
                    old_dict = model[data[0]]
                    model[data[0]] = Ngram()
                    for k, v in old_dict.items():
                        model[data[0]][k] = v
                model[data[0]].update({data[L]: 1})
            else:
                model[data[0]] = Ngram({data[L]: 1})
                model[data[0]].pos = []
                model[data[0]].bool = np.zeros(L, dtype=np.uint8)
    except Exception as e:
        print(f"Error in make_markov_chain: {str(e)}")
        traceback.print_exc()
    
    # Ensure all model elements are Ngram objects with required attributes
    for key in list(model.keys()):
        if not isinstance(model[key], Ngram):
            print(f"Converting {key} to Ngram object")
            temp = model[key]
            model[key] = Ngram()
            if isinstance(temp, dict):
                for k, v in temp.items():
                    model[key][k] = v
            
        # Ensure all required attributes exist
        if not hasattr(model[key], 'pos'):
            model[key].pos = []
        if not hasattr(model[key], 'bool'):
            model[key].bool = np.zeros(L, dtype=np.uint8)
    
    V = len(model)


def calculate_distance(positions, L, option, ngram, min_type=1):
    # Make sure positions is always uint32 to maintain type consistency
    positions = np.array(positions, dtype=np.uint32)
    
    # If positions is empty, return an empty array of uint32
    if len(positions) == 0:
        return np.empty(0, dtype=np.uint32)
        
    # Ensure we have valid L
    if L <= 0:
        L = 1  # Set a minimum valid L
    
    if option == "no":
        return nbc(positions, min_type)
    if option == "ordinary":
        return obc(positions, L, min_type)
    if option == "periodic":
        return pbc(positions, L, ngram, min_type)
    # Default case
    return nbc(positions, min_type)


@jit(nopython=True)
def nbc(positions, min_type):
    number_of_pos = len(positions)
    if number_of_pos <= 1:
        # Create a new empty array of uint32 type instead of returning positions directly
        # This ensures consistent return type
        result = np.empty(0, dtype=np.uint32)
        return result
    
    min_corr = 1 if min_type == 0 else 0
    
    dt = np.empty(number_of_pos - 1, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = (positions[i + 1] - positions[i]) - min_corr
    return dt


@jit(nopython=True)
def obc(positions, L, min_type):
    number_of_pos = len(positions)
    
    # Handle edge case
    if number_of_pos == 0:
        return np.empty(0, dtype=np.uint32)
    
    min_corr = 1 if min_type == 0 else 0
    
    dt = np.empty(number_of_pos + 1, dtype=np.uint32)
    dt[0] = positions[0] - min_corr
    
    for i in range(number_of_pos - 1):
        dt[i + 1] = (positions[i + 1] - positions[i]) - min_corr
    
    dt[-1] = (L - positions[-1]) - min_corr
    return dt


@jit(nopython=True)         
def pbc(positions, L, ngram, min_type):
    number_of_pos = len(positions)
    
    # Handle edge case
    if number_of_pos == 0:
        return np.empty(0, dtype=np.uint32)
    
    dt = np.zeros(number_of_pos, dtype=np.uint32)

    min_corr = 1 if min_type == 0 else 0
    
    for i in range(number_of_pos - 1):
        dt[i] = (positions[i + 1] - positions[i]) - min_corr
    
    dt[-1] = (L - positions[-1] + positions[0]) - min_corr
    return dt

@jit(nopython=True)
def s(window):
    suma = 0
    for i in range(len(window)):
        suma += window[i]
    return suma


@njit(fastmath=True)
def mse(x):
    if len(x) == 0:
        return 0.0
    t = x.mean()
    st = np.mean(x ** 2)
    return np.sqrt(max(0, st - (t ** 2)))  # Avoid negative values under the square root


@jit(nopython=True, fastmath=True)
def R(x):
    if len(x) <= 1:
        return 0.0
    t = np.mean(x)
    # Avoid division by zero
    if t == 0:
        return 0.0
    ts = np.std(x)
    return ts / t


@njit(fastmath=True)
def make_windows(x, wi, l, wsh):
    sums = []
    for i in range(0, l - wi, wsh):
        sums.append(np.sum(x[i:i + wi]))
    return np.array(sums)


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)


def prepere_data(data, n, split):
    """Prepare data for analysis based on n-gram size and split method"""
    global L
    
    # Check for invalid inputs
    if data is None:
        print("Error: data is None")
        return None
    
    if not data:
        print("Error: data is empty")
        return None
        
    if n is None:
        print("Error: n is None")
        return None
    
    if n < 1:
        print(f"Error: invalid n value: {n}")
        return None
    
    if split not in ["word", "letter", "symbol"]:
        print(f"Error: invalid split method: {split}")
        return None
    
    try:
        temp_data = []
        
        # Process data based on n and split method
        if n == 1:
            if split == "word":
                try:
                    temp = []
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    data = re.sub(r'--', ' -', data)
                    processor = NgrammProcessor()
                    # Process text
                    processor.preprocess(data)
                    # Get words in text
                    words = processor.get_words()
                    
                    if not words:
                        print("Warning: NgrammProcessor returned empty words list")
                        return None
                    
                    for i in words:
                        temp.append(i)
                    
                    L = len(temp)
                    if L == 0:
                        print("Warning: processed word list is empty")
                        return None
                    
                    return temp
                except Exception as e:
                    print(f"Error in word processing: {str(e)}")
                    traceback.print_exc()
                    return None
                
            elif split == 'letter':
                try:
                    processed_data = remove_punctuation(data)
                    if not processed_data:
                        print("Warning: remove_punctuation returned empty result")
                        return None
                    
                    for i in processed_data:
                        for j in i:
                            if is_valid_letter(j):
                                continue
                            temp_data.append(j)
                    
                    L = len(temp_data)
                    if L == 0:
                        print("Warning: processed letter list is empty")
                        return None
                    
                    return temp_data
                except Exception as e:
                    print(f"Error in letter processing: {str(e)}")
                    traceback.print_exc()
                    return None
                
            elif split == 'symbol':
                try:
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    
                    for i in data:
                        for j in i:
                            if j == " ":
                                temp_data.append("space")
                                continue
                            elif i == "\n":
                                temp_data.append("space")
                                continue
                            elif i == "\ufeff":
                                temp_data.append("space")
                                continue
                            j = j.lower()
                            temp_data.append(j)
                    
                    L = len(temp_data)
                    if L == 0:
                        print("Warning: processed symbol list is empty")
                        return None
                    
                    return temp_data
                except Exception as e:
                    print(f"Error in symbol processing: {str(e)}")
                    traceback.print_exc()
                    return None
        
        elif n > 1:
            # Higher order n-grams
            if split == "word":
                try:
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    data = re.sub(r'--', ' -', data)
                    processor = NgrammProcessor()
                    # Process text
                    processor.preprocess(data)
                    # Get words in text
                    words = processor.get_words()
                    
                    if not words:
                        print("Warning: NgrammProcessor returned empty words list for n>1")
                        return None
                    
                    L = len(words)
                    if L < n:
                        print(f"Warning: text too short for n-gram size. Words: {L}, n: {n}")
                        return None
                    
                    for i in range(L - n + 1):
                        window = tuple(words[i:i + n])
                        temp_data.append(window)
                    
                    return temp_data
                except Exception as e:
                    print(f"Error in word n-gram processing: {str(e)}")
                    traceback.print_exc()
                    return None
                
            elif split == "letter":
                try:
                    processed_data = remove_punctuation(data.split())
                    processed_data = remove_empty_strings(processed_data)
                    
                    letter_data = []
                    for i in processed_data:
                        for j in i:
                            if is_valid_letter(j):
                                continue
                            letter_data.append(j)
                    
                    L = len(letter_data)
                    if L < n:
                        print(f"Warning: text too short for n-gram size. Letters: {L}, n: {n}")
                        return None
                    
                    for i in range(L - n + 1):
                        window = tuple(letter_data[i:i + n])
                        temp_data.append(window)
                    
                    return temp_data
                except Exception as e:
                    print(f"Error in letter n-gram processing: {str(e)}")
                    traceback.print_exc()
                    return None
                
            elif split == 'symbol':
                try:
                    symbol_data = []
                    data = re.sub(r'\n+', '\n', data)
                    data = re.sub(r'\n\s\s', '\n', data)
                    data = re.sub(r'﻿', '', data)
                    
                    for i in data:
                        for j in i:
                            if j == " ":
                                symbol_data.append("space")
                                continue
                            elif i == "\n":
                                symbol_data.append("space")
                                continue
                            elif i == "\ufeff":
                                symbol_data.append("space")
                                continue
                            j = j.lower()
                            symbol_data.append(j)
                    
                    L = len(symbol_data)
                    if L < n:
                        print(f"Warning: text too short for n-gram size. Symbols: {L}, n: {n}")
                        return None
                    
                    for i in range(L - n + 1):
                        window = tuple(symbol_data[i:i + n])
                        temp_data.append(window)
                    
                    return temp_data
                except Exception as e:
                    print(f"Error in symbol n-gram processing: {str(e)}")
                    traceback.print_exc()
                    return None
        
        # Return None for unsupported cases
        print(f"Unsupported configuration: n={n}, split={split}")
        return None
    
    except Exception as e:
        print(f"Unexpected error in prepere_data: {str(e)}")
        print(f"Parameters: n={n}, split={split}, data length={len(data) if data else 'None'}")
        traceback.print_exc()
        return None


def dfa(data, args):
    wi, wh, l = args
    
    # Handle edge cases
    if wi <= 0 or l <= 0 or wh <= 0:
        return np.empty(0, dtype=np.uint8), 0.0
    
    # Calculate range of values
    range_length = len(range(0, l - wi, wh))
    if range_length <= 0:
        return np.empty(0, dtype=np.uint8), 0.0
    
    count = np.zeros(range_length, dtype=np.uint8)
    
    try:
        for index, i in enumerate(range(0, l - wi, wh)):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))
        
        # Handle empty count array
        if len(count) == 0:
            return count, 0.0
            
        return count, mse(count)
    except Exception as e:
        print(f"Error in dfa function: {str(e)}")
        traceback.print_exc()
        return np.empty(0, dtype=np.uint8), 0.0

class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.dt = np.empty(0, dtype=np.uint32)  # Initialize with empty array
        self.R = 0
        self.a = 0
        self.b = 0
        self.temp_dfa = []
        self.goodness = 0
        self.wh, self.l = wh, l

    def func(self, w):
        try:
            self.count[w], self.dfa[w] = dfa(self.data, (w, self.wh, self.l))
        except Exception as e:
            print(f"Error in newNgram.func: {str(e)}")
            traceback.print_exc()
            # Set default values on error
            self.count[w] = np.empty(0, dtype=np.uint8)
            self.dfa[w] = 0.0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# NLP processor class
class NgrammProcessor:
    def __init__(self, ignore_punctuation: bool = True):
        self.ignore_punctuation = ignore_punctuation
        self.words = []

    def preprocess(self, text: str):
        # Remove punctuation if needed
        if self.ignore_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        mixed_array = text.split()
        real_strings = [item for item in mixed_array if isinstance(item, str) and not is_number(item)]
        self.words = real_strings

    def get_words(self, remove_empty_entries: bool = False) -> list:
        words = self.words
        if remove_empty_entries:
            words = [word for word in words if word]
        words = [word.lower() for word in words]
        return words


def is_valid_letter(char):
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    if is_number(char) or char in invalid_characters:
        return True
    else:
        return False


def remove_empty_strings(arr):
    return [item for item in arr if item != '\ufeff']


# Utility function for data conversion
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.uint8) or isinstance(obj, np.uint32) or isinstance(obj, np.float64):
        return obj.item()
    else:
        return str(obj)


# Helper to make ngram keys serializable for JSON
def make_keys_serializable(data):
    result = {}
    for key, value in data.items():
        if isinstance(key, tuple):
            key = " ".join(key)
        result[str(key)] = value
    return result


def calculate_window_params(text, n_size, split):
    """Calculate window parameters based on text length"""
    global L  # Keep track of the global L variable since it's used throughout the code
    
    # Store the original value to detect changes
    original_L = L
    
    try:
        # Try to prepare the data and check if it's valid
        data = prepere_data(text, n_size, split)
        
        if data is None:
            print(f"Warning: prepere_data returned None for text of length {len(text)}, n_size={n_size}, split={split}")
            return None
        
        if len(data) == 0:
            print(f"Warning: prepere_data returned empty data for text of length {len(text)}, n_size={n_size}, split={split}")
            return None
        
        # Verify that L was properly set by prepere_data
        if L <= 0:
            print(f"Warning: L was not set properly. L={L} after prepere_data")
            return None
        
        # Calculate window parameters
        try:
            wm = int(L / 20)
            if wm <= 0:
                print(f"Warning: Calculated wm is zero or negative: wm={wm}, L={L}")
                wm = 10  # Set a reasonable default
                
            w = int(wm / 20)
            if w <= 0:
                print(f"Warning: Calculated w is zero or negative: w={w}, wm={wm}, L={L}")
                w = 1  # Set a reasonable default
            
            return {
                "w": w,  # Min window
                "wh": w,  # Window shift
                "we": w,  # Window expansion
                "wm": wm,  # Max window
                "length": L  # Text length
            }
        except Exception as e:
            print(f"Error calculating window parameters: {str(e)}")
            print(f"Variables: L={L}, original L={original_L}")
            return None
        
    except Exception as e:
        print(f"Error in calculate_window_params: {str(e)}")
        print(f"Input parameters: text length={len(text)}, n_size={n_size}, split={split}")
        print(f"L value: {L}")
        traceback.print_exc()
        return None


def analyze_text(file_text, n_size, split, condition, f_min, w, wh, we, wm, definition, 
                min_type=1, do_preprocess=False, do_syllables=False, do_cv=False):
    """Core analysis function that processes text"""
    global data, L, V, model, df
    
    start_time = time()
    
    try:
        # Apply preprocessing if requested
        if do_preprocess:
            file_text = preprocess_text(file_text)
            
        # Split into syllables if requested
        if do_syllables:
            file_text = split_text_into_syllables(file_text)
            
        # Convert to CV sequence if requested
        if do_cv:
            file_text = convert_to_cv(file_text)
        
        data = prepere_data(file_text, n_size, split)
        
        if data is None:
            return {
                "error": "Failed to prepare data for analysis",
                "dataframe": [],
                "vocabulary": 0,
                "time": 0,
                "length": 0,
                "ngram_data": {}
            }
        
        if definition == "dynamic":
            windows = list(range(w, wm, we))
            new_ngram_obj = newNgram(data, wh, L)
            
            for w_val in windows:
                new_ngram_obj.func(w_val)
                
            temp_v = []
            temp_pos = []
            for i, ngram in enumerate(data):
                if ngram not in temp_v:
                    temp_v.append(ngram)
                    temp_pos.append(i)
                    
            new_ngram_obj.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), L, condition, ngram, min_type)
            new_ngram_obj.R = round(R(new_ngram_obj.dt), 8)
            c, _ = curve_fit(fit, [*new_ngram_obj.dfa.keys()], [*new_ngram_obj.dfa.values()], method='lm', maxfev=5000)
            new_ngram_obj.a = round(c[0], 8)
            new_ngram_obj.b = round(c[1], 8)
            new_ngram_obj.temp_dfa = []
            
            for w_val in new_ngram_obj.dfa.keys():
                new_ngram_obj.temp_dfa.append(fit(w_val, new_ngram_obj.a, new_ngram_obj.b))
                
            new_ngram_obj.goodness = round(r2_score([*new_ngram_obj.dfa.values()], new_ngram_obj.temp_dfa), 8)
            
            df = pd.DataFrame({
                'rank': [1],
                'ngram': ['new_ngram'],
                "F": [len(temp_pos)],
                'R': [new_ngram_obj.R],
                "a": [new_ngram_obj.a],
                "b": [new_ngram_obj.b],
                'goodness': [new_ngram_obj.goodness],
                'has_error': [False]  # No error
            })
            
            V = len(temp_v)
            
            # Create a serializable version of the new_ngram_obj for the response
            ngram_data = {
                "new_ngram": {
                    "R": new_ngram_obj.R,
                    "a": new_ngram_obj.a,
                    "b": new_ngram_obj.b,
                    "goodness": new_ngram_obj.goodness,
                    "dfa": make_keys_serializable(new_ngram_obj.dfa),
                    "bool": []  # No bool data for dynamic mode
                }
            }
    
        else:
            make_markov_chain(data, order=n_size)
            df = make_dataframe(model, f_min)
    
            # Initialize arrays to hold results
            temp_b = []
            temp_R = []
            temp_error = []
            temp_ngram = []
            temp_a = []
            error_flags = []  # New array to track error states
    
            for i, ngram in enumerate(df['ngram']):
                try:
                    # Calculate distance
                    model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_type)
    
                    # Process each window size
                    windows = list(range(w, wm, we))
                    model[ngram].counts = {}
                    model[ngram].fa = {}
                    
                    for wind in windows:
                        try:
                            model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh)
                            model[ngram].fa[wind] = mse(model[ngram].counts[wind])
                        except Exception as e:
                            print(f"Error processing window {wind} for ngram {ngram}: {str(e)}")
                            model[ngram].counts[wind] = np.array([])
                            model[ngram].fa[wind] = 0.0
    
                    # Calculate curve fit
                    model[ngram].temp_fa = []
                    ff = [*model[ngram].fa.values()]
                    
                    if len(ff) > 0 and not all(v == 0 for v in ff) and len(windows) > 0:
                        try:
                            c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                            model[ngram].a = c[0]
                            model[ngram].b = c[1]
                            
                            for w_val in windows:
                                model[ngram].temp_fa.append(fit(w_val, c[0], c[1]))
                            
                            # Calculate goodness of fit
                            goodness = round(r2_score(ff, model[ngram].temp_fa), 5)
                        except Exception as e:
                            # Curve fitting failed
                            print(f"Curve fit failed for {ngram}: {str(e)}")
                            c = [0, 0]
                            model[ngram].a = 0
                            model[ngram].b = 0
                            model[ngram].temp_fa = [0] * len(windows)
                            goodness = 0
                    else:
                        c = [0, 0]
                        model[ngram].a = 0
                        model[ngram].b = 0
                        model[ngram].temp_fa = []
                        goodness = 0
                    
                    # Calculate R value
                    r = round(R(np.array(model[ngram].dt)), 8)
                    model[ngram].R = r
                    
                    # Append results
                    temp_error.append(goodness)
                    temp_b.append(round(c[1], 8))
                    temp_a.append(round(c[0], 8))
                    
                    if isinstance(ngram, tuple):
                        temp_ngram.append(" ".join(ngram))
                    else:
                        temp_ngram.append(ngram)
                    
                    temp_R.append(r)
                    error_flags.append(False)  # No error for this n-gram
                    
                except Exception as e:
                    # Handle error for this n-gram
                    print(f"Error processing n-gram {ngram}: {str(e)}")
                    traceback.print_exc()
                    
                    # Add placeholder data
                    temp_error.append(0)
                    temp_b.append(0)
                    temp_a.append(0)
                    
                    if isinstance(ngram, tuple):
                        temp_ngram.append(" ".join(ngram))
                    else:
                        temp_ngram.append(ngram)
                    
                    temp_R.append(0)
                    error_flags.append(True)  # Mark this n-gram as having an error
                    
                    # Create basic structures for the model if they don't exist
                    if not hasattr(model[ngram], 'R'):
                        model[ngram].R = 0
                    if not hasattr(model[ngram], 'a'):
                        model[ngram].a = 0
                    if not hasattr(model[ngram], 'b'):
                        model[ngram].b = 0
                    if not hasattr(model[ngram], 'fa'):
                        model[ngram].fa = {}
                    if not hasattr(model[ngram], 'temp_fa'):
                        model[ngram].temp_fa = []
    
            # Update the dataframe with results
            # df['R'] = temp_R
            # df['b'] = temp_b
            # df['a'] = temp_a
            # df['goodness'] = temp_error
            # df['has_error'] = error_flags  # Add error flags to the dataframe
            
            # Create a new dataframe with all columns
            df_data = {
                "rank": list(range(1, len(temp_ngram) + 1)),
                "ngram": temp_ngram,
                "F": df["F"].tolist(),
                "R": temp_R,
                "a": temp_a,
                "b": temp_b,
                "goodness": temp_error,
                "has_error": error_flags
            }
            df = pd.DataFrame(df_data)
            
            df = df.sort_values(by="F", ascending=False)
            df['rank'] = range(1, len(df) + 1)
            df = df.set_index(pd.Index(np.arange(len(df))))
    
            # Create a dictionary of serializable model data
            ngram_data = {}
            for ngram in df['ngram']:
                if n_size > 1 and ngram != "new_ngram":
                    key = tuple(ngram.split())
                else:
                    key = ngram
                    
                row_data = df[df['ngram'] == ngram].iloc[0]
                has_error = row_data.get('has_error', False)
                
                if has_error:
                    # For n-grams with errors, provide minimal data
                    ngram_data[str(ngram)] = {
                        "R": 0,
                        "a": 0,
                        "b": 0,
                        "fa": {},
                        "temp_fa": [],
                        "bool": [],
                        "has_error": True
                    }
                else:
                    # For valid n-grams, provide full data
                    ngram_data[str(ngram)] = {
                        "R": getattr(model[key], 'R', 0),
                        "a": getattr(model[key], 'a', 0),
                        "b": getattr(model[key], 'b', 0),
                        "fa": make_keys_serializable(getattr(model[key], 'fa', {})),
                        "temp_fa": getattr(model[key], 'temp_fa', []),
                        "bool": model[key].bool.tolist() if hasattr(model[key], 'bool') else [],
                        "has_error": False
                    }
    
        # Calculate vocabulary size
        voc = str(V)
        voc = int(voc) - 1
        
        execution_time = round(time() - start_time, 4)
        
        return {
            "dataframe": df.to_dict(orient='records'),
            "vocabulary": voc,
            "time": execution_time,
            "length": L,
            "ngram_data": ngram_data
        }
        
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        traceback.print_exc()
        return {
            "error": f"Analysis failed: {str(e)}",
            "dataframe": [],
            "vocabulary": 0,
            "time": 0,
            "length": 0,
            "ngram_data": {}
        }


def generate_markov_graph(n_size):
    """Generate network graph of Markov chain"""
    global model, df
    
    if df is None:
        return {"error": "No data available. Please analyze text first."}
    
    try:
        # Create graph data
        g = nx.MultiGraph()
        temp = {}
        
        for ngram in df['ngram']:
            if n_size > 1 and ngram != "new_ngram":
                node = tuple(ngram.split())
            else:
                node = ngram
                
            g.add_node(node)
            if n_size > 1 and isinstance(node, tuple) and len(node) > 0:
                temp[node[0]] = node
        
        # Add edges
        for node in g.nodes():
            if node == "new_ngram":
                continue
                
            if n_size > 1 and isinstance(node, tuple):
                node_str = " ".join(node)
            else:
                node_str = node
                
            node_obj = model[node]
            
            for next_word, weight in node_obj.items():
                if isinstance(weight, int) and next_word in temp:
                    g.add_edge(node, temp[next_word], weight=weight)
        
        # Generate positions
        pos = nx.spring_layout(g)
        
        # Create node and edge data for visualization
        nodes = []
        for node in g.nodes():
            if isinstance(node, tuple):
                node_str = " ".join(node)
            else:
                node_str = str(node)
                
            nodes.append({
                "id": node_str,
                "connections": len(list(g.neighbors(node)))
            })
        
        edges = []
        for u, v, data in g.edges(data=True):
            if isinstance(u, tuple):
                u_str = " ".join(u)
            else:
                u_str = str(u)
                
            if isinstance(v, tuple):
                v_str = " ".join(v)
            else:
                v_str = str(v)
                
            edges.append({
                "source": u_str,
                "target": v_str,
                "weight": data.get("weight", 1)
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# Corpus processing functionality
def process_corpus(files, n_size, split, condition, min_type, fmin_for_lmin, fmin_for_lmax, 
                  w, wh, we, wm, definition, do_preprocess=False, do_syllables=False, do_cv=False):
    """Process a corpus of text files and return aggregated results"""
    global L, model, V, df
    
    # Prepare storage for results
    corpus_results = []
    length_info = []
    file_names = []
    
    # First pass: collect all file lengths to calculate appropriate F_min
    for file_name, file_content in files.items():
        try:
            # Apply preprocessing if requested
            processed_content = file_content
            if do_preprocess:
                processed_content = preprocess_text(processed_content)
            
            # Split into syllables if requested
            if do_syllables:
                processed_content = split_text_into_syllables(processed_content)
                
            # Convert to CV sequence if requested
            if do_cv:
                processed_content = convert_to_cv(processed_content)
                
            data = prepere_data(processed_content, n_size, split)
            if data:
                file_length = len(data)
                length_info.append({"file": file_name, "length": file_length})
                file_names.append(file_name)
        except Exception as e:
            print(f"Error in first pass for file {file_name}: {str(e)}")
            traceback.print_exc()
            # Still add the file to the list, but with a placeholder length
            length_info.append({"file": file_name, "length": 0, "error": True})
            file_names.append(file_name)
    
    # Find Lmin and Lmax from valid files
    sorted_by_len = sorted([item for item in length_info if item.get("length", 0) > 0], 
                           key=lambda x: x["length"])
    
    if not sorted_by_len:
        return {"error": "No valid files found in corpus"}
        
    Lmin_actual = sorted_by_len[0]["length"] if sorted_by_len else 0
    Lmax_actual = sorted_by_len[-1]["length"] if sorted_by_len else 0
    
    # Calculate F_min slope
    if Lmax_actual == Lmin_actual or Lmin_actual == 0:
        slope = 0
    else:
        slope = (fmin_for_lmax - fmin_for_lmin) / (Lmax_actual - Lmin_actual)
    
    # Process each file
    start_all = time()
    
    for i, item in enumerate(length_info, start=1):
        file_name = item["file"]
        file_length = item.get("length", 0)
        has_error = item.get("error", False)
        
        # Initialize metrics with default values
        R_avg = 0
        dR = 0
        Rw_avg = 0
        dRw = 0
        b_avg = 0
        db = 0
        bw_avg = 0
        dbw = 0
        Vcount = 0
        f_min_for_this = 0
        processing_time = 0
        
        try:
            if has_error:
                raise Exception("Error detected in first pass")
                
            file_content = files[file_name]
            
            # Apply preprocessing if requested
            processed_content = file_content
            if do_preprocess:
                processed_content = preprocess_text(processed_content)
            
            # Split into syllables if requested
            if do_syllables:
                processed_content = split_text_into_syllables(processed_content)
                
            # Convert to CV sequence if requested
            if do_cv:
                processed_content = convert_to_cv(processed_content)
            
            # Calculate F_min for this file
            f_min_for_this = fmin_for_lmin + slope * (file_length - Lmin_actual) if file_length > 0 else fmin_for_lmin
            f_min_for_this = round(f_min_for_this)
            
            # Start processing this file
            t0 = time()
            
            # Prepare data for analysis
            try:
                data = prepere_data(processed_content, n_size, split)
                
                if data is None:
                    raise Exception("Failed to prepare data")
                
                if len(data) == 0:
                    raise Exception("Empty data returned from prepere_data")
                
                # Dynamic or static analysis
                if definition == "dynamic":
                    # Dynamic analysis code
                    try:
                        windows = list(range(w, wm, we))
                        new_ngram_obj = newNgram(data, wh, file_length)
                        
                        for w_val in windows:
                            new_ngram_obj.func(w_val)
                            
                        temp_v = []
                        temp_pos = []
                        for idx, ngram in enumerate(data):
                            if ngram not in temp_v:
                                temp_v.append(ngram)
                                temp_pos.append(idx)
                        
                        if len(temp_pos) == 0:
                            raise Exception("No positions found for n-grams")
                            
                        new_ngram_obj.dt = calculate_distance(np.array(temp_pos, dtype=np.uint32), file_length, condition, ngram, min_type)
                        # Check if dt is not empty before calculating R
                        if len(new_ngram_obj.dt) > 0:
                            new_ngram_obj.R = round(R(new_ngram_obj.dt), 8)
                        else:
                            new_ngram_obj.R = 0  # Set default R value for empty dt
                        new_ngram_obj.R = round(R(new_ngram_obj.dt), 8) if len(new_ngram_obj.dt) > 0 else 0
                        
                        try:
                            dfa_keys = list(new_ngram_obj.dfa.keys())
                            dfa_values = list(new_ngram_obj.dfa.values())
                            
                            if len(dfa_keys) == 0 or len(dfa_values) == 0:
                                raise Exception("Empty DFA keys or values")
                                
                            c, _ = curve_fit(fit, dfa_keys, dfa_values, method='lm', maxfev=5000)
                            new_ngram_obj.a = round(c[0], 8)
                            new_ngram_obj.b = round(c[1], 8)
                            
                            new_ngram_obj.temp_dfa = []
                            for w_val in dfa_keys:
                                new_ngram_obj.temp_dfa.append(fit(w_val, new_ngram_obj.a, new_ngram_obj.b))
                                
                            new_ngram_obj.goodness = round(r2_score(dfa_values, new_ngram_obj.temp_dfa), 8)
                            
                            R_avg = new_ngram_obj.R
                            dR = 0
                            Rw_avg = R_avg
                            dRw = 0
                            b_avg = new_ngram_obj.b
                            db = 0
                            bw_avg = b_avg
                            dbw = 0
                            Vcount = len(temp_v)
                        except Exception as e:
                            print(f"Error in curve fitting for {file_name}: {str(e)}")
                            traceback.print_exc()
                            R_avg = 0
                            dR = 0
                            Rw_avg = 0
                            dRw = 0
                            b_avg = 0
                            db = 0
                            bw_avg = 0
                            dbw = 0
                            Vcount = len(temp_v) if temp_v else 0
                            
                    except Exception as e:
                        print(f"Error in dynamic analysis for {file_name}: {str(e)}")
                        traceback.print_exc()
                        raise
                        
                else:
                    # Static analysis
                    try:
                        # Create Markov chain
                        make_markov_chain(data, order=n_size)
                        
                        # Ensure all model elements have the proper attributes
                        for key in list(model.keys()):
                            if not isinstance(model[key], Ngram):
                                temp = model[key]
                                model[key] = Ngram()
                                if isinstance(temp, dict):
                                    for k, v in temp.items():
                                        model[key][k] = v
                                
                            # Ensure all required attributes exist
                            if not hasattr(model[key], 'pos'):
                                model[key].pos = []
                            if not hasattr(model[key], 'bool'):
                                model[key].bool = np.zeros(L, dtype=np.uint8)
                            if not hasattr(model[key], 'fa'):
                                model[key].fa = {}
                            if not hasattr(model[key], 'counts'):
                                model[key].counts = {}
                        
                        try:
                            # Create dataframe
                            df_onefile = make_dataframe(model, f_min_for_this)
                            
                            # Calculate R, b, goodness for each n-gram
                            windows = list(range(w, wm, we))
                            temp_b = []
                            temp_R = []
                            temp_error = []
                            temp_a = []
                            temp_ngram = []
                            
                            for ngram in df_onefile["ngram"]:
                                if ngram == "new_ngram":
                                    continue
                                    
                                try:
                                    # Calculate dt (distances)
                                    if not hasattr(model[ngram], 'pos') or not model[ngram].pos:
                                        model[ngram].pos = []
                                        
                                    if not hasattr(model[ngram], 'dt'):
                                        model[ngram].dt = np.array([], dtype=np.uint32)
                                    
                                    model[ngram].dt = calculate_distance(
                                        np.array(model[ngram].pos, dtype=np.uint32),
                                        file_length,
                                        condition,
                                        ngram,
                                        min_type
                                    )
                                    
                                    # Calculate fa for each window size
                                    for wind in windows:
                                        try:
                                            if not hasattr(model[ngram], 'bool') or model[ngram].bool is None:
                                                model[ngram].bool = np.zeros(file_length, dtype=np.uint8)
                                                
                                            count_arr = make_windows(model[ngram].bool, wi=wind, l=file_length, wsh=wh)
                                            
                                            if not hasattr(model[ngram], 'fa'):
                                                model[ngram].fa = {}
                                                
                                            if not hasattr(model[ngram], 'counts'):
                                                model[ngram].counts = {}
                                                
                                            if len(count_arr) == 0:
                                                model[ngram].fa[wind] = 0.0
                                            else:
                                                model[ngram].fa[wind] = mse(count_arr)
                                            model[ngram].counts[wind] = count_arr
                                        except Exception as e:
                                            print(f"Error in make_windows for {ngram}: {str(e)}")
                                            traceback.print_exc()
                                            model[ngram].fa[wind] = 0.0
                                            model[ngram].counts[wind] = np.array([])
                                    
                                    # Curve fitting
                                    try:
                                        if not hasattr(model[ngram], 'fa'):
                                            model[ngram].fa = {}
                                            
                                        ff = [*model[ngram].fa.values()]
                                        
                                        if len(ff) > 0 and not all(v == 0 for v in ff) and len(windows) > 0:
                                            try:
                                                c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                                                model[ngram].a = c[0]
                                                model[ngram].b = c[1]
                                                
                                                if not hasattr(model[ngram], 'temp_fa'):
                                                    model[ngram].temp_fa = []
                                                    
                                                model[ngram].temp_fa = [fit(w_val, c[0], c[1]) for w_val in windows]
                                                temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
                                                temp_b.append(round(c[1], 8))
                                                temp_a.append(round(c[0], 8))
                                            except Exception as e:
                                                print(f"Curve fit failed for {ngram}: {str(e)}")
                                                traceback.print_exc()
                                                model[ngram].a = 0
                                                model[ngram].b = 0
                                                temp_error.append(0)
                                                temp_b.append(0)
                                                temp_a.append(0)
                                        else:
                                            model[ngram].a = 0
                                            model[ngram].b = 0
                                            temp_error.append(0)
                                            temp_b.append(0)
                                            temp_a.append(0)
                                    except Exception as e:
                                        print(f"Error in curve_fit preparation for {ngram}: {str(e)}")
                                        traceback.print_exc()
                                        model[ngram].a = 0
                                        model[ngram].b = 0
                                        temp_error.append(0)
                                        temp_b.append(0)
                                        temp_a.append(0)
                                    
                                    # Calculate R
                                    try:
                                        if not hasattr(model[ngram], 'dt') or len(model[ngram].dt) == 0:
                                            model[ngram].dt = np.array([0], dtype=np.uint32)
                                            
                                        rr = round(R(np.array(model[ngram].dt)), 8)
                                        temp_R.append(rr)
                                        model[ngram].R = rr
                                    except Exception as e:
                                        print(f"Error calculating R for {ngram}: {str(e)}")
                                        traceback.print_exc()
                                        temp_R.append(0)
                                        model[ngram].R = 0
                                    
                                    # Format ngram
                                    if isinstance(ngram, tuple):
                                        temp_ngram.append(" ".join(ngram))
                                    else:
                                        temp_ngram.append(ngram)
                                except Exception as e:
                                    print(f"Error processing ngram {ngram} in file {file_name}: {str(e)}")
                                    traceback.print_exc()
                                    # Still add the ngram to maintain counts, but with placeholder values
                                    temp_error.append(0)
                                    temp_b.append(0)
                                    temp_a.append(0)
                                    temp_R.append(0)
                                    if isinstance(ngram, tuple):
                                        temp_ngram.append(" ".join(ngram))
                                    else:
                                        temp_ngram.append(ngram)
                            
                            # Add calculations to dataframe
                            if temp_R:
                                df_onefile["R"] = pd.Series(temp_R, index=df_onefile.index[:-1])  # Exclude new_ngram
                                df_onefile["b"] = pd.Series(temp_b, index=df_onefile.index[:-1])
                                df_onefile["a"] = pd.Series(temp_a, index=df_onefile.index[:-1])
                                df_onefile["goodness"] = pd.Series(temp_error, index=df_onefile.index[:-1])
                                
                                # Calculate weighted metrics
                                df_no_new = df_onefile[df_onefile["ngram"] != "new_ngram"].copy()
                                if not df_no_new.empty and "F" in df_no_new.columns:
                                    try:
                                        df_no_new["w"] = df_no_new["F"] / df_no_new["F"].sum() if df_no_new["F"].sum() > 0 else 0
                                        R_avg = df_no_new["R"].mean() if not df_no_new["R"].isna().all() else 0
                                        dR = df_no_new["R"].std() if not df_no_new["R"].isna().all() else 0
                                        df_no_new["Rw"] = df_no_new["R"] * df_no_new["w"]
                                        Rw_avg = df_no_new["Rw"].sum() if not df_no_new["Rw"].isna().all() else 0
                                        dRw = np.sqrt(((df_no_new["R"] - Rw_avg)**2 * df_no_new["w"]).sum()) if not df_no_new["R"].isna().all() else 0
                                        
                                        b_avg = df_no_new["b"].mean() if not df_no_new["b"].isna().all() else 0
                                        db = df_no_new["b"].std() if not df_no_new["b"].isna().all() else 0
                                        df_no_new["bw"] = df_no_new["b"] * df_no_new["w"]
                                        bw_avg = df_no_new["bw"].sum() if not df_no_new["bw"].isna().all() else 0
                                        dbw = np.sqrt(((df_no_new["b"] - bw_avg)**2 * df_no_new["w"]).sum()) if not df_no_new["b"].isna().all() else 0
                                        Vcount = df_no_new["ngram"].nunique()
                                    except Exception as e:
                                        print(f"Error calculating metrics for {file_name}: {str(e)}")
                                        traceback.print_exc()
                                        R_avg = 0
                                        dR = 0
                                        Rw_avg = 0
                                        dRw = 0
                                        b_avg = 0
                                        db = 0
                                        bw_avg = 0
                                        dbw = 0
                                        Vcount = 0
                                else:
                                    R_avg = 0
                                    dR = 0
                                    Rw_avg = 0
                                    dRw = 0
                                    b_avg = 0
                                    db = 0
                                    bw_avg = 0
                                    dbw = 0
                                    Vcount = 0
                            else:
                                R_avg = 0
                                dR = 0
                                Rw_avg = 0
                                dRw = 0
                                b_avg = 0
                                db = 0
                                bw_avg = 0
                                dbw = 0
                                Vcount = 0
                                
                        except Exception as e:
                            print(f"Error in make_dataframe for {file_name}: {str(e)}")
                            traceback.print_exc()
                            raise Exception(f"Failed to create dataframe: {str(e)}")
                            
                    except Exception as e:
                        print(f"Error in static analysis for {file_name}: {str(e)}")
                        traceback.print_exc()
                        raise
            
            except Exception as e:
                print(f"Error preparing data for {file_name}: {str(e)}")
                traceback.print_exc()
                raise
                
            # Time for this file
            processing_time = round(time() - t0, 4)
            
            # Add to results
            corpus_results.append({
                "№": i,
                "file": file_name,
                "F_min": f_min_for_this,
                "L": file_length,
                "V": Vcount,
                "time": processing_time,
                "R_avg": R_avg,
                "dR": dR,
                "Rw_avg": Rw_avg,
                "dRw": dRw,
                "b_avg": b_avg,
                "db": db,
                "bw_avg": bw_avg,
                "dbw": dbw,
                "has_error": False  # No error for this file
            })
            
        except Exception as e:
            # Handle error for this file
            print(f"Error processing file {file_name}: {str(e)}")
            traceback.print_exc()
            
            # Add placeholder result with error marker
            corpus_results.append({
                "№": i,
                "file": file_name,
                "F_min": "-",
                "L": "-",
                "V": "-",
                "time": "-",
                "R_avg": "-",
                "dR": "-",
                "Rw_avg": "-",
                "dRw": "-",
                "b_avg": "-",
                "db": "-",
                "bw_avg": "-",
                "dbw": "-",
                "has_error": True  # Mark this file as having an error
            })
    
    # Total processing time
    total_time = round(time() - start_all, 4)
    
    return {
        "corpus_results": corpus_results,
        "total_time": total_time,
        "file_count": len(corpus_results)
    }
# API Routes
@app.route('/api/calculate-windows', methods=['POST'])
def api_calculate_windows():
    """Calculate window parameters based on text length"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received in request", "context": "Request parsing"}), 400
        
        # Extract parameters
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Empty text provided. Please upload a valid text file", "context": "Input validation"}), 400
        
        try:
            n_size = int(data.get('n_size', 1))
            if n_size < 1:
                return jsonify({"error": f"Invalid n_size value: {n_size}. Must be at least 1", "context": "Parameter validation"}), 400
        except ValueError:
            return jsonify({"error": f"n_size must be an integer, got: {data.get('n_size')}", "context": "Parameter conversion"}), 400
        
        split = data.get('split', 'word')
        if split not in ['word', 'letter', 'symbol']:
            return jsonify({"error": f"Invalid split value: {split}. Must be 'word', 'letter', or 'symbol'", "context": "Parameter validation"}), 400
        
        # Process the data through intermediate steps to track errors
        try:
            # First try to prepare data
            prepared_data = prepere_data(text, n_size, split)
            if not prepared_data:
                return jsonify({
                    "error": "Failed to prepare data for analysis", 
                    "context": "prepere_data function",
                    "details": f"Function returned None with parameters: n_size={n_size}, split={split}, text_length={len(text)}"
                }), 400
            
            # Now calculate window parameters
            result = calculate_window_params(text, n_size, split)
            if not result:
                return jsonify({
                    "error": "Failed to calculate window parameters", 
                    "context": "calculate_window_params function",
                    "details": f"Data was prepared successfully (length: {len(prepared_data)}) but window calculation failed"
                }), 400
            
            return jsonify(result)
        
        except Exception as e:
            # Handle specific data processing errors
            error_location = "unknown"
            if "prepere_data" in str(e):
                error_location = "prepere_data function"
            elif "L =" in str(e):
                error_location = "global L variable assignment"
            elif "calculate_window_params" in str(e):
                error_location = "calculate_window_params function"
            
            return jsonify({
                "error": f"Data processing error: {str(e)}", 
                "context": error_location,
                "trace": traceback.format_exc(),
                "parameters": {
                    "n_size": n_size,
                    "split": split,
                    "text_length": len(text)
                }
            }), 500
    
    except Exception as e:
        # Catch-all for any other errors
        return jsonify({
            "error": f"Unexpected error: {str(e)}", 
            "context": "api_calculate_windows",
            "trace": traceback.format_exc()
        }), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Analyze text with the specified parameters"""
    try:
        data = request.get_json()
        
        # Get parameters from request
        file_text = data.get('text', '')
        n_size = int(data.get('n_size', 1))
        split = data.get('split', 'word')
        condition = data.get('condition', 'no')
        f_min = int(data.get('f_min', 0))
        w = int(data.get('w', 10))
        wh = int(data.get('wh', 10))
        we = int(data.get('we', 10))
        wm = int(data.get('wm', 100))
        definition = data.get('definition', 'static')
        min_type = int(data.get('min_type', 1))
        
        # Preprocessing options
        do_preprocess = data.get('do_preprocess', False)
        do_syllables = data.get('do_syllables', False)
        do_cv = data.get('do_cv', False)  # CV parameter
        
        # Process the text with new options
        result = analyze_text(
            file_text, n_size, split, condition, f_min, 
            w, wh, we, wm, definition, min_type, 
            do_preprocess, do_syllables, do_cv
        )
        
        # Check if there was an error with the entire analysis
        if "error" in result and result["error"]:
            return jsonify({
                "error": result["error"],
                "dataframe": [],
                "vocabulary": 0,
                "time": 0,
                "length": 0,
                "ngram_data": {}
            }), 400
        
        # Add information about preprocessing to the result
        result['preprocessing'] = {
            'text_preprocessed': do_preprocess,
            'text_syllabled': do_syllables,
            'text_cv_converted': do_cv
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "trace": traceback.format_exc(),
            "context": "Text analysis API"
        }), 500


@app.route('/api/markov-graph', methods=['POST'])
def api_markov_graph():
    """Generate Markov chain graph data"""
    try:
        data = request.get_json()
        n_size = int(data.get('n_size', 1))
        
        result = generate_markov_graph(n_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/api/upload-corpus', methods=['POST'])
def api_upload_corpus():
    """Handle corpus upload (either as a zip file or multiple individual files)"""
    try:
        # Get preprocessing options from form data
        do_preprocess = request.form.get('do_preprocess', 'false').lower() == 'true'
        do_syllables = request.form.get('do_syllables', 'false').lower() == 'true'
        do_cv = request.form.get('do_cv', 'false').lower() == 'true'  # CV parameter
        
        if 'zip_file' in request.files:
            # Process zip file
            zip_file = request.files['zip_file']
            
            # Create a temporary directory to extract files
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Extract zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Read all text files
                files = {}
                for root, _, file_names in os.walk(temp_dir):
                    for file_name in file_names:
                        if file_name.endswith('.txt'):
                            file_path = os.path.join(root, file_name)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    files[file_name] = f.read()
                            except UnicodeDecodeError:
                                try:
                                    with open(file_path, 'r', encoding='latin-1') as f:
                                        files[file_name] = f.read()
                                except Exception as e:
                                    print(f"Could not read file {file_name}: {str(e)}")
                
                if not files:
                    return jsonify({"error": "No .txt files found in the uploaded zip archive"}), 400
                    
                # Get parameters from form data
                n_size = int(request.form.get('n_size', 1))
                split = request.form.get('split', 'word')
                condition = request.form.get('condition', 'no')
                min_type = int(request.form.get('min_type', 1))
                fmin_for_lmin = int(request.form.get('fmin_for_lmin', 1))
                fmin_for_lmax = int(request.form.get('fmin_for_lmax', 5))
                w = int(request.form.get('w', 10))
                wh = int(request.form.get('wh', 10))
                we = int(request.form.get('we', 10))
                wm = int(request.form.get('wm', 100))
                definition = request.form.get('definition', 'static')
                
                # Process corpus with preprocessing options
                result = process_corpus(
                    files, n_size, split, condition, min_type,
                    fmin_for_lmin, fmin_for_lmax, w, wh, we, wm, definition,
                    do_preprocess, do_syllables, do_cv
                )
                
                # Add preprocessing info to the result
                result['preprocessing'] = {
                    'text_preprocessed': do_preprocess,
                    'text_syllabled': do_syllables,
                    'text_cv_converted': do_cv
                }
                
                return jsonify(result)
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                
        elif 'files[]' in request.files:
            # Process multiple individual files
            uploaded_files = request.files.getlist('files[]')
            
            files = {}
            for file in uploaded_files:
                if file.filename.endswith('.txt'):
                    try:
                        content = file.read().decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            content = file.read().decode('latin-1')
                        except Exception as e:
                            print(f"Could not decode file {file.filename}: {str(e)}")
                            continue
                    files[file.filename] = content
            
            if not files:
                return jsonify({"error": "No valid .txt files found in the uploaded files"}), 400
                
            # Get parameters from form data
            n_size = int(request.form.get('n_size', 1))
            split = request.form.get('split', 'word')
            condition = request.form.get('condition', 'no')
            min_type = int(request.form.get('min_type', 1))
            fmin_for_lmin = int(request.form.get('fmin_for_lmin', 1))
            fmin_for_lmax = int(request.form.get('fmin_for_lmax', 5))
            w = int(request.form.get('w', 10))
            wh = int(request.form.get('wh', 10))
            we = int(request.form.get('we', 10))
            wm = int(request.form.get('wm', 100))
            definition = request.form.get('definition', 'static')
            
            print(f"Processing corpus with parameters: n_size={n_size}, split={split}, condition={condition}, min_type={min_type}")
            
            # Process corpus with preprocessing options
            result = process_corpus(
                files, n_size, split, condition, min_type,
                fmin_for_lmin, fmin_for_lmax, w, wh, we, wm, definition,
                do_preprocess, do_syllables, do_cv
            )
            
            # Add preprocessing info to the result
            result['preprocessing'] = {
                'text_preprocessed': do_preprocess,
                'text_syllabled': do_syllables,
                'text_cv_converted': do_cv
            }
            
            return jsonify(result)
            
        else:
            return jsonify({"error": "No files were uploaded. Please upload a zip file or multiple text files"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/api/get-ngram-data', methods=['POST'])
def api_get_ngram_data():
    """Get detailed data for a specific ngram"""
    try:
        data = request.get_json()
        ngram = data.get('ngram')
        n_size = int(data.get('n_size', 1))
        
        if not ngram:
            return jsonify({"error": "Ngram parameter is required"}), 400
            
        if model is None:
            return jsonify({"error": "No analysis data available. Please analyze text first."}), 400
        
        # Get the appropriate key for the model
        if n_size > 1 and ngram != "new_ngram":
            key = tuple(ngram.split())
        else:
            key = ngram
            
        if key not in model:
            return jsonify({"error": f"Ngram '{ngram}' not found in the model"}), 404
            
        ngram_obj = model[key]
        
        # Create a serializable version of the ngram data
        result = {
            "positions": [int(pos) for pos in ngram_obj.pos] if hasattr(ngram_obj, 'pos') else [],
            "bool": ngram_obj.bool.tolist() if hasattr(ngram_obj, 'bool') else [],
            "R": getattr(ngram_obj, 'R', 0),
            "a": getattr(ngram_obj, 'a', 0),
            "b": getattr(ngram_obj, 'b', 0),
            "fa": {int(k): float(v) for k, v in ngram_obj.fa.items()} if hasattr(ngram_obj, 'fa') else {},
            "counts": {int(k): v.tolist() for k, v in ngram_obj.counts.items()} if hasattr(ngram_obj, 'counts') else {},
            "temp_fa": getattr(ngram_obj, 'temp_fa', []),
            "distribution": {str(k): v for k, v in ngram_obj.items()} if ngram_obj else {}
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

def compute_fourier_distribution(binary_sequence, shuffle=False):
    """
    Обчислює розподіл Фур'є для бінарної послідовності
    
    Args:
        binary_sequence (list): бінарна послідовність (0 і 1)
        shuffle (bool): чи перемішувати послідовність (для "шуму")
        
    Returns:
        tuple: (частоти, амплітуди)
    """
    # Перемішуємо, якщо необхідно
    if shuffle:
        sequence = binary_sequence.copy()
        random.shuffle(sequence)
    else:
        sequence = binary_sequence
    
    # Видаляємо DC-компоненту (середнє значення)
    sequence = np.array(sequence)
    sequence_centered = sequence - np.mean(sequence)
    
    # Обчислюємо FFT
    N = len(sequence_centered)
    yf = fft(sequence_centered)
    
    # Обчислюємо частоти і амплітуди, пропускаючи нульову частоту
    xf = fftfreq(N)[1:N//2]  # Починаємо з індексу 1, щоб пропустити DC
    amplitudes = 2.0/N * np.abs(yf[1:N//2])  # Також пропускаємо перший елемент
    
    return xf, amplitudes

def compute_fourier_distribution_waiting_times(waiting_times, shuffle=False):
    """
    Обчислює розподіл Фур'є для часів очікування з використанням формули Kobayashi-Musha
    
    Args:
        waiting_times (list): часи очікування
        shuffle (bool): чи перемішувати послідовність
        
    Returns:
        tuple: (частоти, амплітуди)
    """
    if not waiting_times or len(waiting_times) < 2:
        return np.array([]), np.array([])
    
    # Перетворюємо в рівномірний ряд згідно з формулою Kobayashi-Musha
    uniform_sequence = convert_to_uniform_sequence(waiting_times)
    
    # Перемішуємо після перетворення, якщо необхідно
    if shuffle:
        uniform_sequence = uniform_sequence.copy()
        np.random.shuffle(uniform_sequence)
    
    # Видаляємо DC-компоненту (середнє значення)
    uniform_sequence_centered = uniform_sequence - np.mean(uniform_sequence)
    
    # Обчислюємо FFT
    N = len(uniform_sequence_centered)
    yf = fft(uniform_sequence_centered)
    
    # Обчислюємо частоти і амплітуди, пропускаючи нульову частоту
    xf = fftfreq(N)[1:N//2]  # Починаємо з індексу 1
    amplitudes = 2.0/N * np.abs(yf[1:N//2])  # Пропускаємо перший елемент
    
    return xf, amplitudes

def generate_fourier_plots(binary_sequence, syllable_type, is_waiting_times=False):
    """
    Генерує графіки розподілів Фур'є та залежностей сигнал-шум (оновлена версія)
    
    Args:
        binary_sequence (list): бінарна послідовність (0 і 1)
        syllable_type (str): тип складу
        is_waiting_times (bool): чи це аналіз часів очікування
        
    Returns:
        dict: словник з base64-закодованими зображеннями графіків
    """
    # Обчислюємо розподіл Фур'є для сигналу
    xf_signal, amp_signal = compute_fourier_distribution(binary_sequence)
    
    # Обчислюємо розподіл Фур'є для шуму (перемішана послідовність)
    xf_noise, amp_noise = compute_fourier_distribution(binary_sequence, shuffle=True)
    
    # Перевіряємо, чи є дані для побудови графіків
    if len(xf_signal) == 0 or len(amp_signal) == 0:
        return {}
    
    # Обчислюємо відношення сигнал-шум
    epsilon = 1e-10
    signal_noise_ratio = amp_signal / (amp_noise + epsilon)
    
    # Створюємо графіки
    plots = {}
    
    # Графік розподілу Фур'є (сигнал)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, amp_signal, 'b-', label='Сигнал', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    title = f'Розподіл Фур\'є для типу складу "{syllable_type}" (без DC-компоненти)'
    if is_waiting_times:
        title = f'Розподіл Фур\'є для часів очікування (тип складу "{syllable_type}") (без DC-компоненти)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей для кращого відображення
    if len(amp_signal) > 0:
        ax.set_ylim(0, max(amp_signal) * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['fourier_signal'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік розподілу Фур'є (шум)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_noise, amp_noise, 'r-', label='Шум', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    title = f'Розподіл Фур\'є для шуму (перемішана послідовність) - тип складу "{syllable_type}" (без DC-компоненти)'
    if is_waiting_times:
        title = f'Розподіл Фур\'є для шуму (перемішані часи очікування) - тип складу "{syllable_type}" (без DC-компоненти)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей
    if len(amp_noise) > 0:
        ax.set_ylim(0, max(amp_noise) * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['fourier_noise'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік співвідношення сигнал-шум
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, signal_noise_ratio, 'g-', alpha=0.8, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Відношення сигнал/шум')
    title = f'Відношення сигнал/шум - тип складу "{syllable_type}" (без DC-компоненти)'
    if is_waiting_times:
        title = f'Відношення сигнал/шум для часів очікування - тип складу "{syllable_type}" (без DC-компоненти)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['signal_noise'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік порівняння сигналу і шуму
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, amp_signal, 'b-', label='Сигнал', alpha=0.7, linewidth=2)
    ax.plot(xf_noise, amp_noise, 'r-', label='Шум', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    title = f'Порівняння сигналу і шуму - тип складу "{syllable_type}" (без DC-компоненти)'
    if is_waiting_times:
        title = f'Порівняння сигналу і шуму для часів очікування - тип складу "{syllable_type}" (без DC-компоненти)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей
    max_amp = max(max(amp_signal) if len(amp_signal) > 0 else 0, 
                  max(amp_noise) if len(amp_noise) > 0 else 0)
    if max_amp > 0:
        ax.set_ylim(0, max_amp * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plots
def generate_rl_plots(corpus_results):
    """
    Генерує графіки залежностей R_avg(L) та Rw_avg(L) в лінійному та логарифмічному масштабах
    
    Args:
        corpus_results: результати аналізу корпусу текстів
        
    Returns:
        dict: словник з base64-закодованими зображеннями графіків
    """
    # Збір даних для графіків
    lengths = [result['L'] for result in corpus_results]
    r_avg_values = [result['R_avg'] for result in corpus_results]
    rw_avg_values = [result['Rw_avg'] for result in corpus_results]
    
    # Сортування за довжиною для коректного відображення
    sorted_data = sorted(zip(lengths, r_avg_values, rw_avg_values))
    lengths = [x[0] for x in sorted_data]
    r_avg_values = [x[1] for x in sorted_data]
    rw_avg_values = [x[2] for x in sorted_data]
    
    # Створення графіків
    plots = {}
    
    # 1. R_avg(L) - лінійний масштаб
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, r_avg_values, 'o-', color='blue', label='R_avg')
    ax.set_xlabel('L (довжина тексту)')
    ax.set_ylabel('R_avg')
    ax.set_title('Залежність R_avg від довжини тексту (лінійний масштаб)')
    ax.grid(True)
    ax.legend()
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['r_avg_linear'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # 2. R_avg(L) - логарифмічний масштаб
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lengths, r_avg_values, 'o-', color='blue', label='R_avg')
    ax.set_xlabel('L (довжина тексту)')
    ax.set_ylabel('R_avg')
    ax.set_title('Залежність R_avg від довжини тексту (логарифмічний масштаб)')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['r_avg_log'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # 3. Rw_avg(L) - лінійний масштаб
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, rw_avg_values, 'o-', color='green', label='Rw_avg')
    ax.set_xlabel('L (довжина тексту)')
    ax.set_ylabel('Rw_avg')
    ax.set_title('Залежність Rw_avg від довжини тексту (лінійний масштаб)')
    ax.grid(True)
    ax.legend()
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['rw_avg_linear'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # 4. Rw_avg(L) - логарифмічний масштаб
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lengths, rw_avg_values, 'o-', color='green', label='Rw_avg')
    ax.set_xlabel('L (довжина тексту)')
    ax.set_ylabel('Rw_avg')
    ax.set_title('Залежність Rw_avg від довжини тексту (логарифмічний масштаб)')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['rw_avg_log'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Додаємо графік з обома залежностями (R_avg та Rw_avg) в логарифмічному масштабі
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lengths, r_avg_values, 'o-', color='blue', label='R_avg')
    ax.loglog(lengths, rw_avg_values, 'o-', color='green', label='Rw_avg')
    ax.set_xlabel('L (довжина тексту)')
    ax.set_ylabel('Коефіцієнт варіації')
    ax.set_title('Порівняння R_avg та Rw_avg (логарифмічний масштаб)')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['both_log'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plots

# Додайте цей код до backend/main.py

@app.route('/api/rl-plots', methods=['POST'])
def api_rl_plots():
    """Generate R_avg(L) and Rw_avg(L) dependency plots"""
    try:
        data = request.get_json()
        corpus_results = data.get('corpus_results', [])
        
        if not corpus_results:
            return jsonify({"success": False, "error": "No corpus results provided"}), 400
        
        # Підготовка даних для графіків
        valid_results = []
        artificial_results = []
        
        for result in corpus_results:
            # Пропускаємо файли з помилками
            if result.get('has_error', False) or '-' in str(result.get('L', '-')):
                continue
                
            # Перетворення всіх значень до числових типів
            processed_result = {
                'file': result.get('file', ''),
                'L': float(result.get('L', 0)),
                'R_avg': float(result.get('R_avg', 0)),
                'Rw_avg': float(result.get('Rw_avg', 0)),
                'b_avg': float(result.get('b_avg', 0)),
                'bw_avg': float(result.get('bw_avg', 0))
            }
            
            # Визначення, чи є текст "штучним"
            file_name = processed_result['file'].lower()
            is_artificial = ('artificial' in file_name or 'synth' in file_name or 
                             'штучний' in file_name or 'синтет' in file_name or
                             'штучн' in file_name or 'artificial_' in file_name)
            
            if is_artificial:
                artificial_results.append(processed_result)
            else:
                valid_results.append(processed_result)
        
        # Перевірка наявності даних для побудови графіків
        if not valid_results and not artificial_results:
            return jsonify({"success": False, "error": "No valid data found in corpus results"}), 400
        
        # Імпортування необхідних бібліотек для графіків
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64
        
        # Функція для створення графіка
        def create_plot(x_data_regular, y_data_regular, x_data_artificial, y_data_artificial, 
                        title, xlabel, ylabel, log_x=False, log_y=False):
            plt.figure(figsize=(10, 6))
            
            # Побудова точок для звичайних текстів
            if x_data_regular and y_data_regular:
                if log_x and log_y:
                    plt.loglog(x_data_regular, y_data_regular, 'o', label='Звичайні тексти', color='blue')
                elif log_x:
                    plt.semilogx(x_data_regular, y_data_regular, 'o', label='Звичайні тексти', color='blue')
                elif log_y:
                    plt.semilogy(x_data_regular, y_data_regular, 'o', label='Звичайні тексти', color='blue')
                else:
                    plt.plot(x_data_regular, y_data_regular, 'o', label='Звичайні тексти', color='blue')
            
            # Побудова точок для штучних текстів
            if x_data_artificial and y_data_artificial:
                if log_x and log_y:
                    plt.loglog(x_data_artificial, y_data_artificial, '^', label='Штучні тексти', color='red')
                elif log_x:
                    plt.semilogx(x_data_artificial, y_data_artificial, '^', label='Штучні тексти', color='red')
                elif log_y:
                    plt.semilogy(x_data_artificial, y_data_artificial, '^', label='Штучні тексти', color='red')
                else:
                    plt.plot(x_data_artificial, y_data_artificial, '^', label='Штучні тексти', color='red')
            
            # Додавання тренду (лінії регресії) для всіх даних
            all_x = x_data_regular + x_data_artificial
            all_y = y_data_regular + y_data_artificial
            
            if all_x and all_y:
                if log_x and log_y:
                    # Регресія в логарифмічному масштабі
                    log_x_data = np.log10(all_x)
                    log_y_data = np.log10(all_y)
                    coeffs = np.polyfit(log_x_data, log_y_data, 1)
                    polynomial = np.poly1d(coeffs)
                    
                    # Побудова лінії тренду
                    x_range = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), 100)
                    log_x_range = np.log10(x_range)
                    log_y_range = polynomial(log_x_range)
                    y_range = 10 ** log_y_range
                    
                    plt.loglog(x_range, y_range, '--', color='green', 
                             label=f'Тренд: y = {10**coeffs[1]:.4f} * x^{coeffs[0]:.4f}')
                else:
                    # Звичайна лінійна регресія
                    coeffs = np.polyfit(all_x, all_y, 1)
                    polynomial = np.poly1d(coeffs)
                    
                    # Побудова лінії тренду
                    x_range = np.linspace(min(all_x), max(all_x), 100)
                    y_range = polynomial(x_range)
                    
                    if log_x:
                        plt.semilogx(x_range, y_range, '--', color='green', 
                                   label=f'Тренд: y = {coeffs[1]:.4f} + {coeffs[0]:.4f} * x')
                    elif log_y:
                        plt.semilogy(x_range, y_range, '--', color='green', 
                                   label=f'Тренд: y = {coeffs[1]:.4f} + {coeffs[0]:.4f} * x')
                    else:
                        plt.plot(x_range, y_range, '--', color='green', 
                               label=f'Тренд: y = {coeffs[1]:.4f} + {coeffs[0]:.4f} * x')
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Додавання анотацій для кожної точки
            for i, (x, y, result) in enumerate(zip(x_data_regular, y_data_regular, valid_results)):
                plt.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            for i, (x, y, result) in enumerate(zip(x_data_artificial, y_data_artificial, artificial_results)):
                plt.annotate(f"A{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
            
            # Побудова таблиці з легендою внизу графіка
            table_data = []
            for i, result in enumerate(valid_results):
                table_data.append([f"{i+1}", result['file'], f"{result['L']:.0f}", f"{result['R_avg']:.4f}", f"{result['Rw_avg']:.4f}"])
            
            for i, result in enumerate(artificial_results):
                table_data.append([f"A{i+1}", result['file'], f"{result['L']:.0f}", f"{result['R_avg']:.4f}", f"{result['Rw_avg']:.4f}"])
            
            # Збереження графіка в байтовий потік
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Конвертація зображення в base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
            
            return img_str
        
        # Створення різних графіків
        plots = {}
        
        # Підготовка даних
        x_regular = [r['L'] for r in valid_results]
        y_r_avg_regular = [r['R_avg'] for r in valid_results]
        y_rw_avg_regular = [r['Rw_avg'] for r in valid_results]
        
        x_artificial = [r['L'] for r in artificial_results]
        y_r_avg_artificial = [r['R_avg'] for r in artificial_results]
        y_rw_avg_artificial = [r['Rw_avg'] for r in artificial_results]
        
        # Генерація всіх графіків
        plots['r_avg_linear'] = create_plot(
            x_regular, y_r_avg_regular, 
            x_artificial, y_r_avg_artificial,
            'Залежність R_avg від довжини тексту (лінійний масштаб)',
            'Довжина тексту (L)', 'R_avg'
        )
        
        plots['r_avg_log'] = create_plot(
            x_regular, y_r_avg_regular, 
            x_artificial, y_r_avg_artificial,
            'Залежність R_avg від довжини тексту (логарифмічний масштаб)',
            'Довжина тексту (L)', 'R_avg', 
            log_x=True, log_y=True
        )
        
        plots['rw_avg_linear'] = create_plot(
            x_regular, y_rw_avg_regular, 
            x_artificial, y_rw_avg_artificial,
            'Залежність Rw_avg від довжини тексту (лінійний масштаб)',
            'Довжина тексту (L)', 'Rw_avg'
        )
        
        plots['rw_avg_log'] = create_plot(
            x_regular, y_rw_avg_regular, 
            x_artificial, y_rw_avg_artificial,
            'Залежність Rw_avg від довжини тексту (логарифмічний масштаб)',
            'Довжина тексту (L)', 'Rw_avg', 
            log_x=True, log_y=True
        )
        
        # Графік, що показує обидва параметри в логарифмічному масштабі
        plt.figure(figsize=(10, 6))
        
        if x_regular:
            plt.loglog(x_regular, y_r_avg_regular, 'o', label='R_avg (звичайні тексти)', color='blue')
            plt.loglog(x_regular, y_rw_avg_regular, 's', label='Rw_avg (звичайні тексти)', color='green')
        
        if x_artificial:
            plt.loglog(x_artificial, y_r_avg_artificial, '^', label='R_avg (штучні тексти)', color='red')
            plt.loglog(x_artificial, y_rw_avg_artificial, 'd', label='Rw_avg (штучні тексти)', color='orange')
        
        plt.title('Порівняння залежностей R_avg та Rw_avg (логарифмічний масштаб)')
        plt.xlabel('Довжина тексту (L)')
        plt.ylabel('Значення R_avg / Rw_avg')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Збереження графіка
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plots['both_log'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()
        
        return jsonify({
            "success": True,
            "plots": plots,
            "stats": {
                "valid_count": len(valid_results),
                "artificial_count": len(artificial_results)
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False, 
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    """Preprocess text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        do_preprocess = data.get('do_preprocess', False)
        do_syllables = data.get('do_syllables', False)
        do_cv = data.get('do_cv', False)  # New parameter for CV conversion
        
        result = text
        
        # Apply preprocessing if requested
        if do_preprocess:
            result = preprocess_text(result)
            
        # Split into syllables if requested
        if do_syllables:
            result = split_text_into_syllables(result)
            
        # Convert to CV sequence if requested
        if do_cv:
            result = convert_to_cv(result)
        
        return jsonify({
            "original_text": text,
            "processed_text": result
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "trace": traceback.format_exc(),
            "context": "Text preprocessing API"
        }), 500

def analyze_syllable_types(text):
    """
    Аналізує типи складів у тексті та повертає найчастіші типи
    
    Args:
        text (str): вхідний текст
        
    Returns:
        dict: інформація про типи складів та їх частоти
    """
    # Препроцесинг тексту
    processed_text = preprocess_text(text)
    
    # Розбиття на склади
    syllable_text = split_text_into_syllables(processed_text)
    
    # Конвертація у CV-послідовність
    cv_text = convert_to_cv(syllable_text)
    
    # Розділення на склади (кожен склад - це одиниця CV-паттерну між пробілами)
    syllables = cv_text.split()
    
    # Підрахунок частот типів складів
    syllable_counts = {}
    for syllable in syllables:
        if syllable:  # Пропускаємо порожні рядки
            syllable_counts[syllable] = syllable_counts.get(syllable, 0) + 1
    
    # Сортування за частотою
    sorted_syllables = sorted(syllable_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Отримуємо топ-2 найчастіших
    most_frequent = sorted_syllables[0] if len(sorted_syllables) > 0 else (None, 0)
    second_frequent = sorted_syllables[1] if len(sorted_syllables) > 1 else (None, 0)
    
    return {
        'syllables': syllables,
        'syllable_counts': syllable_counts,
        'most_frequent': {
            'type': most_frequent[0],
            'count': most_frequent[1]
        },
        'second_frequent': {
            'type': second_frequent[0],
            'count': second_frequent[1]
        },
        'total_syllables': len(syllables)
    }

def create_binary_sequence(syllables, target_syllable_type):
    """
    Створює бінарну послідовність позицій target_syllable_type у списку складів
    
    Args:
        syllables (list): список складів
        target_syllable_type (str): тип складу для пошуку
        
    Returns:
        list: бінарна послідовність (1 якщо склад відповідає типу, 0 інакше)
    """
    binary_sequence = []
    for syllable in syllables:
        binary_sequence.append(1 if syllable == target_syllable_type else 0)
    return binary_sequence

def process_file_fourier_analysis(filename, file_content):
    """
    Обробляє один файл для аналізу Фур'є
    
    Args:
        filename (str): ім'я файлу
        file_content (str): вміст файлу
        
    Returns:
        dict: результати аналізу файлу
    """
    try:
        # Аналізуємо типи складів
        syllable_analysis = analyze_syllable_types(file_content)
        
        # Створюємо бінарні послідовності для найчастіших типів складів
        most_frequent_type = syllable_analysis['most_frequent']['type']
        second_frequent_type = syllable_analysis['second_frequent']['type']
        
        result = {
            'filename': filename,
            'text_length': len(file_content),
            'syllables_count': syllable_analysis['total_syllables'],
            'most_frequent_syllable': syllable_analysis['most_frequent'],
            'second_frequent_syllable': syllable_analysis['second_frequent']
        }
        
        # Генеруємо графіки Фур'є для найчастішого типу складу
        if most_frequent_type:
            binary_seq_most = create_binary_sequence(syllable_analysis['syllables'], most_frequent_type)
            if sum(binary_seq_most) > 0:  # Перевіряємо, що є хоча б одне входження
                fourier_plots_most = generate_fourier_plots(binary_seq_most, most_frequent_type)
                result['fourier_plots'] = fourier_plots_most
                result['binary_sequence_length'] = len(binary_seq_most)
                result['target_syllable_occurrences'] = sum(binary_seq_most)
        
        # Додаємо аналіз для другого найчастішого типу
        if second_frequent_type:
            binary_seq_second = create_binary_sequence(syllable_analysis['syllables'], second_frequent_type)
            if sum(binary_seq_second) > 0:
                fourier_plots_second = generate_fourier_plots(binary_seq_second, second_frequent_type)
                result['fourier_plots_second'] = fourier_plots_second
        
        result['error'] = False
        return result
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        traceback.print_exc()
        return {
            'filename': filename,
            'error': True,
            'error_message': str(e),
            'text_length': len(file_content),
            'syllables_count': 0,
            'most_frequent_syllable': {'type': None, 'count': 0},
            'second_frequent_syllable': {'type': None, 'count': 0}
        }

def convert_to_uniform_sequence(intervals):
    """
    Перетворює нерівномірні інтервали в рівномірний ряд за формулою Kobayashi-Musha
    
    Args:
        intervals (list): список інтервалів часу
        
    Returns:
        numpy.array: рівномірний ряд висот імпульсів
    """
    if len(intervals) == 0:
        return np.array([])
    
    N = len(intervals)
    T = sum(intervals)  # загальна довжина часу
    Delta_T = T / N     # середній інтервал часу
    
    # Застосовуємо формулу Kobayashi-Musha (1):
    # h_j = (T_{j+1} - T_j)((j - 1) · ΔT - Σ_{k=0}^{j-1} T_k)/T_j + T_j
    
    h = np.zeros(N)
    cumsum_T = 0
    
    for j in range(N):
        T_j = intervals[j]
        
        # Розраховуємо кумулятивну суму до j-1
        if j > 0:
            cumsum_T += intervals[j-1]
        
        # Застосовуємо формулу
        if j < N - 1:
            T_j_plus_1 = intervals[j + 1]
        else:
            T_j_plus_1 = T_j  # для останнього елементу
            
        h[j] = (T_j_plus_1 - T_j) * ((j * Delta_T - cumsum_T) / T_j) + T_j
    
    return h

def analyze_syllable_waiting_times(text):
    """
    Аналізує часи очікування для найчастіших типів складів у тексті
    
    Args:
        text (str): вхідний текст
        
    Returns:
        dict: інформація про типи складів, їх позиції та часи очікування
    """
    # Препроцесинг тексту
    processed_text = preprocess_text(text)
    
    # Розбиття на склади
    syllable_text = split_text_into_syllables(processed_text)
    
    # Конвертація у CV-послідовність
    cv_text = convert_to_cv(syllable_text)
    
    # Розділення на склади
    syllables = cv_text.split()
    
    # Підрахунок частот типів складів
    syllable_counts = {}
    for syllable in syllables:
        if syllable:
            syllable_counts[syllable] = syllable_counts.get(syllable, 0) + 1
    
    # Сортування за частотою
    sorted_syllables = sorted(syllable_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Отримуємо топ-2 найчастіших
    most_frequent = sorted_syllables[0] if len(sorted_syllables) > 0 else (None, 0)
    second_frequent = sorted_syllables[1] if len(sorted_syllables) > 1 else (None, 0)
    
    # Знаходимо позиції найчастіших типів складів
    result = {
        'syllables': syllables,
        'syllable_counts': syllable_counts,
        'most_frequent': {
            'type': most_frequent[0],
            'count': most_frequent[1],
            'positions': [],
            'waiting_times': []
        },
        'second_frequent': {
            'type': second_frequent[0],
            'count': second_frequent[1],
            'positions': [],
            'waiting_times': []
        },
        'total_syllables': len(syllables)
    }
    
    # Знаходимо позиції та розраховуємо часи очікування
    for syllable_info in [result['most_frequent'], result['second_frequent']]:
        if syllable_info['type']:
            # Знаходимо всі позиції цього типу складу
            positions = []
            for i, syllable in enumerate(syllables):
                if syllable == syllable_info['type']:
                    positions.append(i)
            
            syllable_info['positions'] = positions
            
            # Розраховуємо часи очікування (інтервали між входженнями)
            if len(positions) > 1:
                waiting_times = []
                for i in range(1, len(positions)):
                    waiting_times.append(positions[i] - positions[i-1])
                syllable_info['waiting_times'] = waiting_times
            else:
                syllable_info['waiting_times'] = []
    
    return result

def create_binary_sequence_waiting_times(waiting_times):
    """
    Створює бінарну послідовність для часів очікування
    Кожен час очікування представляється як послідовність нулів завершена одиницею
    
    Args:
        waiting_times (list): список часів очікування
        
    Returns:
        list: бінарна послідовність
    """
    if not waiting_times:
        return []
    
    binary_sequence = []
    for wt in waiting_times:
        # Додаємо wt-1 нулів та одну одиниці
        binary_sequence.extend([0] * (wt - 1))
        binary_sequence.append(1)
    
    return binary_sequence

def compute_fourier_distribution_waiting_times(waiting_times, shuffle=False):
    """
    Обчислює розподіл Фур'є для часів очікування з використанням формули Kobayashi-Musha
    
    Args:
        waiting_times (list): часи очікування
        shuffle (bool): чи перемішувати послідовність
        
    Returns:
        tuple: (частоти, амплітуди)
    """
    if not waiting_times or len(waiting_times) < 2:
        return np.array([]), np.array([])
    
    # Перетворюємо в рівномірний ряд згідно з формулою Kobayashi-Musha
    uniform_sequence = convert_to_uniform_sequence(waiting_times)
    
    # Перемішуємо після перетворення, якщо необхідно
    if shuffle:
        uniform_sequence = uniform_sequence.copy()
        np.random.shuffle(uniform_sequence)
    
    # Видаляємо DC-компоненту (середнє значення)
    uniform_sequence_centered = uniform_sequence - np.mean(uniform_sequence)
    
    # Обчислюємо FFT
    N = len(uniform_sequence_centered)
    yf = fft(uniform_sequence_centered)
    
    # Обчислюємо частоти і амплітуди, пропускаючи нульову частоту
    xf = fftfreq(N)[1:N//2]  # Починаємо з індексу 1
    amplitudes = 2.0/N * np.abs(yf[1:N//2])  # Пропускаємо перший елемент
    
    return xf, amplitudes

def generate_fourier_plots_waiting_times(waiting_times, syllable_type):
    """
    Генерує графіки розподілів Фур'є та залежностей сигнал-шум для часів очікування (оновлена версія)
    
    Args:
        waiting_times (list): часи очікування
        syllable_type (str): тип складу
        
    Returns:
        dict: словник з base64-закодованими зображеннями графіків
    """
    if not waiting_times or len(waiting_times) < 2:
        return {}
    
    # Обчислюємо розподіл Фур'є для сигналу (часи очікування)
    xf_signal, amp_signal = compute_fourier_distribution_waiting_times(waiting_times)
    
    # Обчислюємо розподіл Фур'є для шуму (перемішані після перетворення)
    xf_noise, amp_noise = compute_fourier_distribution_waiting_times(waiting_times, shuffle=True)
    
    # Перевіряємо, чи є дані для побудови графіків
    if len(xf_signal) == 0 or len(amp_signal) == 0:
        return {}
    
    # Обчислюємо відношення сигнал-шум
    epsilon = 1e-10
    signal_noise_ratio = amp_signal / (amp_noise + epsilon)
    
    # Створюємо графіки
    plots = {}
    
    # Графік розподілу Фур'є (сигнал) - часи очікування
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, amp_signal, 'b-', label='Сигнал (часи очікування)', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    ax.set_title(f'Розподіл Фур\'є для часів очікування (тип складу "{syllable_type}") (без DC-компоненти)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей
    if len(amp_signal) > 0:
        ax.set_ylim(0, max(amp_signal) * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['fourier_signal'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік розподілу Фур'є (шум) - часи очікування
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_noise, amp_noise, 'r-', label='Шум (перемішані часи очікування)', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    ax.set_title(f'Розподіл Фур\'є для шуму часів очікування (тип складу "{syllable_type}") (без DC-компоненти)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей
    if len(amp_noise) > 0:
        ax.set_ylim(0, max(amp_noise) * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['fourier_noise'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік співвідношення сигнал-шум для часів очікування
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, signal_noise_ratio, 'g-', alpha=0.8, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Відношення сигнал/шум')
    ax.set_title(f'Відношення сигнал/шум для часів очікування (тип складу "{syllable_type}") (без DC-компоненти)')
    ax.grid(True, alpha=0.3)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['signal_noise'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Графік порівняння сигналу і шуму для часів очікування
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_signal, amp_signal, 'b-', label='Сигнал (часи очікування)', alpha=0.7, linewidth=2)
    ax.plot(xf_noise, amp_noise, 'r-', label='Шум (перемішані часи очікування)', alpha=0.7, linewidth=2)
    ax.set_xlabel('Частота')
    ax.set_ylabel('Амплітуда')
    ax.set_title(f'Порівняння сигналу і шуму для часів очікування (тип складу "{syllable_type}") (без DC-компоненти)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Встановлюємо обмеження осей
    max_amp = max(max(amp_signal) if len(amp_signal) > 0 else 0, 
                  max(amp_noise) if len(amp_noise) > 0 else 0)
    if max_amp > 0:
        ax.set_ylim(0, max_amp * 1.1)
    
    # Конвертація в base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plots
def process_file_fourier_analysis_extended(filename, file_content):
    """
    Розширена обробка файлу для аналізу Фур'є з часами очікування
    
    Args:
        filename (str): ім'я файлу
        file_content (str): вміст файлу
        
    Returns:
        dict: результати аналізу файлу
    """
    try:
        # Аналізуємо типи складів з часами очікування
        syllable_analysis = analyze_syllable_waiting_times(file_content)
        
        # Створюємо бінарні послідовності для найчастіших типів складів
        most_frequent_type = syllable_analysis['most_frequent']['type']
        second_frequent_type = syllable_analysis['second_frequent']['type']
        
        result = {
            'filename': filename,
            'text_length': len(file_content),
            'syllables_count': syllable_analysis['total_syllables'],
            'most_frequent_syllable': syllable_analysis['most_frequent'],
            'second_frequent_syllable': syllable_analysis['second_frequent']
        }
        
        # Генеруємо графіки Фур'є для найчастішого типу складу (позиції)
        if most_frequent_type:
            binary_seq_most = create_binary_sequence(syllable_analysis['syllables'], most_frequent_type)
            if sum(binary_seq_most) > 0:
                # Спочатку перетворюємо, потім перемішуємо для шуму
                fourier_plots_most = generate_fourier_plots(binary_seq_most, most_frequent_type)
                result['fourier_plots'] = fourier_plots_most
                result['binary_sequence_length'] = len(binary_seq_most)
                result['target_syllable_occurrences'] = sum(binary_seq_most)
            
            # Генеруємо графіки Фур'є для часів очікування найчастішого типу
            waiting_times_most = syllable_analysis['most_frequent']['waiting_times']
            if waiting_times_most and len(waiting_times_most) > 1:
                fourier_plots_wt_most = generate_fourier_plots_waiting_times(waiting_times_most, most_frequent_type)
                result['fourier_plots_waiting_times'] = fourier_plots_wt_most
                result['waiting_times_count'] = len(waiting_times_most)
        
        # Додаємо аналіз для другого найчастішого типу
        if second_frequent_type:
            binary_seq_second = create_binary_sequence(syllable_analysis['syllables'], second_frequent_type)
            if sum(binary_seq_second) > 0:
                fourier_plots_second = generate_fourier_plots(binary_seq_second, second_frequent_type)
                result['fourier_plots_second'] = fourier_plots_second
            
            # Генеруємо графіки Фур'є для часів очікування другого найчастішого типу
            waiting_times_second = syllable_analysis['second_frequent']['waiting_times']
            if waiting_times_second and len(waiting_times_second) > 1:
                fourier_plots_wt_second = generate_fourier_plots_waiting_times(waiting_times_second, second_frequent_type)
                result['fourier_plots_waiting_times_second'] = fourier_plots_wt_second
        
        result['error'] = False
        return result
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        traceback.print_exc()
        return {
            'filename': filename,
            'error': True,
            'error_message': str(e),
            'text_length': len(file_content),
            'syllables_count': 0,
            'most_frequent_syllable': {'type': None, 'count': 0, 'waiting_times': []},
            'second_frequent_syllable': {'type': None, 'count': 0, 'waiting_times': []}
        }

@app.route('/api/fourier-analysis', methods=['POST'])
def api_fourier_analysis():
    """API endpoint для аналізу Фур'є типів складів у множинних файлах (оновлений з часами очікування)"""
    try:
        # Отримуємо файли з запиту
        if 'files[]' not in request.files:
            return jsonify({"error": "Не знайдено файлів у запиті"}), 400
        
        uploaded_files = request.files.getlist('files[]')
        
        if not uploaded_files:
            return jsonify({"error": "Не завантажено жодного файлу"}), 400
        
        start_time = time()
        file_analyses = []
        total_syllables = 0
        
        for file in uploaded_files:
            if file.filename.endswith('.txt'):
                try:
                    # Читаємо вміст файлу
                    content = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = file.read().decode('latin-1')
                    except Exception as e:
                        print(f"Could not decode file {file.filename}: {str(e)}")
                        # Додаємо запис про помилку
                        file_analyses.append({
                            'filename': file.filename,
                            'error': True,
                            'error_message': f"Encoding error: {str(e)}",
                            'text_length': 0,
                            'syllables_count': 0,
                            'most_frequent_syllable': {'type': None, 'count': 0},
                            'second_frequent_syllable': {'type': None, 'count': 0}
                        })
                        continue
                
                # Обробляємо файл з розширеним аналізом
                analysis_result = process_file_fourier_analysis_extended(file.filename, content)
                file_analyses.append(analysis_result)
                
                if not analysis_result.get('error', False):
                    total_syllables += analysis_result.get('syllables_count', 0)
            else:
                # Файл не є .txt файлом
                file_analyses.append({
                    'filename': file.filename,
                    'error': True,
                    'error_message': "Файл не є текстовим (.txt)",
                    'text_length': 0,
                    'syllables_count': 0,
                    'most_frequent_syllable': {'type': None, 'count': 0},
                    'second_frequent_syllable': {'type': None, 'count': 0}
                })
        
        processing_time = time() - start_time
        
        return jsonify({
            'success': True,
            'file_analyses': file_analyses,
            'total_files_processed': len(file_analyses),
            'total_syllables_analyzed': total_syllables,
            'processing_time': processing_time
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Помилка аналізу Фур'є: {str(e)}", 
            "trace": traceback.format_exc(),
            "context": "Fourier analysis API"
        }), 500



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)