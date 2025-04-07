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
            data['F'][i] = sum(model[ngram].bool)
            continue
        data["F"][i] = len(model[ngram].pos)

    df = pd.DataFrame(data=data)
    return df


def make_markov_chain(data, order=1):
    global model, L, V
    model = dict()
    L = len(data) - order
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)
    model['new_ngram'].pos = []
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
                    print('Wait for symbol calculation')
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
            model[data[L]].update({data[0]: 1})
        else:
            model[data[L]] = {data[0]: 1}

            # Connect the first word with the last one
        if data[0] in model:
            model[data[0]].update({data[L]: 1})
        else:
            model[data[0]] = {data[L]: 1}
    V = len(model)


def calculate_distance(positions, L, option, ngram, min_type=1):
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
        return positions
    
    dt = np.empty(number_of_pos - 1, dtype=np.uint32)
    for i in range(number_of_pos - 1):
        dt[i] = (positions[i + 1] - positions[i]) + min_type
    return dt


@jit(nopython=True)
def obc(positions, L, min_type):
    number_of_pos = len(positions)
    dt = np.empty(number_of_pos + 1, dtype=np.uint32)
    dt[0] = positions[0] + min_type
    
    for i in range(number_of_pos - 1):
        dt[i + 1] = (positions[i + 1] - positions[i]) + min_type
    
    dt[-1] = (L - positions[-1]) + min_type
    return dt


@jit(nopython=True)         
def pbc(positions, L, ngram, min_type):
    number_of_pos = len(positions)
    dt = np.zeros(number_of_pos, dtype=np.uint32)

    min_corr = 1

    if min_type==1:
        min_corr = 1
    
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
    t = x.mean()
    st = np.mean(x ** 2)
    return np.sqrt(st - (t ** 2))


@jit(nopython=True, fastmath=True)
def R(x):
    if len(x) == 1:
        return 0.0
    t = np.mean(x)
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
    count = np.empty(len(range(wi, l, wh)), dtype=np.uint8)
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
    return count, mse(count)


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w):
        self.count[w], self.dfa[w] = dfa(self.data, (w, self.wh, self.l))


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
            'goodness': [new_ngram_obj.goodness]
        })
        
        V = len(temp_v)
        
        # Create a serializable version of the new_ngram_obj for the response
        ngram_data = {
            "R": new_ngram_obj.R,
            "a": new_ngram_obj.a,
            "b": new_ngram_obj.b,
            "goodness": new_ngram_obj.goodness,
            "dfa": make_keys_serializable(new_ngram_obj.dfa)
        }

    else:
        make_markov_chain(data, order=n_size)
        df = make_dataframe(model, f_min)

        for index, ngram in enumerate(df['ngram']):
            model[ngram].dt = calculate_distance(np.array(model[ngram].pos, dtype=np.uint32), L, condition, ngram, min_type)

        def func(wind):
            model[ngram].counts[wind] = make_windows(model[ngram].bool, wi=wind, l=L, wsh=wh)
            model[ngram].fa[wind] = mse(model[ngram].counts[wind])

        windows = list(range(w, wm, we))
        temp_b = []
        temp_R = []
        temp_error = []
        temp_ngram = []
        temp_a = []

        for i, ngram in enumerate(df["ngram"]):
            for wind in windows:
                func(wind)

            model[ngram].temp_fa = []
            ff = [*model[ngram].fa.values()]
            c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
            model[ngram].a = c[0]
            model[ngram].b = c[1]
            
            for w_val in windows:
                model[ngram].temp_fa.append(fit(w_val, c[0], c[1]))
                
            temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
            temp_b.append(round(c[1], 8))
            temp_a.append(round(c[0], 8))

            if isinstance(ngram, tuple):
                temp_ngram.append(" ".join(ngram))

            r = round(R(np.array(model[ngram].dt)), 8)
            temp_R.append(r)
            model[ngram].R = r

        if n_size > 1:
            temp_ngram.append("new_ngram")
            df["ngram"] = temp_ngram

        df['R'] = temp_R
        df['b'] = temp_b
        df['a'] = temp_a
        df['goodness'] = temp_error
        df = df.sort_values(by="F", ascending=False)
        df['rank'] = range(1, len(temp_R) + 1)
        df = df.set_index(pd.Index(np.arange(len(df))))
        
        # Create a dictionary of serializable model data
        ngram_data = {}
        for ngram in df['ngram']:
            if n_size > 1 and ngram != "new_ngram":
                key = tuple(ngram.split())
            else:
                key = ngram
                
            ngram_data[str(ngram)] = {
                "R": getattr(model[key], 'R', 0),
                "a": getattr(model[key], 'a', 0),
                "b": getattr(model[key], 'b', 0),
                "fa": make_keys_serializable(getattr(model[key], 'fa', {})),
                "temp_fa": getattr(model[key], 'temp_fa', []),
                "bool": model[key].bool.tolist() if hasattr(model[key], 'bool') else []
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
    
    # Prepare storage for results
    corpus_results = []
    length_info = []
    file_names = []
    
    # First pass: collect all file lengths to calculate appropriate F_min
    for file_name, file_content in files.items():
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
    
    # Find Lmin and Lmax
    sorted_by_len = sorted(length_info, key=lambda x: x["length"])
    if not sorted_by_len:
        return {"error": "No valid files found in corpus"}
        
    Lmin_actual = sorted_by_len[0]["length"]
    Lmax_actual = sorted_by_len[-1]["length"]
    
    # Calculate F_min slope
    if Lmax_actual == Lmin_actual:
        slope = 0
    else:
        slope = (fmin_for_lmax - fmin_for_lmin) / (Lmax_actual - Lmin_actual)
    
    # Process each file
    start_all = time()
    
    for i, item in enumerate(sorted_by_len, start=1):
        file_name = item["file"]
        file_length = item["length"]
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
        f_min_for_this = fmin_for_lmin + slope * (file_length - Lmin_actual)
        f_min_for_this = round(f_min_for_this)
        
        # Start processing this file
        t0 = time()
        data = prepere_data(processed_content, n_size, split)
        
        if definition == "dynamic":
            # Dynamic analysis
            # Create windows array
            windows = list(range(w, wm, we))
            new_ngram_obj = newNgram(data, wh, file_length)
            
            for w_val in windows:
                new_ngram_obj.func(w_val)
                
            temp_v = []
            temp_pos = []
            for i, ngram in enumerate(data):
                if ngram not in temp_v:
                    temp_v.append(ngram)
                    temp_pos.append(i)
                    
            new_ngram_obj.dt = calculate_distance(np.array(temp_pos, dtype=np.uint8), file_length, condition, ngram, min_type)
            new_ngram_obj.R = round(R(new_ngram_obj.dt), 8)
            
            try:
                c, _ = curve_fit(fit, [*new_ngram_obj.dfa.keys()], [*new_ngram_obj.dfa.values()], method='lm', maxfev=5000)
                new_ngram_obj.a = round(c[0], 8)
                new_ngram_obj.b = round(c[1], 8)
                
                new_ngram_obj.temp_dfa = []
                for w_val in new_ngram_obj.dfa.keys():
                    new_ngram_obj.temp_dfa.append(fit(w_val, new_ngram_obj.a, new_ngram_obj.b))
                    
                new_ngram_obj.goodness = round(r2_score([*new_ngram_obj.dfa.values()], new_ngram_obj.temp_dfa), 8)
                
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
                R_avg = dR = Rw_avg = dRw = b_avg = db = bw_avg = dbw = 0
                Vcount = len(temp_v) if temp_v else 0
            
        else:
            # Static analysis
            make_markov_chain(data, order=n_size)
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
                    
                # Calculate dt (distances)
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
                        count_arr = make_windows(model[ngram].bool, wi=wind, l=file_length, wsh=wh)
                        if len(count_arr) == 0:
                            model[ngram].fa[wind] = 0.0
                        else:
                            model[ngram].fa[wind] = mse(count_arr)
                        model[ngram].counts[wind] = count_arr
                    except Exception as e:
                        print(f"Error in make_windows for {ngram}: {str(e)}")
                        model[ngram].fa[wind] = 0.0
                        model[ngram].counts[wind] = np.array([])
                
                # Curve fitting
                try:
                    ff = [*model[ngram].fa.values()]
                    if len(ff) > 0 and not all(v == 0 for v in ff) and len(windows) > 0:
                        try:
                            c, _ = curve_fit(fit, windows, ff, method='lm', maxfev=5000)
                            model[ngram].a = c[0]
                            model[ngram].b = c[1]
                            
                            model[ngram].temp_fa = [fit(w_val, c[0], c[1]) for w_val in windows]
                            temp_error.append(round(r2_score(ff, model[ngram].temp_fa), 5))
                            temp_b.append(round(c[1], 8))
                            temp_a.append(round(c[0], 8))
                        except Exception as e:
                            print(f"Curve fit failed for {ngram}: {str(e)}")
                            print(f"Windows: {windows}")
                            print(f"Values: {ff}")
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
                    model[ngram].a = 0
                    model[ngram].b = 0
                    temp_error.append(0)
                    temp_b.append(0)
                    temp_a.append(0)
                
                # Calculate R
                try:
                    rr = round(R(np.array(model[ngram].dt)), 8)
                    temp_R.append(rr)
                    model[ngram].R = rr
                except Exception as e:
                    print(f"Error calculating R for {ngram}: {str(e)}")
                    temp_R.append(0)
                    model[ngram].R = 0
                
                # Format ngram
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
                        df_no_new["w"] = df_no_new["F"] / df_no_new["F"].sum()
                        R_avg = df_no_new["R"].mean()
                        dR = df_no_new["R"].std()
                        df_no_new["Rw"] = df_no_new["R"] * df_no_new["w"]
                        Rw_avg = df_no_new["Rw"].sum()
                        dRw = np.sqrt(((df_no_new["R"] - Rw_avg)**2 * df_no_new["w"]).sum())
                        
                        b_avg = df_no_new["b"].mean()
                        db = df_no_new["b"].std()
                        df_no_new["bw"] = df_no_new["b"] * df_no_new["w"]
                        bw_avg = df_no_new["bw"].sum()
                        dbw = np.sqrt(((df_no_new["b"] - bw_avg)**2 * df_no_new["w"]).sum())
                        Vcount = df_no_new["ngram"].nunique()
                    except Exception as e:
                        print(f"Error calculating metrics for {file_name}: {str(e)}")
                        R_avg = dR = Rw_avg = dRw = b_avg = db = bw_avg = dbw = 0
                        Vcount = 0
                else:
                    R_avg = dR = Rw_avg = dRw = b_avg = db = bw_avg = dbw = 0
                    Vcount = 0
            else:
                R_avg = dR = Rw_avg = dRw = b_avg = db = bw_avg = dbw = 0
                Vcount = 0
        
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
            "dbw": dbw
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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
