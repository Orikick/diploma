import os
from pathlib import Path

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
    suffixes = ['ський', 'цький', 'зький']
    for suffix in suffixes:
        if index + len(suffix) <= len(word) and word[index:index+len(suffix)].lower() == suffix:
            return len(suffix)
    return 0

def has_soft_sign_ahead(word, index):
    """Check if next character is a soft sign"""
    return index + 1 < len(word) and word[index + 1] == 'ь'

def split_into_syllables(word):
    """Split Ukrainian word into syllables"""
    if not word:
        return []
        
    syllables = []
    current_syllable = ""
    i = 0
    
    # Preserve apostrophe but remove other punctuation
    word = ''.join(c for c in word if c.isalpha() or c == "'")
    
    if not word:
        return []
    
    while i < len(word):
        # Check for special suffixes
        suffix_len = has_special_suffix(word, i)
        if suffix_len > 0:
            if current_syllable:
                syllables.append(current_syllable)
            syllables.append(word[i:i+suffix_len])
            break
            
        # Add characters until we find a vowel
        while i < len(word) and not is_vowel(word[i]):
            # Check for дж/дз
            if i < len(word) - 1 and is_special_consonant_pair(word, i):
                if current_syllable and any(is_vowel(c) for c in current_syllable):
                    syllables.append(current_syllable)
                    current_syllable = ""
                current_syllable = word[i:i+2]
                i += 2
            else:
                # Check for soft sign after current character
                if has_soft_sign_ahead(word, i):
                    current_syllable += word[i] + word[i+1]
                    i += 2
                else:
                    current_syllable += word[i]
                    i += 1
        
        # Add the vowel if we found one
        if i < len(word):
            current_syllable += word[i]
            i += 1
            
            # If this is not the end of the word
            if i < len(word):
                # Check for special suffix first
                if has_special_suffix(word, i):
                    syllables.append(current_syllable)
                    current_syllable = word[i:]
                    break
                    
                # Look ahead for the next vowel or special consonant pair
                next_pos = i
                while next_pos < len(word):
                    if is_vowel(word[next_pos]) or (next_pos < len(word) - 1 and 
                       is_special_consonant_pair(word, next_pos)):
                        break
                    next_pos += 1
                
                # If we found another vowel or special pair
                if next_pos < len(word):
                    consonant_count = next_pos - i
                    
                    # If there's only one consonant or it's дж/дз, start new syllable
                    if consonant_count == 1 or is_special_consonant_pair(word, i):
                        syllables.append(current_syllable)
                        current_syllable = ""
                    # If there are multiple consonants
                    elif consonant_count > 1:
                        # Check for special suffix before splitting
                        suffix_start = i
                        while suffix_start < next_pos:
                            if has_special_suffix(word, suffix_start):
                                current_syllable += word[i:suffix_start]
                                syllables.append(current_syllable)
                                current_syllable = word[suffix_start:]
                                i = len(word)  # Exit main loop
                                break
                            suffix_start += 1
                            
                        if i < len(word):  # If no suffix found
                            # Calculate split point considering soft sign
                            mid_point = i + (consonant_count // 2)
                            if has_soft_sign_ahead(word, mid_point - 1):
                                mid_point += 1
                                
                            current_syllable += word[i:mid_point]
                            syllables.append(current_syllable)
                            current_syllable = word[mid_point:next_pos]
                            i = next_pos
                            continue
            
            if current_syllable:
                syllables.append(current_syllable)
                current_syllable = ""
    
    # Handle any remaining syllable
    if current_syllable:
        if syllables and not any(is_vowel(c) for c in current_syllable):
            syllables[-1] += current_syllable
        else:
            syllables.append(current_syllable)
    
    return syllables

def process_text_file(input_path, output_path):
    """Process a single text file and save syllable version"""
    try:
        # Читаємо вхідний файл
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Розбиваємо текст на слова
        words = text.split()
        
        # Обробляємо кожне слово
        processed_words = []
        for word in words:
            # Відділяємо пунктуацію від слова
            prefix = ''
            suffix = ''
            while word and not (word[0].isalpha() or word[0] == "'"):
                prefix += word[0]
                word = word[1:]
            while word and not (word[-1].isalpha() or word[-1] == "'"):
                suffix = word[-1] + suffix
                word = word[:-1]
            
            # Розбиваємо слово на склади
            if word:
                syllables = split_into_syllables(word)
                if syllables:
                    processed_word = prefix + ' '.join(syllables) + suffix
                else:
                    processed_word = prefix + word + suffix
            else:
                processed_word = prefix + suffix
            processed_word = processed_word.replace("ь", "").replace("Ь", "")
            processed_words.append(processed_word)
            
        
        # Зберігаємо результат
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(' '.join(processed_words))
            
        return True
    except Exception as e:
        print(f"Помилка при обробці файлу {input_path}: {str(e)}")
        return False

def process_all_files():
    """Process all files from processed_texts to syllables_texts"""
    # Створюємо папку для результатів, якщо її немає
    input_dir = Path('processed_texts')
    output_dir = Path('syllables_texts')
    output_dir.mkdir(exist_ok=True)
    
    # Перевіряємо наявність вхідної папки
    if not input_dir.exists():
        print("Помилка: Папка 'processed_texts' не існує")
        return
    
    # Обробляємо всі .txt файли
    success_count = 0
    total_count = 0
    
    for file_path in input_dir.glob('*.txt'):
        total_count += 1
        output_path = output_dir / file_path.name
        
        if process_text_file(file_path, output_path):
            success_count += 1
            print(f"Успішно оброблено: {file_path.name}")
        else:
            print(f"Помилка при обробці: {file_path.name}")
    
    print(f"\nОбробка завершена. Успішно оброблено {success_count} з {total_count} файлів.")

if __name__ == "__main__":
    process_all_files()