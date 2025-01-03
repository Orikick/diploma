import os
import re

# Шлях до вхідної та вихідної папок
input_folder = "syllables_texts"
output_folder = "cv_texts"

# Перевірка, чи існує вихідна папка, і створення її, якщо потрібно
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Голосні української мови
vowels = "аеєиіiїоуюяАЕЄИІIЇОУЮЯ"

# Функція для перетворення тексту на CV-послідовності
def text_to_cv(text):
    # Видаляємо м'які знаки
    text = text.replace("ь", "").replace("Ь", "")
    
    # Перетворення кожного символу на C або V
    cv_text = "".join(["v" if char in vowels else "c" if re.match(r"[\u0410-\u044f]", char) else char for char in text])
    return cv_text

# Обробка кожного файлу в папці
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Читаємо вміст файлу
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Перетворення тексту
        cv_text = text_to_cv(text)

        # Запис результату у вихідний файл
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cv_text)

print("Перетворення завершено! Файли збережено у папці cv_texts.")
