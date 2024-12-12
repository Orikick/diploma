import os
import string

# Шлях до папки з текстовими файлами
INPUT_FOLDER = "texts"
OUTPUT_FOLDER = "processed_texts"

# Функція для обробки тексту
def process_text(text):
    # Приведення тексту до нижнього регістру
    text = text.lower()
    # Видалення розділових знаків та цифр
    translator = str.maketrans("", "", string.punctuation + string.digits + "—" + "…")
    text = text.translate(translator)
    return text

# Перевіряємо, чи існує папка для збереження оброблених файлів, якщо ні - створюємо її
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Проходимо по всіх файлах у папці
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Читаємо вміст файлу
        with open(input_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Обробляємо текст
        processed_content = process_text(content)

        # Записуємо оброблений текст у новий файл
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(processed_content)

print("Обробка завершена. Оброблені файли збережені у папці:", OUTPUT_FOLDER)
