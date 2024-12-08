import requests
from bs4 import BeautifulSoup
import csv
import os
from transliterate import translit

# Функція для скрейпінгу однієї сторінки
def scrape_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Отримання контенту
        content = soup.find("article", {"id": "content"})
        
        # Парсинг автора (оновлено для відповідності структурі сайту)
        author_block = soup.find("div", {"class": "page-title"})
        author = author_block.find("h2") if author_block else None

        if content:
            for tag in content.find_all(["em", "blockquote"]):
                tag.decompose()
            text = content.get_text(separator="\n").strip()
        else:
            text = "Текстовий блок не знайдено."

        # Перевірка автора
        author_name = author.get_text(strip=True) if author else "Автор невідомий"

        return author_name, text

    except requests.exceptions.RequestException as e:
        return "Помилка", f"Помилка під час запиту: {e}"


# Шлях до вхідного CSV
input_csv = "input_links.csv"  # Змініть на ваш файл
output_folder = "texts"  # Папка для текстів

# Створення папки, якщо вона не існує
os.makedirs(output_folder, exist_ok=True)

# Функція для обробки кількох посилань
def scrape_multiple_links(input_csv, output_folder):
    try:
        with open(input_csv, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            links = [row[0] for row in reader]  # Передбачається, що URL у першій колонці

        for link in links:
            print(f"Скрейпінг: {link}")
            author_name, text = scrape_page(link)

            # Транслітерація імені автора
            author_translit = translit(author_name, 'uk', reversed=True).replace("'", "").replace(" ", "_")
            author_filename = author_translit if author_name != "Автор невідомий" else "unknown_author"

            # Формування імені файлу
            tid = link.split("tid=")[-1]  # Отримання tid з URL
            filename = f"{author_filename}_{tid}.txt"
            filepath = os.path.join(output_folder, filename)

            # Збереження тексту у файл
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"Текст збережено у файл: {filepath}")

    except FileNotFoundError:
        print("Файл не знайдено. Перевірте шлях до CSV.")

# Виклик функції
scrape_multiple_links(input_csv, output_folder)
