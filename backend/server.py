import requests
from bs4 import BeautifulSoup

# URL сторінки для скрейпінгу
url = "https://www.ukrlib.com.ua/books/printit.php?tid=297"  

def scrape_page(url):
    try:
        # запит до сторінки
        response = requests.get(url)
        response.raise_for_status()  # Перевірка на помилки

        # Парсер HTML сторінки
        soup = BeautifulSoup(response.content, "html.parser")

        content = soup.find("article", {"id": "content"})

        if content:
            # Ігнорування елементів
            for tag in content.find_all(["em", "blockquote"]):
                tag.decompose()

            text = content.get_text(separator="\n").strip()

            return text
        else:
            return "Текстовий блок не знайдено."

    except requests.exceptions.RequestException as e:
        return f"Помилка під час запиту: {e}"


text = scrape_page(url)


print(text)
