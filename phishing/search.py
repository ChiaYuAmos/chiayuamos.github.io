import requests
from bs4 import BeautifulSoup
url = input("請輸入網址: ")
html = requests.get(url)
sp = BeautifulSoup(html.text, 'html5lib')
with open('web.txt', 'w', encoding='utf-8') as file:
    file.write(sp.prettify())