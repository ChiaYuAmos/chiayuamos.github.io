import requests
from bs4 import BeautifulSoup
while(1):
    url = input("請輸入網址(input q to quick): ")
    if url == "q":
        break
    html = requests.get(url)
    sp = BeautifulSoup(html.text, 'html5lib')
    Notuse = ["https:","www","/",".","com","tw"]
    for i in Notuse:
        url = url.replace(i,'')
    txt = url + ".txt"
    with open(txt, 'w', encoding='utf-8') as file:
        file.write(sp.prettify())