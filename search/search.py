import subprocess
import urllib.parse
import os
from bs4 import BeautifulSoup

def main():
    while True:
        query = input("想搜尋的關鍵字（輸入exit離開）: ")
        if query == "exit":
            break
        encoded_query = urllib.parse.quote(query)
        cmd = f"curl 'https://www.google.com/search?q={encoded_query}' -H 'User-Agent: Mozilla/5.0'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        text = result.stdout.decode('utf-8', errors='replace')
        soup = BeautifulSoup(text, 'html.parser')
        hrefs = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/url?q='):
                href = urllib.parse.unquote(href.split('&')[0].replace('/url?q=', ''))
                hrefs.append(href)
        txt = f"{query}.txt"
        with open(txt, 'w', encoding='utf-8') as file:
            for href in hrefs[:5]:
                file.write(href + "\n")
        
        with open("facebooklogin.txt", 'r', encoding='utf-8') as file:
            comparison_content = file.read()
        
        with open(txt, 'r', encoding='utf-8') as file:
            urls = file.read().splitlines()
        
        scores = []
        for url in urls:
            cmd = f"curl '{url}' -H 'User-Agent: Mozilla/5.0'"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            page_content = result.stdout.decode('utf-8', errors='replace')
            count = 1000.0
            # print(page_content)
            for j in comparison_content:
                if(j not in page_content):
                    count -= 0.05
            scores.append((url, count))
        for url, score in scores:
            print(f"URL: {url}\t Score: {score}")
if __name__ == "__main__":
    main()
