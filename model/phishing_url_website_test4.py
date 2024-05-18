import pandas as pd
import requests
from bs4 import BeautifulSoup
import difflib
from urllib.parse import urlparse
import tldextract

# 定義特徵提取函數
def extract_features(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 檢查請求是否成功
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None

    extracted = tldextract.extract(url)
    domain = extracted.domain
    tld = extracted.suffix

    URLSimilarityIndex = difflib.SequenceMatcher(None, domain, url).ratio()
    NoOfOtherSpecialCharsInURL = sum(1 for char in url if not char.isalnum() and char not in ['.', '-', '_'])
    SpacialCharRatioInURL = NoOfOtherSpecialCharsInURL / len(url)
    IsHTTPS = 1 if urlparse(url).scheme == "https" else 0
    LineOfCode = len(html_content.split('\n'))
    title = soup.title.string if soup.title else ''
    DomainTitleMatchScore = difflib.SequenceMatcher(None, domain, title).ratio()
    URLTitleMatchScore = difflib.SequenceMatcher(None, url, title).ratio()
    IsResponsive = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0
    HasDescription = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
    social_nets = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']
    HasSocialNet = 1 if any(net in html_content for net in social_nets) else 0
    HasSubmitButton = 1 if soup.find('button', attrs={'type': 'submit'}) else 0
    HasCopyrightInfo = 1 if 'copyright' in html_content.lower() else 0
    NoOfImage = len(soup.find_all('img'))
    NoOfJS = len(soup.find_all('script'))
    NoOfSelfRef = len([link for link in soup.find_all('a', href=True) if domain in link['href']])

    return {
        "URL": url,
        "Domain": domain,
        "TLD": tld,
        "URLSimilarityIndex": URLSimilarityIndex,
        "NoOfOtherSpecialCharsInURL": NoOfOtherSpecialCharsInURL,
        "SpacialCharRatioInURL": SpacialCharRatioInURL,
        "IsHTTPS": IsHTTPS,
        "LineOfCode": LineOfCode,
        "Title": title,
        "DomainTitleMatchScore": DomainTitleMatchScore,
        "URLTitleMatchScore": URLTitleMatchScore,
        "IsResponsive": IsResponsive,
        "HasDescription": HasDescription,
        "HasSocialNet": HasSocialNet,
        "HasSubmitButton": HasSubmitButton,
        "HasCopyrightInfo": HasCopyrightInfo,
        "NoOfImage": NoOfImage,
        "NoOfJS": NoOfJS,
        "NoOfSelfRef": NoOfSelfRef
    }

# 定義網址列表及其標籤
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的URL
test_urls = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 提取特徵並保存為 DataFrame
data = []
for url in test_urls:
    features = extract_features(url)
    if features:  # 如果成功提取特徵，則將其添加到數據列表中
        data.append(features)

df = pd.DataFrame(data)

# 將 DataFrame 保存為 CSV 文件
df.to_csv("url_features.csv", index=False)
