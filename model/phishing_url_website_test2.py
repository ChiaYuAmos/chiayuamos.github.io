import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import difflib
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    try:
        response = requests.get(url, timeout=10)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
    except requests.RequestException as e:
        # print(f"Error accessing {url}: {e}")
        return [0]*15
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
    return [
        URLSimilarityIndex,
        NoOfOtherSpecialCharsInURL,
        SpacialCharRatioInURL,
        IsHTTPS,
        LineOfCode,
        DomainTitleMatchScore,
        URLTitleMatchScore,
        IsResponsive,
        HasDescription,
        HasSocialNet,
        HasSubmitButton,
        HasCopyrightInfo,
        NoOfImage,
        NoOfJS,
        NoOfSelfRef
    ]

data = pd.read_csv("phishing_url_website.csv")
object_cols = [i for i in data.columns.tolist() if data[i].dtype=="object"]
data.drop(object_cols, axis=1, inplace=True)
top_corr = data.corrwith(data["label"]).abs().sort_values(ascending=False)[:6].index.to_list()
scaler = MinMaxScaler()
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
rfc_model = RandomForestClassifier()
rfc_model.fit(scaled_X_train, y_train)
rfc_preds = rfc_model.predict(scaled_X_test)
print(classification_report(y_test, rfc_preds))

true_website=[]
phishing_website=[]
website = []
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_url = file.read().split("\n")
for url in test_url:
    if(url==('True Website'or'Phishing Website'or "")):
        continue
    else:
        website.append(url)
for url in website:
    features = extract_features(url)  
    print(f"{url}'s features: {features}\n\n")
    scaled_features = scaler.transform([features])
    prediction = rfc_model.predict(scaled_features)
    if prediction == 1:
        phishing_website.append(url)
    else:
        true_website.append(url)

print(f"True Website: {true_website}")
print()
print(f"Phishing Website: {phishing_website}")