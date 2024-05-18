import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import difflib
import tldextract

def extract_features(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check if request was successful
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

    features = [
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

    if len(features) != 15:
        return None

    return features

# 讀取數據並處理
data = pd.read_csv("phishing_url_website.csv")
object_cols = [i for i in data.columns.tolist() if data[i].dtype == "object"]
data.drop(object_cols, axis=1, inplace=True)
top_corr = data.corrwith(data["label"]).abs().sort_values(ascending=False)[:6].index.to_list()

scaler = MinMaxScaler()
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Model tuning with Stratified Cross-Validation
rfc_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rfc_model, scaled_X_train, y_train, cv=stratified_kfold)
print("Stratified Cross-Validation Accuracy Scores:", scores)
print("Mean Stratified Cross-Validation Accuracy:", scores.mean())

rfc_model.fit(scaled_X_train, y_train)
rfc_preds = rfc_model.predict(scaled_X_test)
print(classification_report(y_test, rfc_preds))

true_website = []
phishing_website = []
fake_website=[]
website = []

# 讀取測試網址
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_url = file.read().split("\n")

# 過濾無效的URL
test_url = [url for url in test_url if url and url not in ['True Website', 'Phishing Website', '']]

for url in test_url:
    features = extract_features(url)
    if features is None:  # 檢查特徵是否為None
        fake_website.append(url)
        continue

    scaled_features = scaler.transform([features])
    prediction = rfc_model.predict(scaled_features)

    if prediction == 1:
        phishing_website.append(url)
    else:
        true_website.append(url)

print("True Website:", true_website)
print("Phishing Website:", phishing_website)
print("Fake Website:", fake_website)
