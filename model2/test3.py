import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from bs4 import BeautifulSoup
import requests
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier
import idna  # 引入 idna 模組以進行編碼驗證

# 讀取數據
data = pd.read_csv('Phishing_Dataset2.csv', encoding='utf-8')

# 檢查缺失值
print("數據集中的缺失值:")
print(data.isnull().sum())

# URL 驗證函數
def is_valid_url(url):
    try:
        result = urlparse(url)
        # 檢查 URL 是否可以使用 IDNA（國際化域名應用）編碼
        idna.encode(result.netloc)
        return all([result.scheme, result.netloc])
    except (ValueError, UnicodeError, idna.IDNAError):
        return False

# 特徵提取函數
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'domain_length': len(parsed_url.netloc),
        'tld': parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else '',
        'has_ip': int(parsed_url.netloc.replace('.', '').isdigit()),  # 判斷是否有 IP 地址
        'num_special_chars': sum(not c.isalnum() and c not in ['.', ':', '/'] for c in url),
        'has_https': int(parsed_url.scheme == 'https'),
        'has_www': int('www.' in parsed_url.netloc)
    }
    return features

# 特徵提取並建立特徵數據框
features = data['url'].apply(lambda x: extract_features(str(x)))
features = features.dropna()  # 刪除無效的URL

features_df = pd.DataFrame(features.tolist())
features_df.to_csv('Phishing_Dataset3_features.csv', index=False, encoding='utf-8')

# 處理 TLD 特徵（使用 One-Hot Encoding）
features_df = pd.get_dummies(features_df, columns=['tld'])

# 準備數據
X = features_df
y = data['label'][features.index].apply(lambda x: 1 if x == 'malicious' else 0)  # 轉換標籤為二元格式

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化數據
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 應用 SMOTE 處理類別不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 建立 MLP 模型
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(X_resampled, y_resampled, epochs=75, batch_size=32, validation_split=0.2, verbose=1)

# 預測
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# 評估模型
print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# 預測新的網址
def predict_url(model, url, scaler):
    features = extract_features(url)
    if features is None:
        return 'invalid', None
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=['tld'])
    # 確保列一致，填補缺失的列
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    features_scaled = scaler.transform(features_df)
    
    # 獲取原始預測分數（介於 0 到 1 之間）
    prediction = model.predict(features_scaled)[0][0]
    
    # 將預測分數縮放到 0.0 到 10.0 之間
    scaled_score = prediction * 10.0
    
    # 根據縮放後的分數確定標籤
    label = 'malicious' if scaled_score < 5.0 else 'benign'
    
    return label, scaled_score

true_website = []
phishing_website = []
# 測試網址
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的URL
urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 打印預測結果
for url in urls_to_test:
    result, score = predict_url(model, url, scaler)
    if result == 'invalid':
        continue  # 跳過無效的URL
    if result == "malicious":
        phishing_website.append(url)
    else:
        true_website.append(url)
    print(f"URL: {url}\nPrediction: {result}\nScore: {score}\n")

print("True Website:\n", "\t\n".join(true_website))
print("\nPhishing Website:\n", "\t\n".join(phishing_website))

# 將預測結果和特徵存儲到 CSV
results = []
for url in urls_to_test:
    features, prediction = predict_url(model, url, scaler)
    result = {
        'url': url,
        'prediction': 'malicious' if prediction < 5.0 else 'benign', 
        'score': prediction  # 添加预测分数
    }
    result.update(features)
    results.append(result)
    print(f"URL: {url}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {features}\n")


results_df = pd.DataFrame(results)
results_df.to_csv('Phishing_Dataset3_result3.csv', index=False, encoding='utf-8')