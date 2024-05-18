import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 讀取數據
data = pd.read_csv('Phishing_Dataset.csv')

# 檢查缺失值
print("Missing values in dataset:")
print(data.isnull().sum())

# 特徵提取函數
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'domain_length': len(parsed_url.netloc),
        'tld': parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else '',
        'has_ip': int(parsed_url.netloc.replace('.', '').isdigit()), 
        'num_special_chars': sum(not c.isalnum() and c not in ['.', ':', '/'] for c in url),
        'has_https': int(parsed_url.scheme == 'https'),
        'has_www': int('www.' in parsed_url.netloc)
    }
    return features

# 提取特徵並建立特徵數據框
features = data['url'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())

# 處理 TLD 特徵
features_df = pd.get_dummies(features_df, columns=['tld'])

# 準備數據
X = features_df
y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)  # 轉換標籤為二元格式

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化數據
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立 MLP 模型
model = Sequential()
model.add(Dense(300, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 預測
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# 評估模型
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 預測新的網址
def predict_url(model, url, scaler):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=['tld'])
    # 確保列一致，填補缺失的列
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    return 'malicious' if prediction > 0.5 else 'benign'

# 測試網址
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的URL
urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 打印預測結果
for url in urls_to_test:
    result = predict_url(model, url, scaler)
    print(f"URL: {url}\nPrediction: {result}\n")