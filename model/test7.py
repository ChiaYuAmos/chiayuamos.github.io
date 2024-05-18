import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from bs4 import BeautifulSoup
import requests
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier
from sklearn.impute import SimpleImputer

# 讀取數據
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')

# 檢查缺失值
print("數據集中的缺失值:")
print(data.isnull().sum())

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
        'has_www': int('www.' in parsed_url.netloc),
        'pct_ext_hyperlinks': 0.0,
        'pct_ext_resource_urls': 0.0,
        'ext_favicon': 0.0,
    }
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            features.update({
                'pct_ext_hyperlinks': len(soup.find_all('a', href=True)) / (len(soup.find_all('a')) + 1),
                'pct_ext_resource_urls': len(soup.find_all(src=True)) / (len(soup.find_all()) + 1),
                'ext_favicon': int(bool(soup.find('link', rel='icon', href=True))),
            })
    except requests.exceptions.RequestException:
        pass  
    return features

# 特徵提取並建立特徵數據框
# features = data['url'].apply(lambda x: extract_features(str(x)))
# features_df = pd.DataFrame(features.tolist())
# features_df.to_csv('Phishing_Dataset4_features.csv', index=False, encoding='utf-8')
features_df = pd.read_csv('Phishing_Dataset7_features.csv')

# 處理 TLD 特徵（使用 One-Hot Encoding）
features_df = pd.get_dummies(features_df, columns=['tld'])

# 準備數據
X = features_df
y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)  # 轉換標籤為二元格式

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化數據
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

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
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 預測
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# 評估模型
print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

def scale_prediction(prediction):
    return prediction * 10.0

# 預測新的網址
def predict_url(model, url, scaler):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=['tld'])
    # 確保列一致，填補缺失的列
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    features_scaled = scaler.transform(features_df)
    raw_prediction = model.predict(features_scaled)[0][0]  # 获取原始预测分数
    scaled_prediction = scale_prediction(raw_prediction)  # 缩放到0.0-10.0
    if np.isnan(scaled_prediction):
        scaled_prediction = 0.0  # 处理NaN情况
    return features, scaled_prediction

true_website = []
phishing_website = []
# 測試網址
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的URL
urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 打印預測結果
results = []
for url in urls_to_test:
    features, prediction = predict_url(model, url, scaler)
    result = {
        'url': url,
        'prediction': 'malicious' if prediction < 5.0 else 'benign',  # 5.0以下为钓鱼网站
        'score': prediction  # 添加预测分数
    }
    result.update(features)
    results.append(result)
    print(f"URL: {url}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {features}\n")

# 將結果存儲到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Phishing_Dataset7_result7.csv', index=False, encoding='utf-8')

print("True Website:\n", "\t\n".join([r['url'] for r in results if r['prediction'] == 'benign']))
print("\nPhishing Website:\n", "\t\n".join([r['url'] for r in results if r['prediction'] == 'malicious']))

# 預測函數，檢查URL是否已在結果文件中存在
def predict_url_2(model, url, scaler):
    # Check if the result file exists, if not, create an empty DataFrame
    results_df = pd.read_csv('Phishing_Dataset7_result7.csv')

    if url in results_df['url'].values:
        # print(f"URL: {url} 已存在於 Phishing_Dataset7_result7.csv 中，不重新預測。")
        return results_df.loc[results_df['url'] == url].iloc[0].to_dict()
    else:
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        features_df = pd.get_dummies(features_df, columns=['tld'])
        features_df = features_df.reindex(columns=X.columns, fill_value=0)
        features_scaled = scaler.transform(features_df)
        raw_prediction = model.predict(features_scaled)[0][0]
        scaled_prediction = scale_prediction(raw_prediction)
        if np.isnan(scaled_prediction):
            scaled_prediction = 0.0

        result = {
            'url': url,
            'prediction': 'malicious' if scaled_prediction < 5.0 else 'benign',
            'score': scaled_prediction
        }
        result.update(features)

        # Append the new result to results_df using pd.concat
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        results_df.to_csv('Phishing_Dataset7_result7.csv', index=False, encoding='utf-8')
        # print(f"URL: {url} 預測為: {result['prediction']}，分數: {result['score']}")
        return result

# # Interactive mode for URL input
# while True:
#     url = input("Input URL or Exit: ")
#     if url.lower() == "exit":
#         break
#     result = predict_url_2(model, url