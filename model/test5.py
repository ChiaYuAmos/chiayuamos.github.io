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

start_time = time.time()
# 新的特徵提取函數
def extract_features(url):
    try:
        print(f"Processing URL: {url}")  
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        tld = hostname.split('.')[-1] if '.' in hostname else ''
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
            'num_dots': hostname.count('.'),
            'subdomain_level': hostname.count('.') - 1,
            'path_level': path.count('/'),
            'num_dash': url.count('-'),
            'num_dash_in_hostname': hostname.count('-'),
            'at_symbol': int('@' in url),
            'tilde_symbol': int('~' in url),
            'num_underscore': url.count('_'),
            'num_percent': url.count('%'),
            'num_query_components': query.count('&') + 1 if query else 0,
            'num_ampersand': url.count('&'),
            'num_hash': url.count('#'),
            'num_numeric_chars': sum(c.isdigit() for c in url),
            'no_https': int(parsed_url.scheme != 'https'),
            'random_string': int(bool(re.search(r'\b[a-zA-Z]{10,}\b', hostname))),
            'ip_address': int(bool(re.match(r'\b\d{1,3}(\.\d{1,3}){3}\b', hostname))),
            'domain_in_subdomains': int(any(tld in hostname for tld in ['com', 'org', 'net', 'edu', 'gov'])),
            'domain_in_paths': int(any(tld in path for tld in ['com', 'org', 'net', 'edu', 'gov'])),
            'https_in_hostname': int('https' in hostname),
            'hostname_length': len(hostname),
            'path_length': len(path),
            'query_length': len(query),
            'double_slash_in_path': int('//' in path),
            'num_sensitive_words': sum(word in url for word in ['secure', 'account', 'update', 'login']),
            'embedded_brand_name': int(any(brand in hostname for brand in ['facebook', 'google', 'paypal'])),
            'insecure_forms': 0,
            'relative_form_action': 0,
            'ext_form_action': 0,
            'abnormal_form_action': 0,
            # 'feedback':5.0
        }

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                features.update({
                    'pct_ext_hyperlinks': len(soup.find_all('a', href=True)) / (len(soup.find_all('a')) + 1),
                    'pct_ext_resource_urls': len(soup.find_all(src=True)) / (len(soup.find_all()) + 1),
                    'ext_favicon': int(bool(soup.find('link', rel='icon', href=True))),
                    'insecure_forms': int(any(form['action'].startswith('http:') for form in soup.find_all('form', action=True))),
                    'relative_form_action': int(any(form['action'].startswith('/') for form in soup.find_all('form', action=True))),
                    'ext_form_action': int(any(not form['action'].startswith(url) for form in soup.find_all('form', action=True))),
                    'abnormal_form_action': int(any(form['action'] in ['#', 'about:blank', 'javascript:true'] for form in soup.find_all('form', action=True))),
                    # 更多 HTML 相關特徵可以在這裡添加
                })
        except requests.exceptions.RequestException:
            pass
        print(f"Finished processing URL: {url}\n{features}") 
        return features
    
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return {}
# 讀取數據
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')
# 特徵提取並建立特徵數據框
# features = data['url'].apply(lambda x: extract_features(str(x)))
# features_df = pd.DataFrame(features.tolist())
# features_df.to_csv('Phishing_Dataset4_features.csv', index=False, encoding='utf-8')
features_df = pd.read_csv('Phishing_Dataset6_features_2.csv')
# print(features_df)
# 處理 TLD 特徵（使用 One-Hot Encoding）
features_df = pd.get_dummies(features_df, columns=['tld'])
# 準備數據
X = features_df
y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)  # 轉換標籤為二元格式
# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 移除 'url' 列和其他可能與 URL 直接相關的列
# columns_to_remove = ['url']
# X_train = X_train.drop(columns_to_remove, axis=1)
# X_test = X_test.drop(columns_to_remove, axis=1)
# print(X_train.dtypes)
# quit()
# 標準化數據
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.impute import SimpleImputer

# 檢查 NaN 值
# print("NaN values in X_train:", np.isnan(X_train).sum())
# print("NaN values in X_test:", np.isnan(X_test).sum())

# 使用 SimpleImputer 填充 NaN 值
imputer = SimpleImputer(strategy='most_frequent')  # 或者使用 'median' 或 'most_frequent'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# print(X_train.shape)
# quit()
# X_train = X_train[:, np.isnan(X_train).sum(axis=0) == 0]
# X_test = X_test[:, np.isnan(X_test).sum(axis=0) == 0]


# 再次檢查 NaN 值
print("NaN values in X_train after imputation:", np.isnan(X_train).sum())
print("NaN values in X_test after imputation:", np.isnan(X_test).sum())

# # 檢查無窮大值
# print("Inf values in X_train:", np.isinf(X_train).sum())
# print("Inf values in X_test:", np.isinf(X_test).sum())

# 建立 MLP 模型

from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.04)))
model.add(Dropout(0.5))  
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
# model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1)
# # 訓練模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[reduce_lr],verbose=1)
# from keras.callbacks import EarlyStopping

# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[reduce_lr, early_stopping], verbose=1)
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='mlp_model.png', show_shapes=True, show_layer_names=True)
# 預測
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# 評估模型
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
# 繪製訓練與驗證的損失
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.title('LOSS')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# 繪製訓練與驗證的準確率
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='test_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# 查看權重
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(f"Layer: {layer.name}")
#     for i, weight in enumerate(weights):
#         print(f"  Weight {i}: shape {weight.shape}")
#         print(f"  {weight}\n")
quit()

def scale_prediction(prediction):
    return prediction * 10.0

# quit()
def predict_url(model, url, scaler):
    # 提取 URL 的特徵
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    # features_df = features_df.drop(columns=['url'])
    # 移除不必要的欄位 'url'
    if 'url' in features_df.columns:
        features_df = features_df.drop(columns=['url'])
    # 對 'tld' 進行 one-hot 編碼
    features_df = pd.get_dummies(features_df, columns=['tld'])
    # 確保列名與訓練模型時的一致，缺失特徵填補為 0
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    # 進行標準化
    print(features_df)
    features_scaled = scaler.transform(features_df)
    # 預測結果
    raw_prediction = model.predict(features_scaled)[0][0]
    # 將預測結果進行縮放
    scaled_prediction = scale_prediction(raw_prediction)
    # 處理 NAN 的情況
    if np.isnan(scaled_prediction):
        scaled_prediction = 0.0

    return features, scaled_prediction

# 測試網址列表
true_website = []
phishing_website = []

with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的 URL
urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 打印預測結果
results = []
for url in urls_to_test:
    features, prediction = predict_url(model, url, scaler)
    result = {
        'url': url,
        'prediction': 'malicious' if prediction < 5.0 else 'benign',  # 分數低於 5.0 的是釣魚網站
        'score': prediction  # 添加預測分數
    }
    result.update(features)  # 更新特徵信息
    results.append(result)
    print(f"URL: {url}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {features}\n")

# 將結果存儲到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Phishing_Dataset5_result5.csv', index=False, encoding='utf-8')

# 記錄時間
end_time = time.time()
print(end_time - start_time)