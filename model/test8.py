import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from bs4 import BeautifulSoup
import requests
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# 讀取數據集
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')

# 檢查數據集中缺失值
print("數據集中的缺失值:")
print(data.isnull().sum())

# 特徵提取函數
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url': url,
        'url_length': len(url),
        'domain_length': len(parsed_url.netloc),
        'tld': parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else '',
        'has_ip': int(parsed_url.netloc.replace('.', '').isdigit()),
        'num_special_chars': sum(not c.isalnum() and c not in ['.', ':', '/'] for c in url),
        'has_https': int(parsed_url.scheme == 'https'),
        'has_www': int('www.' in parsed_url.netloc),
        'pct_ext_hyperlinks': 0.0,
        'pct_ext_resource_urls': 0.0,
        'ext_favicon': 0.0,
        'feedback': 5.0
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

# 讀取經過特徵提取處理的數據
features_df = pd.read_csv('Phishing_Dataset8_features.csv')

# 進行 One-Hot 編碼
features_df = pd.get_dummies(features_df, columns=['tld'])

# 分離特徵和標籤
X = features_df.drop(columns=['url'])  # 不包括 'url' 欄位
y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)

# 查找數值欄位
numeric_cols = X.select_dtypes(include=[np.number]).columns

# 對數值欄位進行均值填補
imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 對數值數據進行標準化
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 應用 SMOTE 解決類別不平衡問題
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

# 預測測試集
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# 打印混淆矩陣和分類報告
print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

model.save('phishing_detection_model.h5')

# 預測 URL 的函數
def scale_prediction(prediction):
    return prediction * 10.0

# 保存訓練過程中的特徵名稱
feature_names = X.columns.tolist()

def predict_url(model, url, scaler, feature_names):
    features = extract_features(url)
    features_df = pd.DataFrame([features])

    # Perform One-Hot Encoding on the 'tld' column
    features_df = pd.get_dummies(features_df, columns=['tld'])
    # Add missing columns from feature_names
    missing_cols = [col for col in feature_names if col not in features_df.columns]
    if missing_cols:
        for col in missing_cols:
            features_df[col] = 0

    # Remove extra columns not present in feature_names
    extra_cols = [col for col in features_df.columns if col not in feature_names]
    features_df.drop(extra_cols, axis=1, inplace=True)

    # Reorder columns to match training feature set
    features_df = features_df.reindex(columns=feature_names, fill_value=0)

    # Scale the features
    features_scaled = scaler.transform(features_df)

    # Make the prediction
    raw_prediction = model.predict(features_scaled)[0][0]
    scaled_prediction = scale_prediction(raw_prediction)
    if np.isnan(scaled_prediction):
        scaled_prediction = 0.0
    
    return features, scaled_prediction

# 測試 URL 的部分
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]
results = []
for url in urls_to_test:
    try:
        features, prediction = predict_url(model, url, scaler, feature_names)
        result = {
            'url': url,
            'prediction': 'malicious' if prediction < 5.0 else 'benign',
            'score': prediction
        }
        result.update(features)
        results.append(result)
        print(f"URL: {url}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {features}\n")
    except Exception as e:
        print(f"Error processing URL {url}: {e}")

# 儲存結果
results_df = pd.DataFrame(results)
results_df.to_csv('Phishing_Dataset8_result8.csv', index=False)