import pandas as pd
import numpy as np
import re
import csv
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model  # 用於生成模型圖
import os 
import http.server
import socketserver
import joblib
from urllib.parse import parse_qs
from datetime import datetime


# start_time = time.time()
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
# 特徵提取並建立特徵數據框
# features = data['url'].apply(lambda x: extract_features(str(x)))
# features_df = pd.DataFrame(features.tolist())
# features_df.to_csv('Phishing_Dataset_features.csv', index=False, encoding='utf-8')


def model(data,features_df,rs,lr,ep,current_date):
    features_df = pd.get_dummies(features_df, columns=['tld'])
    X = features_df
    y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)  #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    imputer = SimpleImputer(strategy='most_frequent') 
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.04)))
    model.add(Dropout(0.5))  
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.04)))
    model.add(Dropout(0.5))  
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1)
    history = model.fit(X_train, y_train, epochs=ep, batch_size=32, validation_split=0.2, callbacks=[reduce_lr],verbose=1)
    plot_model(model, to_file='mlp_model.png', show_shapes=True, show_layer_names=True)
    # 預測
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    # 評估模型
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()  # 計算準確率

    with open("final_model_accuracy.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Accuracy","rs","lr","ep","status"])
        writer.writerow([current_date, accuracy, rs, lr, ep,1])

    plt.rcParams['font.family'] = 'Arial'
    # 繪製訓練與驗證的損失並保存圖片
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.close()

    # 繪製訓練與驗證的準確率並保存圖片
    plt.plot(history.history['accuracy'], label='training_accuracy')
    plt.plot(history.history['val_accuracy'], label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.close()

    return model,scaler,X

    # 查看權重
    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print(f"Layer: {layer.name}")
    #     for i, weight in enumerate(weights):
    #         print(f"  Weight {i}: shape {weight.shape}")
    #         print(f"  {weight}\n")

# 讀取數據
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')
features_df = pd.read_csv('Phishing_Dataset1_features.csv')

rs = 1337
lr = 0.000003
ep = 20
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
train_model,scaler,X = model(data,features_df,rs,lr,ep,current_date)

# # 儲存模型和標準化器
# train_model.save("./model1/final_model.keras")
# joblib.dump(scaler, "./model1/final_model.pkl")
# joblib.dump(X, "./model1/final_X.pkl")
# print("模型和標準化器已成功保存")

# path = f"./model2/{current_date}/"
# os.makedirs(path, exist_ok=True)
# # # 儲存模型和標準化器
# train_model.save(path+"final_model2.keras")
# joblib.dump(scaler, path+"final_model2.pkl")
# joblib.dump(X, path+"final2_X.pkl")
# print("模型和標準化器已成功保存")
