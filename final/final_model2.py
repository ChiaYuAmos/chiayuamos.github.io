# 數據處理與分析
import pandas as pd
import numpy as np
import re
import csv
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# 網頁抓取與處理
from bs4 import BeautifulSoup
import requests

# 機器學習相關
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 深度學習相關
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau

# 視覺化
import matplotlib.pyplot as plt

# 儲存與加載模型
import joblib

# 伺服器與網絡
import os 
import http.server
import socketserver

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
            if response.d_code == 200:
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

def model(data,features_df,rs,lr,ep,current_date):
    features_df = pd.get_dummies(features_df, columns=['tld'])
    X = features_df
    y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)
    print(f"Length of X: {len(X)}")
    print(f"Length of y: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent') 
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 建立 MLP 模型
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.04)))
    model.add(Dropout(0.5))  
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.04)))
    model.add(Dropout(0.5))  
    model.add(Dense(1, activation='sigmoid'))

    # 編譯模型
    # model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1)
    # # 訓練模型
    history = model.fit(X_train, y_train, epochs=ep, batch_size=32, validation_split=0.2, callbacks=[reduce_lr],verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    # 評估模型
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()  
    file_path = "final_model_accuracy.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Date", "Accuracy", "rs", "lr", "ep","state"])
        writer.writerow([current_date, accuracy, rs, lr, ep,0])

    show_picture(history,current_date)

    return model,scaler,X

def show_picture(history,current_date):
    path = f"./model2/{current_date}/"
    if not os.path.exists(path):
        os.makedirs(path)
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Arial'
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'loss_plot.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    # plt.show()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+'accuracy_plot.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    # plt.show()

def compare():

    file_path = "final_model_accuracy.csv"
    df = pd.read_csv(file_path)

    if df['Accuracy'].max() < 0.96:
        max_accuracy_index = df['Accuracy'].idxmax() 
        df['state'] = 0 
        df.loc[max_accuracy_index, 'state'] = 1  
        df.to_csv(file_path, index=False)  
    else:
        print("未進行更新，因為準確率高於或等於 96%")

def load_best_model():
    file_path = "final_model_accuracy.csv"
    df = pd.read_csv(file_path)
    
    best_model_row = df[df['state'] == 1]
    
    if best_model_row.empty:
        print("沒有找到標註為最佳模型的行")
        return None
    
    model_date = best_model_row['Date'].values[0]
    
    model_path = f"./model2/{model_date}/"
    print(f"Attempting to load model from: {model_path}")
    
    
    if not os.path.exists(model_path):
        print(f"模型路徑 {model_path} 不存在")
        return None

    try:
        
        model = load_model(model_path + "final_model2.keras")
        scaler = joblib.load(model_path +"final_model2.pkl")
        X = joblib.load(model_path +"final2_X.pkl")
        print("成功載入最佳模型及相關檔案")
        return model, scaler, X
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")



data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')
data2 = pd.read_csv('Phishing_Dataset2.csv', encoding='utf-8')
features_df = pd.read_csv('Phishing_Dataset1_features.csv')
features_df2 = pd.read_csv('Phishing_Dataset2_features.csv')
combined_data =  pd.concat([data, data2], ignore_index=True)
combined_df = pd.concat([features_df, features_df2], ignore_index=True)

rs = 1337
lr = 0.000003
ep = 20
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model2,scaler,X = model(combined_data,combined_df,rs,lr,ep,current_date)
path = f"./model2/{current_date}/"
os.makedirs(path, exist_ok=True)

model2.save(path+"final_model2.keras")
joblib.dump(scaler, path+"final_model2.pkl")
joblib.dump(X, path+"final2_X.pkl")
print("模型和標準化器已成功保存")


compare()
result = load_best_model()

if result:
    best_model, best_scaler, best_X = result
    best_model.save("final_model.keras")
    joblib.dump(best_scaler,"final_model.pkl")
    joblib.dump(best_X, "final_X.pkl")
else:
    print("未能成功載入最佳模型")

