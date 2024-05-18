import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse
import numpy as np

# Function to extract features from URL
def extract_features_from_url(url):
    parsed_url = urlparse(url)
    features = {
        'TLD': parsed_url.netloc.split('.')[-1],
        'NumDots': url.count('.'),
        'Length': len(url),
        'NumDash': url.count('-'),
        'NumAt': url.count('@'),
        'NumAmpersand': url.count('&'),
        'NumEquals': url.count('='),
        'NumHash': url.count('#'),
        'NumSlash': url.count('/'),
        'NumQuestionMark': url.count('?'),
        'NumPercent': url.count('%'),
        'NumAsterisk': url.count('*'),
        'NumDollar': url.count('$'),
        'NumSpace': url.count(' '),
        'NumExclamation': url.count('!'),
        'NumTilde': url.count('~'),
        'NumComma': url.count(','),
        'NumSemicolon': url.count(';'),
        'NumColon': url.count(':'),
        'NumHttp': url.count('http'),
        'NumHttps': url.count('https'),
        'NumWWW': url.count('www'),
        'NumDigits': sum(c.isdigit() for c in url),
        'NumLetters': sum(c.isalpha() for c in url),
        'HasIP': int(any(part.isdigit() for part in parsed_url.netloc.split('.'))),
        'KeywordLogin': int('login' in url.lower()),
        'KeywordSecure': int('secure' in url.lower()),
        'KeywordBank': int('bank' in url.lower())
    }
    return features

# Load and prepare dataset
dataframe = pd.read_csv('phishing_url_website.csv')
dataframe = dataframe.drop(columns=['URL', 'Domain', 'Title']).dropna()

le = LabelEncoder()
train_tlds = dataframe['TLD']
le.fit(train_tlds)

# Save known classes and include a default label
known_classes = list(le.classes_)
known_classes.append('unknown')
le.classes_ = np.array(known_classes)

dataframe['TLD'] = dataframe['TLD'].apply(lambda x: x if x in known_classes else 'unknown')
dataframe['TLD'] = le.transform(dataframe['TLD'])

X = dataframe.drop(columns=['label'])
y = dataframe['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance with cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(cv_scores)}')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Predict function with feature extraction
true_website = []
phishing_website = []

with open("0_testwebsite", 'r', encoding='utf-8') as file:
    content = file.read()
website = content.split('\n')
print(website)

for i in website:
    if "http" not in i:
        continue

    test_url = i 
    test_url_data = extract_features_from_url(test_url)
    test_url_df = pd.DataFrame([test_url_data])

    # Ensure feature order and include all features from training
    for feature in X.columns:
        if feature not in test_url_df.columns:
            test_url_df[feature] = 0  # Set missing features to 0

    test_url_df = test_url_df[X.columns]  # Ensure columns order is correct

    # Handle unseen TLDs
    if test_url_df['TLD'].iloc[0] not in le.classes_:
        print(f"Unseen TLD detected: {test_url_df['TLD'].values}. Assigning 'unknown'.")
        test_url_df['TLD'] = 'unknown'
    
    test_url_df['TLD'] = le.transform(test_url_df['TLD'])

    # Predict using the model
    prediction = model.predict(test_url_df)

    if prediction[0] == 1:
        phishing_website.append(i)
    else:
        true_website.append(i)

# Final output
print('True Website:', true_website)
print('Phishing Website:', phishing_website)