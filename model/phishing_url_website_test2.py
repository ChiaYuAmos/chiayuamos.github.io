import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse

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
        'NumWWW': url.count('www')
    }
    return features

# Load and prepare dataset
dataframe = pd.read_csv('phishing_url_website.csv')
dataframe = dataframe.drop(columns=['URL', 'Domain', 'Title']).dropna()

le = LabelEncoder()
dataframe['TLD'] = le.fit_transform(dataframe['TLD'])

X = dataframe.drop(columns=['label'])
y = dataframe['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Predict function with feature extraction
while True:
    test_url = input("Enter a URL to test (type 'exit' to quit): ")
    if test_url.lower() == "exit":
        break
    
    test_url_data = extract_features_from_url(test_url)
    test_url_df = pd.DataFrame([test_url_data])

    # Ensure feature order and include all features from training
    for feature in X.columns:
        if feature not in test_url_df.columns:
            test_url_df[feature] = 0  # Set missing features to 0
    
    test_url_df = test_url_df[X.columns]  # Ensure columns order is correct
    
    test_url_df['TLD'] = le.transform(test_url_df['TLD'])

    # Predict using the model
    prediction = model.predict(test_url_df)
    
    if prediction[0] == 1:
        print('Phishing site detected.')
    else:
        print('Legitimate site.')