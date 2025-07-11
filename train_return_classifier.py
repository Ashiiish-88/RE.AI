import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
df = pd.read_csv('data/product_classification_dataset.csv')

# Features and target
text_features = ['product_name', 'return_reason', 'inspector_notes', 'customer_notes']
cat_features = ['warranty_valid', 'packaging_intact', 'category']
num_features = ['age_days']
target = 'classification'

# Encode categorical features
for col in cat_features:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Vectorize text features
vectorizers = {}
text_arrays = []
for col in text_features:
    vect = TfidfVectorizer(max_features=100)
    arr = vect.fit_transform(df[col].astype(str)).toarray()
    vectorizers[col] = vect
    text_arrays.append(arr)

X_text = np.concatenate(text_arrays, axis=1)
X_num = df[num_features + cat_features].values
X = np.concatenate([X_text, X_num], axis=1)
y = df[target]

# Encode target
y_le = LabelEncoder()
y_enc = y_le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

# Metrics
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    ]

train_metrics = get_metrics(y_train, train_pred)
test_metrics = get_metrics(y_test, test_pred)

# Display table
import pandas as pd
results = pd.DataFrame(
    [train_metrics, test_metrics],
    columns=['Accuracy', 'Precision', 'Recall', 'F1'],
    index=['Train', 'Test']
)
print('Classification Report (Train vs Test)')
print(results.round(4))
