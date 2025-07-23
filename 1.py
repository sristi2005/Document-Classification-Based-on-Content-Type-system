import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load datasets AG News Classification Dataset
train_path = 'C:/Users/ankon/Downloads/archive (3)/train.csv'
test_path = 'C:/Users/ankon/Downloads/archive (3)/test.csv'

# Load training and testing datasets
columns = ['Label', 'Title', 'Description']
train_data = pd.read_csv(train_path, header=None, names=columns)
test_data = pd.read_csv(test_path, header=None, names=columns)

# Combine title and description for processing
train_data['Text'] = train_data['Title'] + " " + train_data['Description']
test_data['Text'] = test_data['Title'] + " " + test_data['Description']

# Define stop words
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join([char.lower() if char.isalnum() else ' ' for char in text])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
train_data['Text'] = train_data['Text'].apply(preprocess_text)
test_data['Text'] = test_data['Text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['Text'])
X_test = vectorizer.transform(test_data['Text'])

# Labels
y_train = train_data['Label']
y_test = test_data['Label']

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model and vectorizer (for deployment)
import pickle
with open('document_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully!")
