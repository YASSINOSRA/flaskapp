import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load data
df = pd.read_csv("train.csv")

# Drop unwanted columns
df.drop(columns=["author", "title", "id"], inplace=True, errors='ignore')

# Check for missing values
df.dropna(inplace=True)

# Initialize NLTK Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

# Preprocess function
def preprocess(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Lemmatization and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess)

# Check class distribution in the data
sb.countplot(x='label', data=df, palette='hls')
plt.title('Class Distribution in Training Data')
plt.show()

# Split data into features and labels
X = df['text']
y = df['label']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Initialize and train the model
model = PassiveAggressiveClassifier(class_weight='balanced')  # Handling class imbalance
model.fit(X_train_vect, y_train)

# Save the model and vectorizer using pickle
with open("m3.pkl", "wb") as f:
    pickle.dump(model, f)

with open("v3.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modèle et vecteur sauvegardés avec succès.")

# Evaluate the model's performance
y_pred = model.predict(X_test_vect)
accuracy = np.round(np.mean(y_pred == y_test) * 100, 2)
print(f"Accuracy: {accuracy}%")

# Classification report and confusion matrix
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Define the function for fake news detection
def fake_news_det(news):
    # Preprocess the input text
    review = re.sub(r'[^a-zA-Z\s]', '', news.lower())  # Remove non-alphabetic characters
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    input_data = [' '.join(corpus)]

    # Load the trained model and vectorizer
    loaded_model = pickle.load(open("m3.pkl", "rb"))
    vectorizer = pickle.load(open("v3.pkl", "rb"))

    # Transform the input data using the vectorizer
    vectorized_input_data = vectorizer.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(vectorized_input_data)
    
    if prediction[0] == 1:
        return "Prédiction de l'information : Faux"
    else:
        return "Prédiction de l'information : Vrai"

# Test the function with a sample news article
news = "Les humains utilisent seulement 10% de leur cerveau"
print(fake_news_det(news))
