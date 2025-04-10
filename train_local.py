import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Charger les données (assure-toi d'avoir le fichier train.csv à la racine ou modifie le chemin)
df = pd.read_csv("train.csv")
df.drop(columns=["author", "title", "id"], inplace=True, errors='ignore')
df.dropna(inplace=True)

# Nettoyage des textes
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

def preprocess(text):
    words = text.lower().split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

df['text'] = df['text'].astype(str).apply(preprocess)

# Séparation texte / label
X = df['text']
y = df['label']

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Entraînement du modèle
model = PassiveAggressiveClassifier()
model.fit(X_vect, y)

# Sauvegarde dans le bon environnement local
with open("ya.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vec.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modèle et vecteur sauvegardés avec succès.")
