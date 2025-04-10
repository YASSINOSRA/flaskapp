import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Télécharger les ressources NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Charger les données (assure-toi d'avoir le fichier train.csv à la racine ou modifie le chemin)
df = pd.read_csv("train.csv")
df.drop(columns=["author", "title", "id"], inplace=True, errors='ignore')
df.dropna(inplace=True)

# Initialisation du lemmatizer et des stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

# Fonction de nettoyage des textes
def preprocess(text):
    # Supprimer les caractères non alphabétiques et mettre en minuscule
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenisation des mots
    words = nltk.word_tokenize(text)
    
    # Lemmatization et suppression des stop words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Appliquer la fonction de nettoyage sur la colonne 'text'
df['text'] = df['text'].astype(str).apply(preprocess)

# Séparation du texte et des labels
X = df['text']
y = df['label']

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Entraînement du modèle
model = PassiveAggressiveClassifier()
model.fit(X_vect, y)

# Sauvegarde du modèle et du vecteur dans des fichiers pickle
with open("m1.pkl", "wb") as f:
    pickle.dump(model, f)

with open("v1.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modèle et vecteur sauvegardés avec succès.")
