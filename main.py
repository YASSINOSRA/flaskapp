import pandas as pd
import sklearn
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./templates', static_folder='./static')
run_with_ngrok(app)

# Load the trained model and vectorizer
loaded_model = pickle.load(open("m3.pkl", 'rb'))
vector = pickle.load(open("v3.pkl", 'rb'))

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('french'))

# Function to predict fake news
def fake_news_det(news):
    # Preprocess the news
    review = re.sub(r'[^a-zA-Z\s]', '', news)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = nltk.word_tokenize(review)  # Tokenization
    
    # Lemmatize and remove stopwords
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]
    
    # Convert the cleaned text into a feature vector
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    
    # Make the prediction
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for About page
@app.route('/about')
def info():
    return render_template('info.html')

# Route for Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        
        # Interpret the prediction
        if pred[0] == 1:
            result = "Prédiction de l'information : Faux"
        else:
            result = "Prédiction de l'information : Vrai"
        
        return render_template("prediction.html", prediction_text=result)
    else:
        return render_template('prediction.html', prediction_text="Quelque chose s'est mal passé")

# Run the Flask application
if __name__ == '__main__':
    # Remove debug argument here
    app.debug = True
    app.run()


