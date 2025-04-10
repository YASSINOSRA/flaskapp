from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__,template_folder='./templates',static_folder='./static')

loaded_model = pickle.load(open("m1.pkl", 'rb'))
vector = pickle.load(open("v1.pkl", 'rb'))


lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('french'))
corpus = []

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
     
    return prediction

        

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def info():
    return render_template('info.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        print(pred)
        def predi(pred):
            if pred[0] == 1:
              res=" Prédiction de l'information : Faux"
            else:
              res="Prédiction de l'information : Vrai"
            return res
        result=predi(pred)
        return render_template("prediction.html",  prediction_text="{}".format(result))
    else:
        return render_template('prediction.html', prediction="Something went wrong")



if __name__ == '__main__':
    app.run(debug=True)