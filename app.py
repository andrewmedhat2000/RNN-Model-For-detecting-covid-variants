#our imports
from flask import Flask ,render_template,request
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional,SimpleRNN
import nltk
from nltk.corpus import stopwords
import pandas as pd
from flask import Flask
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
# prepare data
articles = []
labels = []
with open("final1 (7) (1).csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    #read file 
    for row in reader:
        # labels will have word of + or - 
        labels.append(row[1])
        # article has sequence 
        articles.append(row[3])
def appcreate():
    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
    @app.route("/",methods=["POST","GET"])

    def index():
        training_portion = 0.7
        train_size = int(len(articles) * training_portion)
        train_articles = articles[0: train_size]
        vocab_size = 1000
        max_length = 100
        oov_tok = '<OOV>' #  Out of Vocabulary
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        seqweb=request.form.get("seq")       
        if request.method == "POST":
            if ' ' in seqweb:
                seqget=seqweb
                model=tf.keras.models.load_model('model.h5')
                txt=[seqget]
                tokenizer.fit_on_texts(train_articles)
                seq = tokenizer.texts_to_sequences(txt)
                padded = pad_sequences(seq, maxlen=max_length)
                pred=model.predict(padded)
                predicted=0

                if pred>=0.5:
                    predicted=1
                if predicted==1:
                    return render_template("result.html")
                else:
                    return render_template("result negative.html")
            else:
                seqget = [seqweb[i:i+10] for i in range(0, len(seqweb), 10)]

                model=tf.keras.models.load_model('my_model')
                txt=[seqget]
                # this must produce 1 posiyive graeter than half
                tokenizer.fit_on_texts(train_articles)
                seq = tokenizer.texts_to_sequences(txt)
                padded = pad_sequences(seq, maxlen=max_length)
                pred=model.predict(padded)
                predicted=0

                if pred>=0.5:
                    predicted=1
                if predicted==1:
                    return render_template("result.html")
                else:
                    return render_template("result negative.html")
        
        return render_template("home.html")
    if __name__ == "__main__":
        app.debug = True
        app.run()
    return app
app =appcreate()