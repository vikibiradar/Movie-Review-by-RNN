import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model 

#Load the IMDB
word_index=imdb.get_word_index()
reversed_word_index={value:key for key,value in word_index.items()}

#Loading Model
model=load_model('imbd_model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#Function to preprocess user input 
def preprocess_text(text):
    words=text.lower().split()
    encoded_review= [word_index.get(word,2) + 3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review 

# Step 3
## Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

#Step 4

import streamlit as st
#Streamlit app
st.set_page_config(page_title="Movie Sentiment Predictor", layout="wide")
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a Movie Review to classify it as positive or negative")
#user input
user_input=st.text_input('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    #Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please enter a movie Review')


