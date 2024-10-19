from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import streamlit as st

model = load_model('model_.keras')
with open('tokenizer.pkl', 'rb') as file :
    tokenizer = pickle.load(file)
max_sequence_len = 20
st.header('Next Word Predictor')
seed_text = st.text_input('Enter the text')
next_words = st.number_input("Enter the number of words to be predicted", min_value=0, max_value=100, value=0, step=1)
if(st.button('Predict')) :
    if(len(seed_text) > 0) :
        for _ in range(next_words) :
            sequence = tokenizer.texts_to_sequences([seed_text])
            padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len-1)
            predicted = model.predict(padded_sequence, verbose = 0)
            predicted = np.argmax(predicted)
            output_word = ''
            for word, index  in tokenizer.word_index.items() :
                if index == predicted :
                    output_word = word
                    break
            seed_text += ' ' + output_word
        st.subheader('Predicted output is')
        st.write(seed_text)
    else :
        st.write('Enter a valid text')
