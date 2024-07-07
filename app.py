import streamlit as st
import pickle
import numpy as np

# Load model dan vectorizer dari file pickle
with open('grid_search_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


# Fungsi untuk prediksi teks input
def predict(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return prediction[0]

# Aplikasi Streamlit
st.title('Klasifikasi Teks Sentimen dengan KNN')
st.write('Masukkan teks untuk diprediksi apakah itu positif, negatif atau netral.')

user_input = st.text_input('Masukkan teks:', '')

if st.button('Prediksi'):
    if user_input:
        prediction = predict(user_input)
        st.write(f'Prediksi: {prediction}')
    else:
        st.write('Silakan masukkan teks.')