import streamlit as st
import pickle
import wget
import keras
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import os
from langdetect import detect
# Download/load model and encoder function
@st.cache_resource
def get_encoder_and_model():
    """Returns objects `LabelEncoder` and `keras.Model`

    Returns:
        tuple: (encoder, model)
    """
    folder_name='./streamlit_app/model/'
    encoder_filename = folder_name + 'le_encoder.pkl'
    model_filename = folder_name + 'bert_tuned.h5'
    # Download model and encoder
    if not os.path.exists(folder_name): 
        os.mkdir(folder_name)
        url_encoder = r'https://drive.google.com/u/0/uc?id=1qv5G1m1f9vPAVEYT6CauRzIdtxAWkwLi&export=download&confirm=yes'
        url_model = r'https://drive.google.com/u/0/uc?id=16AJS38-rJ5HOfRtK9eJnHdhyf9G5UFDi&export=download&confirm=yes'
        wget.download(url_encoder, out=encoder_filename)
        wget.download(url_model, out=model_filename)
    # Deserialize encoder
    with open(encoder_filename, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    # Load pretrained model
    model = keras.models.load_model(model_filename, custom_objects=dict(KerasLayer=hub.KerasLayer))
    # Return encoder and model
    return label_encoder, model
# Load encoder and model
label_encoder, model = get_encoder_and_model()
# Lambda function for prediction
prediction = lambda desc: label_encoder.inverse_transform(model.predict(desc, verbose=0).argmax(axis=1))[0]
# Show title
st.title('Предсказание жанра фильма по его описанию')
# Show text input
description = st.text_area('Введите описание фильма', '', key='desc', height=450)
# Add clear button
def clear_btn():
    st.session_state.desc = ''
st.button('Очистить', key='clear', on_click=clear_btn)
# Validation and show prediciton
if st.button('Предсказать жанр', key='prediction'):
    if len(description.split()) >= 20:
        if detect(description) == 'en':
            st.write(f'#### Предсказанный жанр: **{prediction([description])}**')
        else:
            st.write('### Описание должно быть на английском языке')
    else:
        st.write('### Описание должно состоять как минимум из 20 слов')