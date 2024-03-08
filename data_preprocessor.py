from data_ingestion import df
from data_cleaner import preprocess_text
import tensorflow as tf
import numpy as np
import string
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

df = df.sample(n=50000, random_state=42)

df = df.reset_index()
df = df.drop(columns=['index','Unnamed: 0'], axis=1)
df.dropna(subset=['text', 'label'])
class_labels = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear' : 4, 'surprise': 5 }


df['clean_text'] = df['text'].apply(preprocess_text)


# Load the tokenizer
with open('models\\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

feature_seq = tokenizer.texts_to_sequences(df.clean_text)

# Max Len in X_train_sequences
maxlen = max(len(tokens) for tokens in feature_seq)


# Perform padding on X_train and X_test sequences
features_padded = tf.keras.preprocessing.sequence.pad_sequences(feature_seq, maxlen=maxlen, padding='post',)

input_size = np.max(features_padded) + 1

