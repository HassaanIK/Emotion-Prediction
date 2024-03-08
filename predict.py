import tensorflow as tf
import numpy as np
import pickle

def predict_emotion(text, model, tokenizer):

    tokenized_text = tokenizer.texts_to_sequences([text])

    # Pad sequences to the same length as your training data
    maxlen = 79  # Assuming maxlen is the same as your training data
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=maxlen, padding='post')
    
    # Make predictions
    predictions = model.predict(padded_text)
    predicted_class = np.argmax(predictions[0])
    class_labels = ['sad🙁','joy😀', 'love🥰', 'anger😠', 'fear😨', 'surprise😲']
    return class_labels[predicted_class]
