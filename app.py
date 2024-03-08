from flask import Flask, request, render_template, jsonify
from predict import predict_emotion
import tensorflow as tf
import pickle

# Load the tokenizer
with open('models\\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the model
model = tf.keras.models.load_model('models\\emotion_model_f.h5')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_emotion(text, model, tokenizer)
        print(f'Prediction: {prediction}')
        return render_template('result.html', prediction=prediction, text=text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)