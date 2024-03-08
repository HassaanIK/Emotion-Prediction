# Emotion Prediction Model

## Overview
This project aims to predict the emotion conveyed in a given text using a deep learning model. The model is trained on a dataset containing 400,000 text samples labeled with six emotions: sad, anger, surprise, love, joy, and fear.

## Technologies
- Python
- NLTK
- TensorFlow
- Flask

## Steps
1. **Data Collection:** The data used for this project is taken from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions).
2. **Data Preprocessing:** Many preprocessing steps are applied on the data to prepare it for training.
3. **Model Architecture:**
    - Built a deep learning model having an `Embedding layer` followed by 4 `Bidirectional GRU` layers having 2M parameters.
    - `Dropout` and `BatchNormalization` layers were also used to prevent overfitting and for regularization.
    - In the last `Flatten` layer followed by two `FC`(FullyConnected) layers were used to get the prediction 
5. **Training:** Trained the model on the prepared data and used techniques like `learning rate scheduling` and `early stopping`.
6. **Evaluation:** Evaluated the model's performance on a separate validation set and achieved **94%** accuracy.
7. **Deployment:** Deployd the trained model as a web app using `Flask`.
8. **Usage:** Use the web application to input text and predict the corresponding emotion.

## Techniques
- Cleaning Text: Remove any unwanted words from the text(eg. stopwords).
- Tokenization: Convert text into tokens for model input.
- Padding: Padd the text to have a same length for every inputs.
- Deep Learning: Use a deep learning model for emotion prediction.
- Flask: Build a web application for interacting with the model.

## Outcomes
- Trained a deep learning model to predict emotions in text.
- Deployed the model as a web application for easy usage.
- This model can be used in various applications such as sentiment analysis in customer reviews, emotion detection in social media posts, and personal assistant applications to understand user emotions.

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Download the pre-trained model weights and place them in the `models/` directory.
3. Run the Flask web application using `python app.py`.
4. Access the application in your web browser at `http://localhost:5000`.

## Web App
![Screenshot (35)](https://github.com/HassaanIK/Emotion-Prediction/assets/139614780/95b4ebbe-7b95-4b9d-afb0-7b0abeb42f61)
![Screenshot (36)](https://github.com/HassaanIK/Emotion-Prediction/assets/139614780/2df9abbe-edde-4f1e-a5ad-97eb3fc4a98a)

