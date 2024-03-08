import tensorflow as tf
from data_preprocessor import input_size, maxlen

# Define the model
model = tf.keras.models.Sequential([
    # Embedding layer
    tf.keras.layers.Embedding(input_dim=input_size, output_dim=50, input_length=maxlen),
    # Dropout
    tf.keras.layers.Dropout(0.5),
    # Bidirectional GRU layers
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(120, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    # Batch Normalization
    tf.keras.layers.BatchNormalization(),
    # Bidirectional GRU layer
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)),
    # Flatten the output
    tf.keras.layers.Flatten(),
    # Dense layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Output layer
    tf.keras.layers.Dense(6, activation='softmax')
])