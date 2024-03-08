from sklearn.model_selection import train_test_split
from data_preprocessor import features_padded, df
# Split the dataset into training (80%) and validation (20%) sets

X_train, X_val, y_train, y_val = train_test_split(features_padded, df.label, test_size=0.2, random_state=2007)
