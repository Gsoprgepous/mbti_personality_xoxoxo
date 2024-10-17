import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

# Excel because I'm going to improve the dataset and expend it
file_path = '/content/mbti.xlsx'
data = pd.read_excel(file_path)

adjective_columns = [f'adjective{i+1}' for i in range(8)]
mbti_column = 'MBTI_type'
career_column = 'career_inclination'

# One-Hot Encoding for adjectives
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
adjectives_encoded = enc.fit_transform(data[adjective_columns])

# Data normalization to improve performance
scaler = StandardScaler()
adjectives_encoded = scaler.fit_transform(adjectives_encoded)

# Label Encoding for MBTI type
mbti_encoder = LabelEncoder()
mbti_encoded = mbti_encoder.fit_transform(data[mbti_column])

# this is our target variable
career_inclination = data[career_column]

X = pd.DataFrame(adjectives_encoded)

# separation into training and test samples
X_train, X_test, y_train_mbti, y_test_mbti = train_test_split(X, mbti_encoded, test_size=0.5, random_state=42)
_, _, y_train_career, y_test_career = train_test_split(X, career_inclination, test_size=0.5, random_state=42)

# making better neural network
input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))

# working with shared layers
x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# output for MBTI type prediction
mbti_output = tf.keras.layers.Dense(len(mbti_encoder.classes_), activation='softmax', name='mbti_output')(x)

# output for "level of career ambition" prediction
career_output = tf.keras.layers.Dense(3, activation='softmax', name='career_output')(x)

# creating a model
model = tf.keras.Model(inputs=input_layer, outputs=[mbti_output, career_output])

# compiling the model
model.compile(optimizer='adam',
              loss={'mbti_output': 'sparse_categorical_crossentropy', 'career_output': 'sparse_categorical_crossentropy'},
              metrics={'mbti_output': 'accuracy', 'career_output': 'accuracy'})

# model training
history = model.fit(X_train, {'mbti_output': y_train_mbti, 'career_output': y_train_career},
                    epochs=250,
                    batch_size=64,
                    validation_split=0.4)

# model evaluation
loss, mbti_accuracy, career_accuracy = model.evaluate(X_test, {'mbti_output': y_test_mbti, 'career_output': y_test_career})
print(f"Точность модели на тестовой выборке (MBTI): {mbti_accuracy * 100:.2f}%")
print(f"Точность модели на тестовой выборке (карьерная склонность): {career_accuracy * 100:.2f}%")

# prediction
def predict_mbti_and_career(student_data):
    # I've decided we gonna have 8 adjectives since that's my eighth repository and 8- means infinity!
    adjectives = student_data[:8]

    # One-Hot Encoding of adjectives and normalization
    adjectives_encoded = enc.transform([adjectives])
    adjectives_encoded = scaler.transform(adjectives_encoded)

    # prediction again
    mbti_pred, career_pred = model.predict(adjectives_encoded)

    # converting predictions into understandable values
    mbti_type = mbti_encoder.inverse_transform([tf.argmax(mbti_pred, axis=1).numpy()[0]])[0]
    career_inclination = tf.argmax(career_pred, axis=1).numpy()[0]

    return mbti_type, career_inclination

# an example
new_student = ['uncurious', 'restless', 'caring', 'emotional', 'nurturing', 'terse', 'shy', 'blushing']
mbti_prediction, career_prediction = predict_mbti_and_career(new_student)
print(f"Predicted MBTI type: {mbti_prediction}")
print(f"Predicted career inclination: {career_prediction}")
