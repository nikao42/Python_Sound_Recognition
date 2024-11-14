import os
from collections import defaultdict

import librosa
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model


def load_and_preprocess_data(classes, data_dir, target_shape=(128, 128)):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)

    return np.array(data), np.array(labels)


def training(classes):

    data_dir = './samples'
    print(f'Starting to train for classes {classes}')

    data, labels = load_and_preprocess_data(classes, data_dir)
    labels = to_categorical(labels, num_classes=len(classes))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    input_shape = X_train[0].shape
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(len(classes), activation='softmax')(x)
    model = Model(input_layer, output_layer)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Change the test to only look at the end (cross validation) -> kfold validation
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(test_accuracy[1])

    model.save(f'audio_classification_model{classes}.h5')

    return (model, classes)


all_classes = ['barks', 'horns', 'not_identified']

models = []

while len(all_classes) > 0:
    c1 = all_classes.pop()
    index = 0
    while index < len(all_classes) and len(all_classes) > 0:
        models.append(training([c1, all_classes[index]]))
        index += 1


def test_prediction(sound_data, classes_len):
    predictions = defaultdict(float)
    for m, c in models:
        curr_predictions = m.predict(sound_data)

        class_probabilities = curr_predictions[0]

        for i, class_label in enumerate(c):
            probability = class_probabilities[i]
            predictions[class_label] += probability

    ans = ''
    curr_larger = 0
    for label, val in predictions.items():
        curr_mean = val/(classes_len-1)

        if curr_mean > curr_larger:
            ans = label
            curr_larger = curr_mean

    return ans, curr_larger



classes=['barks', 'horns', 'not_identified']
right = 0
wrong = 0

for i, class_name in enumerate(classes):
    class_dir = os.path.join('./samples', class_name)
    print(f'# Doing for {class_dir}')
    for filename in os.listdir(class_dir):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(class_dir, filename)
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y=np.array(audio_data), sr=22050)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), (128, 128))
            mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + (128, 128) + (1,))

            class_predicted, prob = test_prediction(np.array(mel_spectrogram), len(classes))
            print(f'{class_predicted=} with {prob=} for {class_name=}')

            if class_predicted == class_name:
                right += 1
            else:
                wrong += 1

print(f'Model has prob of {right/(right+wrong)}')






