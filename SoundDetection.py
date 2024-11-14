from collections import defaultdict

from matplotlib import pyplot as plt
import sounddevice as sd
from numpy import resize
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

data_dir = './samples'
classes = ['horns', 'not_identified', 'barks']

class ProcessingAudio:

    def __init__(self, models: list):
        self.audio_data_accumulated = []
        self.models: list = models
        self.device_id = ''

    def choose_device(self):
        print(f"Escolha o dispositivo de entrada:\n{sd.query_devices()}")
        self.device_id = int(input())
        print(self.device_id)

    def start_detection(self):
        samples_to_collect = 22050

        self.listen(self.device_id)

        predicted_class, prob = self.classify_sound(self.audio_data_accumulated.copy())

        print(f'The audio is classified as: {predicted_class}')

        if len(self.audio_data_accumulated) >= samples_to_collect:
            self.plot_spectrum(np.array(self.audio_data_accumulated), 22050)

            self.audio_data_accumulated = []

    def is_silence(self, audio_data, threshold=0.3):
        """
         Verifica se o áudio é basicamente silêncio com base no limiar
        """

        return np.max(np.abs(audio_data)) < threshold

    def listen(self, device_id):
        with sd.InputStream(callback=self.callback, device=device_id, channels=1, samplerate=22050):
            print("Escutando... Pressione Ctrl+C para parar.")
            time_active = 1  # seconds
            sd.sleep(time_active * 1000)
            print("Terminado.")
            print(len(self.audio_data_accumulated))

        return

    def classify_sound(self, audio_data):

        # Define the target shape for input spectrograms
        target_shape = (128, 128)

        mel_spectrogram = librosa.feature.melspectrogram(y=np.array(audio_data), sr=22050)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

        predictions = defaultdict(float)
        for m, c in self.models:
            curr_predictions = m.predict(mel_spectrogram)

            class_probabilities = curr_predictions[0]

            for i, class_label in enumerate(c):
                probability = class_probabilities[i]
                print(f'{class_label=} with {probability}')
                predictions[class_label] += probability

        class_predicted = ''
        curr_larger = 0
        print(predictions)
        for label, val in predictions.items():
            print(val)
            print(len(classes)-1)
            curr_mean = val / (len(classes) - 1)

            if curr_mean > curr_larger:
                class_predicted = label
                curr_larger = curr_mean

        print(f'{class_predicted=} with probability of {curr_larger} ')
        return class_predicted, curr_mean

    def callback(self, indata, frames, time, status, **kwargs):
        if status:
            print(status)
        audio_data = np.fromstring(indata, dtype=np.float32)

        audio_data = librosa.resample(y=audio_data, orig_sr=44100, target_sr=22050)
        self.audio_data_accumulated.extend(audio_data)

        if self.is_silence(indata):
            print("Silêncio.")
        else:
            print('Som detectado!')

    def plot_spectrum(self, audio_data, sr):
        fft_spectrum = np.fft.fft(audio_data)
        freq = np.fft.fftfreq(len(fft_spectrum), 1 / sr)

        plt.figure(figsize=(10, 4))
        plt.plot(freq[:len(freq) // 2], np.abs(fft_spectrum[:len(fft_spectrum) // 2]))
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Espectro de Frequência")
        plt.show()
