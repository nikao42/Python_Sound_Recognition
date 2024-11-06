'''
Bibliotecas:
numpy (manipulação matemática)
librosa (manipulação de áudio)
numba (necessário para librosa)
sklearn (treinamento do modelo)
sounddevice (escuta ao vivo de áudio)
scipy (necessário para sounddevice)
clara ()
pip install numpy librosa numba sklearn sounddevice scipy clara 
'''

samples = "/Users/nikao/School/UFRJ/2024.2/Telecom/samples"

import librosa
import numpy as np
from matplotlib import pyplot as plt
import time as t 
import wave

def model_train(samples):
    print("'model_train()' started.")
    '''
    #1
    # Carregar o arquivo de áudio
    audio_path = 'audio/buzina.wav' #WAV ou MP3
    y, sr = librosa.load(audio_path)

    # Extrair MFCCs (MFCCs = Mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(mfccs.shape)  # (n_mfcc, frames)
    '''


    #2 - criação do modelo
    from sklearn.ensemble import RandomForestClassifier
    import librosa
    import numpy as np
    import os

    def extract_features(file_path):
        # Carrega o áudio (amostragem de 44.1kHz)
        audio, sample_rate = librosa.load(file_path, sr=44100)
        print(f"New file: {os.path.dirname(file_path)}")
        
        # Extrai MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        print(f"MFCCs: {mfccs.shape}")
        
        # Achata o array de MFCCs para transformar em vetor
        mfccs_flatten = np.mean(mfccs.T, axis=0)
        
        return mfccs_flatten

    def load_data(data_path):
        print("load_data() started.")
        features = []
        labels = []

        for class_label in os.listdir(data_path):
            print(f"New class: {class_label}")
            class_path = os.path.join(data_path, class_label)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(class_label)

        return np.array(features), np.array(labels)

    X, y = load_data(samples)  # Carregando as características e rótulos
    print(f"X: {X}")
    print(f"Y: {y}")


    #3 - divisão de dados
    from sklearn.model_selection import train_test_split

    # Divida os dados em 80% para treinamento e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    #4 - treinamento do modelo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Cria e treina o modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Testa o modelo no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliação da precisão
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisão: {accuracy * 100:.2f}%')



    #5 - métricas do modelo (apenas para análise)
    from sklearn.metrics import confusion_matrix, classification_report

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #5.1 - validação cruzada - Para garantir que o modelo esteja generalizando bem e não overfitting, você pode usar validação cruzada. (eu não sei q porra eh essa, ta falando q é opcional então...)
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X, y, cv=5)  # Validação cruzada 5-fold
    print(f'Média de precisão: {scores.mean() * 100:.2f}%')
    return model

def audio_process():
    #6 - processamento do áudio:
    import sounddevice as sd
    import numpy as np
    import librosa
    print("Escolha o dispositivo de entrada:", sd.query_devices())
    device_id = int(input())
    
    def process_audio(audio_data):
        # Extrai as características do bloco de áudio
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13)
        return mfccs

    '''
    Assumindo que você já treinou seu modelo e ele está carregado
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()  # Substitua com o modelo que você treinou
    '''

    def classify_sound(audio_data):
        # Processa o áudio (extrai MFCCs ou outras características)
        features = process_audio(audio_data)
        features = features.flatten()  # Achatar para alimentar no modelo

        # Realiza a predição
        prediction = model.predict([features])
        return prediction

    def callback(indata, frames, time, status):
        # Verifica se há erros
        if status:
            print(status)

        # Captura o áudio e converte em um array numpy
        audio_data = np.squeeze(indata)

        # Processa o áudio (exemplo: extrair MFCCs)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13)

        # Aqui, você poderia passar as características extraídas para o seu modelo de ML
        prediction = model.predict(mfccs)
        print(f'Som detectado: {prediction}')

    # Configura a captura de áudio (44100 Hz, canal mono)
    with sd.InputStream(callback=callback, device = device_id, channels=1, samplerate=44100):
        print("Escutando... Pressione Ctrl+C para parar.")
        time_active = 10 #seconds
        sd.sleep(time_active*1000)  # Tempo de gravação em milissegundos

audio_data_accumulated = []
def teste_som(model):
    global audio_data_accumulated
    #6 - processamento do áudio:
    import sounddevice as sd
    import numpy as np
    import librosa
    from sklearn.metrics import accuracy_score, classification_report

    print(f"Escolha o dispositivo de entrada:\n{sd.query_devices()}")
    device_id = int(input())
    
    def is_silence(audio_data, threshold=0.3):
        # Verifica se o áudio é basicamente silêncio com base no limiar
        return np.max(np.abs(audio_data)) < threshold

    def process_audio(audio_data):
        # Extrai as características do bloco de áudio
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13,n_fft=512, fmax=5000,n_mels=32)
        mfccs_mean = np.mean(mfccs, axis=1)  # Calcula a média dos coeficientes ao longo do tempo
        return mfccs_mean

    '''
    Assumindo que você já treinou seu modelo e ele está carregado
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()  # Substitua com o modelo que você treinou
    '''
    
    

    def classify_sound(audio_data):
        # Processa o áudio (extrai MFCCs ou outras características)
        features = process_audio(audio_data)

        # Realiza a predição
        prediction = model.predict([features])[0]

        # Realiza a predição com probabilidades
        prediction_probabilities = model.predict_proba([features])[0]  # Retorna as probabilidades das classes
        predicted_class = model.predict([features])[0]  # Classe com maior probabilidade
        print(prediction_probabilities,predicted_class)

        # Encontra a probabilidade associada à classe predita
        confidence = max(prediction_probabilities)

        print(f"Som detectado: {predicted_class} com confiança de {confidence * 100:.4f}%")


        return prediction
    
        
    def callback(indata, frames, time, status,**kwargs):
        # Verifica se há erros
        if status:
            print(status)

        # Captura o áudio e converte em um array numpy
        
        global audio_data_accumulated

        audio_data = np.squeeze(indata)     
        audio_data_accumulated.extend(audio_data)
        
        if is_silence(audio_data):
            print("Silêncio detectado. Ignorando.")
            return  # Ignora a predição se for silêncio
        
        # Processa e realiza a predição
        prediction = classify_sound(audio_data)   
                
        # Imprime o resultado da predição
        print(f"Predição em tempo real: {prediction}")
        
        # Calcula e imprime a média do áudio capturado
        mean_amplitude = np.mean(audio_data)
        print(f"Média do áudio capturado: {mean_amplitude}")
        
        
        
        
        audio_data_accumulated.extend(audio_data)

        # Aqui, você poderia passar as características extraídas para o seu modelo de ML
        #prediction = model.predict(mfccs)
        #print(f'Som detectado: {prediction}')
        
    

    # Configura a captura de áudio (44100 Hz, canal mono)
    with sd.InputStream(callback=callback, device = device_id, channels=1, samplerate=44100):
        print("Escutando... Pressione Ctrl+C para parar.")
        time_active = 1 #seconds
        sd.sleep(time_active*1000)
        print("Terminado.") 
        print(len(audio_data_accumulated))
        
    # Adiciona o novo bloco de áudio ao acumulador
    return None
    
def plot_spectrum(audio_data,sr):
    # Calcula a FFT do áudio para obter o espectro de frequência
    fft_spectrum = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(fft_spectrum), 1/sr)
    
    # Exibe apenas a metade positiva do espectro (frequências reais)
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:len(freq)//2], np.abs(fft_spectrum[:len(fft_spectrum)//2]))
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Espectro de Frequência")
    plt.show()
samples_to_collect = 44100

# Verifica se acumulou amostras suficientes para o período desejado
if len(audio_data_accumulated) >= samples_to_collect:
    # Processa o espectro de frequência
    plot_spectrum(np.array(audio_data_accumulated),44100)
    
    # Limpa o acumulador para reiniciar a coleta
    audio_data_accumulated = []

def save_audio_to_wav(audio_data, sr):
    # Converte o áudio para 16-bit int
    audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)  # Normaliza e converte
    
    # Salva o arquivo WAV
    wav.write("audio_output.wav", sr, audio_data_int16)  # Grava o arquivo WAV
    print("Áudio salvo como 'audio_output.wav'")
                    
#plt.plot(audio_data_accumulated)
#plt.show()


teste_som(model_train(samples))


#model_train()
