from tensorflow.keras.models import load_model
from SoundDetection import ProcessingAudio
import tensorflow as tf

models = []
all_classes = ['barks', 'horns', 'not_identified']

while len(all_classes) > 0:
    c1 = all_classes.pop()
    index = 0
    while index < len(all_classes) and len(all_classes) > 0:
        models.append(
            (
                load_model(f'audio_classification_model{[c1, all_classes[index]]}.h5'),
                [c1, all_classes[index]],
            )
        )
        index += 1

sound_detection = ProcessingAudio(models)
sound_detection.choose_device()
for i in range(4):
    sound_detection.start_detection()
