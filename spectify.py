import librosa
import numpy as np
import os

path_to_ogg = 'openmic-2018/audio/000'

# convert all .ogg files to log-scaled mel spectrograms

# load label file
labels = np.loadtxt('openmic-2018/openmic-2018-aggregated-labels.csv', delimiter=',', dtype=str)

# loop through all .ogg files
for filename in os.listdir(path_to_ogg):
    if filename.endswith(".ogg"):
        # load audio file
        y, sr = librosa.load(path_to_ogg + '/' + filename, sr=44100)
        # compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        # convert to log scale
        log_S = librosa.power_to_db(S, ref=np.max)
        # save as .npy file in /spectrograms
        np.save('spectrograms/' + filename[:-4] + '.npy', log_S)
        continue
    else:
        continue