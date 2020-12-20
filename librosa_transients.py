import librosa

x, sr = librosa.load('../sound-files/first-four-seconds.wav')
onset_frames = librosa.onset.onset_detect(x, sr=sr)
print(onset_frames)

onset_times = librosa.frames_to_time(onset_frames)
print(onset_times)