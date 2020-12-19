import os.path
import sys

import librosa
import pandas
import numpy
import sklearn.decomposition
from skimage.color import hsv2rgb, rgb2hsv

from matplotlib import pyplot as plt
import librosa.display
import seaborn

def decompose_audio(y, sr, bpm, per_beat=8, n_components=16, n_mels=128, fmin=200, fmax=6000):
    """
    Decompose audio using NMF spectrogram decomposition,
    using a fixed number of frames per beat (@per_beat) for a given @bpm
    NOTE: assumes audio to be aligned to the beat
    """
    
    interval = (60/bpm)/per_beat
    T = sklearn.decomposition.NMF(n_components)
    S = numpy.abs(librosa.feature.melspectrogram(y, hop_length=int(sr*interval), n_mels=128, fmin=200, fmax=6000))

    comps, acts = librosa.decompose.decompose(S, transformer=T, sort=True)
    return S, comps, acts

def plot_colorized_activations(acts, ax, hop_length=None, sr=None, value_mod=1.0):

    hsv = numpy.stack([
        numpy.ones(shape=acts.shape),
        numpy.ones(shape=acts.shape),
        acts,
    ], axis=-1)

    # Set hue based on a palette
    colors = seaborn.color_palette("husl", hsv.shape[0])
    for row_no in range(hsv.shape[0]):
        c = colors[row_no]
        c = rgb2hsv(numpy.stack([c]))[0]
        hsv[row_no, :, 0] = c[0]
        hsv[row_no, :, 1] = c[1]
        hsv[row_no, :, 2] *= value_mod

    colored = hsv2rgb(hsv)
    
    # use same kind of order as librosa.specshow
    flipped = colored[::-1, :, :]

    ax.imshow(flipped)
    ax.set(aspect='auto')
    
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax.tick_params(axis='both', which='both', bottom=False, left=False, top=False, labelbottom=False)
    

def plot_activations(S, acts):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(25, 15), sharex=False)
    
    # spectrogram
    db = librosa.amplitude_to_db(S, ref=numpy.max)
    librosa.display.specshow(db, ax=ax[0], y_axis='mel')

    # original activations
    librosa.display.specshow(acts, x_axis='time', ax=ax[1])

    # colorize
    plot_colorized_activations(acts, ax=ax[2], value_mod=3.0)

    # thresholded
    norm = acts / numpy.quantile(acts, 0.90, axis=0, keepdims=True)
    threshold = numpy.quantile(norm, 0.95)
    plot_colorized_activations((norm > threshold).astype(float), ax=ax[3], value_mod=1.0)
    return fig

def main(audio_file, output_file):
    audio_bpm = 130
    sr = 22050
    audio, sr = librosa.load(audio_file, sr=sr)
    S, comps, acts = decompose_audio(y=audio, sr=sr, bpm=audio_bpm)
    fig = plot_activations(S, acts)
    fig.savefig(output_file)

audio_file = '../sound-files/short.wav'
output_file = 'plot.png'

main(audio_file, output_file)