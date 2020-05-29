import glob
import scipy.io.wavfile as wav
import scipy.signal
import numpy as np
import csv
from python_speech_features import mfcc

sample_rate, signal = wav.read("Adela_Micha_1d2_F_alejandra_gil_0002.wav")
