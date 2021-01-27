import parse_excel
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np 



def median_filter(x):
    return signal.medfilt(x,9)


def smooth_filter(x):

    WC = 0.5 # Tuning
    num = [1]
    den = [1/WC, 1]

    (b,a) = signal.bilinear(num, den, fs=1)
    return signal.filtfilt(b,a,x) 



