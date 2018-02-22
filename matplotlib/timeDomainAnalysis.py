import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import scipy.io.wavfile as wf
import praatUtil
import os
from matplotlib import pyplot as plt
import matplotlibUtil
import generalUtility
import sys


def dbfft(x, fs, win=None, ref=32768):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(1, N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref)

    return freq, s_dbfs
    
def process1():
	spf = wave.open('szenario_1a_codriver_PostfilterOn_WnSuppOn.wav','r')
	spf2 = wave.open('szenario_1a_driver_PostfilterOn_WnSuppOn.wav', 'r')

	#Extract Raw Audio from Wav File
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')

	signal2 = spf2.readframes(-1)
	signal2 = np.fromstring(signal2, 'Int16')


	#If Stereo
	if spf.getnchannels() == 2:
	    print 'Just mono files'
	    sys.exit(0)

	plt.figure(1)

	plt.subplot(211)
	plt.plot(signal2)


	plt.subplot(211)
	plt.plot(signal)

	plt.xlabel('Samples [s]')
	plt.ylabel('Amplitude [dB]')
	plt.show()
	return

def process2():
	# Load the file
	fs, signal = wf.read('szenario_1a_codriver_PostfilterOn_WnSuppOn.wav')

	# Take slice
	N = 8192
	win = np.hamming(N)
	freq, s_dbfs = dbfft(signal[0:N], fs, win)

	# Scale from dBFS to dB
	K = 120
	s_db = s_dbfs + K

	plt.plot(freq, s_db)
	plt.grid(True)
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.show()
	return
	
def process3():
	fName = 'szenario_1a_codriver_PostfilterOn_WnSuppOn.wav'
	# for this to work our sound file needs to be in the same directory as this 
	# Python script, and we need to get the path of that script:
	path = sys.path[0] + '/'
	
	#print("path ", path)
	#print("fName ", fName)
	fileNameOnly = generalUtility.getFileNameOnly(fName)
	# calculate the Intensity data using Praat
	dataT, dataI = praatUtil.calculateIntensity(path + fName)
	# normalize the dB data, since it's not calibrated
	dataI -= dataI.max()
	# generate the graph
	graph = matplotlibUtil.CGraph(width = 32, height = 18)
	graph.createFigure()
	ax = graph.getArrAx()[0]
	ax.plot(dataT, dataI, linewidth = 2)
	ax.set_xlabel("Time [s]")
	ax.set_ylabel("SPL [dB]")
	ax.set_title(fileNameOnly)
	graph.padding = 0.1
	graph.adjustPadding(bottom = 2, right = 0.5)
	ax.grid()
	# It is not aesthetically pleasing when graph data goes to all the way to the 
	# upper and lower edges of the graph. I prefer to have a little space.
	matplotlibUtil.setLimit(ax, dataI, 'y', rangeMultiplier = 0.1)
	# every doubling of sound pressure level (SPL) results in an increase of SPL by
	# 6 dB. Therefore, we need to change the y-axis ticks
	matplotlibUtil.formatAxisTicks(ax, 'y', 6, '%d')
	
	# finally, save the graph to a file
	plt.savefig(fileNameOnly + "_intensity.png")
	#plt.plot(dataT, dataI)
	plt.show(graph)

	return

process3()
#process1()	    
	    
	

