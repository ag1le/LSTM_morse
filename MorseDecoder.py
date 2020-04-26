
# coding: utf-8

"""
A Morse Decoder implementation using TensorFlow library.
Learn to classify Morse code sequences using a neural network with CNN + LSTM + CTC

Adapted by:  Mauri Niininen (AG1LE) for Morse code learning

From: Handwritten Text Recognition (HTR) system implemented with TensorFlow.
by Harald Scheidl
See: https://github.com/githubharald/SimpleHTR
"""

from __future__ import division
from __future__ import print_function

import sys
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import random
from numpy.random import normal
import numpy as np
#from morse import Morse
import yaml
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime



class Config():

    def __init__(self, file_name): 
        with open(file_name) as f:
            self.config = yaml.load(f.read())
    
    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)
    
    def __repr__(self):
        return str(self.config)
    



# Read WAV file containing Morse code and create 256x1 (or 16x16) tiles (256 samples/4 seconds)
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import numpy as np

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, periodogram
from peakdetect import peakdet  # download peakdetect from # https://gist.github.com/endolith/250860

def find_peak(fname):
    """Find the signal frequency and maximum value"""
    #print("find_peak",fname)
    Fs, x = wavfile.read(fname)
    f,s = periodogram(x, Fs,'blackman',8192,'linear', False, scaling='spectrum')
    threshold = max(s)*0.8  # only 0.4 ... 1.0 of max value freq peaks included
    maxtab, mintab = peakdet(abs(s[0:int(len(s)/2-1)]), threshold,f[0:int(len(f)/2-1)] )
    try:
        val = maxtab[0,0]
    except:
        print("Error: {}".format(maxtab))
        val = 600.
    return val

# Fs should be 8000 Hz 
# with decimation down to 125 Hz we get 8 msec / sample
# with WPM equals to 20 => Tdit = 1200/WPM = 60 msec   (time of 'dit')
# 4 seconds equals 256 samples ~ 66.67 Tdits 
# word 'PARIS' is 50 Tdits

def demodulate(x, Fs, freq):
    """return decimated and demodulated audio signal envelope at a known CW frequency """
    t = np.arange(len(x))/ float(Fs)
    mixed =  x*((1 + np.sin(2*np.pi*freq*t))/2 )

    #calculate envelope and low pass filter this demodulated signal
    #filter bandwidth impacts decoding accuracy significantly 
    #for high SNR signals 40 Hz is better, for low SNR 20Hz is better
    # 25Hz is a compromise - could this be made an adaptive value?
    low_cutoff = 25. # 25 Hz cut-off for lowpass
    wn = low_cutoff/ (Fs/2.)    
    b, a = butter(3, wn)  # 3rd order butterworth filter
    z = filtfilt(b, a, abs(mixed))
    
    decimate = int(Fs/64) # 8000 Hz / 64 = 125 Hz => 8 msec / sample 
    Ts = 1000.*decimate/float(Fs)
    o = z[0::decimate]/max(z)
    return o

def process_audio_file(fname,x,y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal)/Fs
    o = demodulate(signal[int(Fs*(x)):int(Fs*(x+y))], Fs, tone)
    #print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur

def process_audio_file2(fname,x,y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal)/Fs
    if y - x < 4.0:
        end = x + 4.0
        xi = int(Fs*x)
        yi = int(Fs*y)
        ei = int(Fs*end)
        pad = np.zeros(ei-yi)
        print(f"dur:{dur}x:{x},y:{y}, end:{end}, xi:{xi}, yi:{yi}, ei:{ei}")
        signal = np.insert(signal, slice(yi, ei), pad)
        y = end
        
    o = demodulate(signal[int(Fs*(x)):int(Fs*(x+y))], Fs, tone)
    #print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur

# Read morse.wav from start_time=0 duration=4 seconds
# save demodulated/decimated signal (1,256) to morse.npy 
# options:
# decimate: Fs/16   Fs/64  Fs/64
# duration: 2        4       16
# imgsize : 32       256    1024



import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import write
import sounddevice as sd
import matplotlib.pyplot as plt 
    
class Morse():
    """Generates morse audio files from text. Can add noise to desired SNR level. Add random padding """
    code = {
             '!': '-.-.--',
             '$': '...-..-',
             "'": '.----.',
             '(': '-.--.',
             ')': '-.--.-',
             ',': '--..--',
             '-': '-....-',
             '.': '.-.-.-',
             '/': '-..-.',
             '0': '-----',
             '1': '.----',
             '2': '..---',
             '3': '...--',
             '4': '....-',
             '5': '.....',
             '6': '-....',
             '7': '--...',
             '8': '---..',
             '9': '----.',
             ':': '---...',
             ';': '-.-.-.',
             '>': '.-.-.',     #<AR>
             '<': '.-...',     # <AS>
             '{': '....--',    #<HM>
             '&': '..-.-',     #<INT>
             '%': '...-.-',    #<SK>
             '}': '...-.',     #<VE>
             '=': '-...-',     #<BT>
             '?': '..--..',
             '@': '.--.-.',
             'A': '.-',
             'B': '-...',
             'C': '-.-.',
             'D': '-..',
             'E': '.',
             'F': '..-.',
             'G': '--.',
             'H': '....',
             'I': '..',
             'J': '.---',
             'K': '-.-',
             'L': '.-..',
             'M': '--',
             'N': '-.',
             'O': '---',
             'P': '.--.',
             'Q': '--.-',
             'R': '.-.',
             'S': '...',
             'T': '-',
             'U': '..-',
             'V': '...-',
             'W': '.--',
             'X': '-..-',
             'Y': '-.--',
             'Z': '--..',
             '\\': '.-..-.',
             '_': '..--.-',
             '~': '.-.-',
             ' ': '_'
    }
    def __init__(self, text, file_name=None, SNR_dB=20, f_code=600, Fs=8000, code_speed=20, length_seconds=4, total_seconds=8, play_sound=True):
        self.text = text.upper()
        self.file_name = file_name            # file name to store WAV file 
        self.SNR_dB = SNR_dB                  # target SNR in dB 
        self.f_code = f_code                  # CW tone frequency
        self.Fs = Fs                          # Sampling frequency 
        self.code_speed = code_speed          # code speed in WPM
        self.length_seconds = length_seconds  # caps the CW generation  to 
        self.total_seconds = total_seconds    # pads to the total length if possible 
        self.play_sound = play_sound          # If true play generated audio 

        self.len = self.len_str(self.text)
        self.morsecode = []
        self.t = np.linspace(0., 1.2/self.code_speed, num=int(self.Fs*1.2/self.code_speed), endpoint=True, retstep=False)
        self.Dit = np.sin(2*np.pi*self.f_code*self.t)
        self.ssp = np.zeros(len(self.Dit))
        # one Dah of time is 3 times  dit time
        self.t2 = np.linspace(0., 3*1.2/self.code_speed, num=3*int(self.Fs*1.2/self.code_speed), endpoint=True, retstep=False)
        #Dah = np.concatenate((Dit,Dit,Dit))
        self.Dah = np.sin(2*np.pi*self.f_code*self.t2)
        self.lsp = np.zeros(len(self.Dah))

    def len_dits(self, cws):
        """Return the length of CW string in dit units, including spaces. """
        val = 0
        for ch in cws:
            if ch == '.': # dit len  
                val += 1
            if ch == '-': # dah len 
                val += 3
            if ch=='_':   #  word space
                val += 4
            val += 1 # el space
        val += 2     # char space = 3  (el space +2)
        return val
        
    def len_chr(self, ch):
        s = Morse.code[ch]
        #print(s)
        return self.len_dits(s)
    
    def len_str(self, s):
        i = 0 
        for ch in s:
            val = self.len_chr(ch)
            i += val
            #print(ch, val, i)
        return i-3  #remove last char space at end of string

    def generate(self):
        for ch in self.text:
            s = Morse.code[ch]
            for el in s:
                if el == '.':
                    self.morsecode = np.concatenate((self.morsecode, self.Dit))
                elif el == '-':
                    self.morsecode = np.concatenate((self.morsecode, self.Dah))
                elif el == '_':
                    self.morsecode = np.concatenate((self.morsecode, self.ssp,self.ssp,self.ssp))
                self.morsecode = np.concatenate((self.morsecode, self.ssp))
            self.morsecode = np.concatenate((self.morsecode, self.ssp, self.ssp))

    def SNR(self):
        if self.SNR_dB is not None:
            SNR_linear = 10.0**(self.SNR_dB/10.0)
            power = self.morsecode.var()
            noise_power = power/SNR_linear
            noise = np.sqrt(noise_power)*np.random.normal(0,1,len(self.morsecode))
        self.morsecode = noise + self.morsecode

    def pad_start(self):
        dit = 1.2/self.code_speed # dit duration in seconds
        txt_dits = self.len # calculate the length of text in dit units
        tot_len = txt_dits * dit # calculate total text length in seconds
        if (self.length_seconds - tot_len < 0):
            raise ValueError(f"text length {tot_len:.2f} exceeds audio length {self.length_seconds:.2f}")
        # calculate how many dits will fit in with the text
        pad_dits = int((self.length_seconds - tot_len)/dit)
        # pad with random space to fit proper length
        pad = random.randint(0,pad_dits)
        for i in range(pad):
            self.morsecode = np.concatenate((self.morsecode,self.ssp))

    def pad_end(self):
        if self.total_seconds:
            append_length = self.Fs*self.total_seconds - len(self.morsecode)
            if (append_length > 0):
                self.morsecode = np.concatenate((self.morsecode, np.zeros(append_length)))

    def normalize(self):
        self.morsecode = self.morsecode/max(self.morsecode)

    def audio(self):
        """Generate audio file using other functions"""
        self.morsecode = []
        self.pad_start()
        self.generate()
        self.pad_end()
        self.SNR()
        self.normalize()
        if self.play_sound:
            sd.play(self.morsecode, self.Fs)
        if self.file_name:
            write(self.file_name, self.Fs, self.morsecode)
        



# 24487 words in alphabetical order 
# https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain 
#

import requests
import random
import uuid
import re





def generate_dataset(config):
    "generate audio dataset from a corpus of words"
    URL = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
    directory   = config.value('model.directory')
    corpus_file = config.value('model.corpus')
    filePath    = config.value('model.name')
    fnTrain     = config.value('morse.fnTrain')
    fnAudio     = config.value('morse.fnAudio')
    code_speed  = config.value('morse.code_speed')
    SNR_DB      = config.value('morse.SNR_dB')
    count       = config.value('morse.count')
    length_seconds    = config.value('morse.length_seconds')
    word_max_length = config.value('morse.word_max_length')
    words_in_sample = config.value('morse.words_in_sample')
    print("SNR_DB:{}".format(SNR_DB))
    error_counter = 0

    try: 
        os.makedirs(directory)
    except OSError:
        print("Error: cannot create ", directory)
        
    with open(corpus_file) as corpus:
        words = corpus.read().split("\n")

        

        with open(fnTrain,'w') as mf:
            
            wordcount = len(words)
            print(f"wordcount:{wordcount}")
            words = [w.upper() for w in words if len(w) <= word_max_length]
            for i in range(count): # count of samples to generate 
                sample= random.sample(words, words_in_sample)
                line = ' '.join(sample)  # append space on each sample
                phrase = re.sub(r'[\'.&]', '', line) # remove extra characters
                if len(phrase) <= 1:
                    continue
                speed = random.sample(code_speed,1)
                SNR = random.sample(SNR_DB,1)
                audio_file = "{}SNR{}WPM{}-{}-{}.wav".format(fnAudio, SNR[0], speed[0], phrase[:-1], uuid.uuid4().hex)      
                try:
                    m = Morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, 5, False)
                    m.audio()
                    mf.write(audio_file+' '+phrase+'\n')
                    
                except Exception as err:
                    print(f"ERROR: {audio_file} {err}")
                    error_counter += 1
                    continue
            print(f"completed {count} files from {wordcount}, with {error_counter} errors") 



class Sample:
    "sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

from matplotlib.mlab import specgram
nfft = 256 
overlap = nfft - 56  # overlap value for spectrogram

def get_specgram(signal, rate):
    arr2D, freqs, bins = specgram(
        signal,
        window=np.blackman(nfft),
        Fs=rate,
        NFFT=nfft,
        noverlap=overlap,
        pad_to=32 * nfft,
    )
    return arr2D, freqs, bins

def create_image2(filename, imgSize, dataAugmentation=False):

    imgname = filename+".png"
    
    # Load  image in grayscale if exists
    img = cv2.imread(imgname,0)
        
    if img is None:
        rate, data = wavfile.read(filename)
        arr2D, freqs, bins = get_specgram(data, rate)

        # Get the image data array shape (Freq bins, Time Steps)
        shape = arr2D.shape

        # Find the CW spectrum peak - look across all time steps
        f = int(np.argmax(arr2D[:]) / shape[1])

        time_steps = (4.0/(len(data)/rate))*shape[1]
        #print(f"time_steps{time_steps}")

        # Create a 32x128 array centered to spectrum peak
        img = cv2.resize(arr2D[f - 16 : f + 16][0:568], (128, 32))
        
        if False: # change to True if want to plot 
            fig, ax = plt.subplots(1,1)
            extent = (bins[0], bins[-1], freqs[-1], freqs[0])
            im = ax.imshow(
                arr2D,
                aspect="auto",
                extent=extent,
                interpolation="none",
                cmap="Greys",
                norm=None,
            )
            plt.gca().invert_yaxis()
            plt.show()

        # normalize
        (m, s) = cv2.meanStdDev(img)
        m = m[0][0]
        s = s[0][0]
        img = img - m
        img = img / s if s>0 else img
        img = img*256.
        if img.shape == (32, 128):
            cv2.imwrite(imgname, img)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    
    # transpose for TF
    img = cv2.transpose(img)
    return  img


def create_image(filename, imgSize, dataAugmentation=False):
    
    # get image name from audio file name - assumes 'audio/filename.wav' format
    #name = filename.split('/')
    imgname = filename+".png"
    
    # Load  image in grayscale if exists
    img = cv2.imread(imgname,0)
        
    if img is None:
        #print('.') #could not load image:{} processing audio file'.format(imgname))

        # find the Morse code peak tone 
        tone = find_peak(filename)
        # sample = 16 seconds from audio file into output => (1,1024) 
        # sample = 4 seconds from audio file into output => (1,256) 
        sample = 4 
        o,dur = process_audio_file(filename,0,sample, tone)
        # reshape output into image and resize to match the imgSize of the model (128,32)
        #im = o[0::1].reshape(4,256)
        im = o[0::1].reshape(1,256)
        im = im*256.
        img = cv2.resize(im, imgSize, interpolation = cv2.INTER_AREA)
        # save to file 
        retval = cv2.imwrite(imgname,img)
        if not retval:
            print('Error in writing image:{} retval:{}'.format(imgname,retval))

    
    """
    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5) # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    
    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.zeros([ht, wt]) #* 255
    target[0:newSize[1], 0:newSize[0]] = img
    """
        
    # transpose for TF
    img = cv2.transpose(img)


    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    

    
    # transpose to match tensorflow requirements
    return img


class MorseDataset():

    def __init__(self, config):
        "loader for dataset at given location, preprocess images and text according to parameters"
        # filePath, batchSize, imgSize, maxTextLen 
        self.filePath = config.value("model.directory")
        #assert self.filePath[-1]=='/'
        self.batchSize = config.value("model.batchSize")
        self.imgSize = config.value("model.imgSize")
        self.maxTextLen = config.value("model.maxTextLen")
        self.samples = []
        self.dataAugmentation = False
        self.currIdx = 0

        try: 
            os.makedirs(self.filePath)
        except OSError:
            print("Error: cannot create ", self.filePath)
            #if not os.path.isdir(filePath):
            #    raise
        print(f"MorseDataset: loading {config.value('morse.fnTrain')}")
        with open(config.value('morse.fnTrain'),'r') as f:
            chars = set()
            bad_samples = []

            # read all lines in the file 
            for line in f:
                # ignore comment line
                if not line or line[0]=='#':
                    continue
                
                lineSplit = line.strip().split(' ')
                assert len(lineSplit) >= 2, "line is {}".format(line)
                
                # filenames: audio/*.wav
                fileNameAudio = lineSplit[0]

                # Ground Truth text - open files and append to samples
                #

                gtText = self.truncateLabel(' '.join(lineSplit[1:]), self.maxTextLen)
                gtText = gtText + ' '  # append space to end of each word 
                print(gtText)
                chars = chars.union(set(list(gtText)))

                # put sample into list
                #print("sample text length:{} {}".format(len(gtText), gtText))
                self.samples.append(Sample(gtText, fileNameAudio))
                

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training 
        self.numTrainSamplesPerEpoch = config.value('morse.count')
        
        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))
        file_name = config.value("experiment.fnCharList")
        with open(file_name, 'w') as fn:
            fn.write(str().join(self.charList))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text
        

    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = False #was True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    
    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)
        
        
    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [create_image(self.samples[i].filePath, self.imgSize, self.dataAugmentation) for i in batchRange]
        #imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)






class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model: 
    "minimalistic TF model for Morse Decoder"

    

    def __init__(self, config, decoderType=DecoderType.BestPath, mustRestore=False):
        "init model: add CNN, RNN and CTC and initialize TF"

        # model constants
        self.modelDir = config.value('model.name') 
        self.batchSize = config.value('model.batchSize')  # was 50 
        self.imgSize = config.value('model.imgSize')  # was (128,32)
        self.maxTextLen =  config.value('model.maxTextLen') # was 32
        self.earlyStopping = config.value('model.earlyStopping') #was 5

        self.charList = open(config.value("experiment.fnCharList")).read()
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # input image batch
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, self.imgSize[0], self.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # initialize TF

        (self.sess, self.saver) = self.setupTF()

            
    def setupCNN(self):
        "create CNN layers and return output of these layers"

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        #featureVals = [1, 8, 16, 32, 32, 64]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.cnnOut4d = pool


    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
   
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
                                    
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
                                    
        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        

    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]) , tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.compat.v1.placeholder(tf.float32, shape=[self.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            print("Loading WordBeamSearch...")
            word_beam_search_module = tf.load_op_library('cpp/proj/TFWordBeamSearch.so')
            # prepare information about language (dictionary, characters in dataset, characters forming words) 
            chars = str().join(self.charList)
            wordChars = open(self.modelDir+'wordCharList.txt').read().splitlines()[0]
            corpus = open(self.modelDir+'corpus.txt').read()
            
            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


    def setupTF(self):
        "initialize TF"
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)

        sess=tf.compat.v1.Session() # TF session

        #saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        latestSnapshot = tf.train.latest_checkpoint(self.modelDir) # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + self.modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess,saver)


    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0] # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)
        #print("(indices:{}, values:{}, shape:{})".format(indices, values, shape))
        return (indices, values, shape)


    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"
        
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank=len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label==blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor 
            decoded=ctcOutput[0][0] 

            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0] # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [self.maxTextLen] * numBatchElements, self.learningRate : rate}
        #print(feedDict)
        (_, lossVal) = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal


    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"
        
        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {self.inputImgs : batch.imgs, self.seqLen : [self.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)
        
        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [self.maxTextLen] * numBatchElements}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)
        #print('inferBatch: probs:{} texts:{} '.format(probs, texts))
        return (texts, probs)
    

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, self.modelDir+'snapshot', global_step=self.snapID)
 




def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best validation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = model.earlyStopping  # stop training after this number of epochs without improvement
    accLoss = []
    accChrErrRate = []
    accWordAccuracy = []
    start_time = datetime.datetime.now()
    while True:
        epoch += 1
        print('Epoch: {} Duration:{}'.format(epoch, datetime.datetime.now()-start_time))

        # train
        print('Train NN - imgSize',model.imgSize)
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
            accLoss.append(loss)

        # validate
        charErrorRate, wordAccuracy = validate(model, loader)
        accChrErrRate.append(charErrorRate)
        accWordAccuracy.append(wordAccuracy)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate {:4.1f}% improved, save model'.format(charErrorRate*100.))
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: {:4.1f}% word accuracy: {:4.1f}'.format(charErrorRate*100.0, wordAccuracy*100.))
        else:
            noImprovementSince += 1
            print('Character error rate {:4.1f}% not improved in last {} epochs'.format(charErrorRate*100., noImprovementSince))

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since {} epochs. Training stopped.'.format(earlyStopping))
            break
    end_time = datetime.datetime.now()
    print("Total training time was {}".format(end_time-start_time))
    return accLoss, accChrErrRate, accWordAccuracy


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    #loader.trainSet()
    charErrorRate = float('inf')
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    wordAccuracy = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, probability) = model.inferBatch(batch)
        print(recognized, probability)
        
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    
    try:
        charErrorRate = numCharErr / numCharTotal
        wordAccuracy = numWordOK / numWordTotal
        print('Character error rate: {:4.1f}%. Word accuracy: {:4.1f}%.'.format(charErrorRate*100.0, wordAccuracy*100.0))
        print('numCharTotal:{} numWordTotal:{}'.format(numCharTotal,numWordTotal))
    except:
        print('numCharTotal:{} numWordTotal:{}'.format(numCharTotal,numWordTotal))
    return charErrorRate, wordAccuracy






import sys
import argparse
import cv2
import editdistance
import os.path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return an open file handle

class FilePaths:
    "filenames and paths to data"
    fnCharList = 'morseCharList.txt'
    fnAccuracy = 'model/accuracy.txt'
    fnTrain = 'data/'
    fnInfer = 'audio/6db42dd27d414097b2f02c4ca7a800e9.wav'
    fnCorpus = "morseCorpus.txt"
    fnExperiments = "experiments/"
    fnResults = "results/result.txt"

def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = create_image(fnImg, model.imgSize)
    plt.imshow(img,cmap = cm.Greys_r)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    print(recognized)


#from pyAudioAnalysis.audioSegmentation import silence_removal

def infer_image(model, o):
    im = o[0::1].reshape(1,256)
    im = im*256.
    img = cv2.resize(im, model.imgSize, interpolation = cv2.INTER_AREA)
    fname =f'dummy{uuid.uuid4().hex}.png'
    cv2.imwrite(fname,img)
    img = cv2.transpose(img)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    return fname, recognized, probability


   
def infer_file(model, fname):
    print(f"SILENCE REMOVAL:{remove_silence}")
    if remove_silence:
        print()
        tone = find_peak(fname)
        [Fs,x] = wavfile.read(fname)
        segments = silence_removal(x, Fs, 0.25, 0.05, 0.2, 0.2, False)
        for start, stop in segments: 
            print("*"*80,f"start:{start}, stop:{stop} dur:{stop-start}")
            o,dur = process_audio_file2(fname, start, stop, tone)
            start_time = datetime.datetime.now()
            iname, recognized, probability = infer_image(model, o[0:256])
            stop_time = datetime.datetime.now()
            if True: #probability[0] > 0.00005:
                print(f'File:{iname}')
                print('Recognized:', '"' + recognized[0] + '"')
                print('Probability:', probability[0])
                print('Duration:{}'.format(stop_time-start_time))
        return 
    else:        
        sample = 4.0 
        start = 0.
        tone = find_peak(fname)
        o,dur = process_audio_file(fname,start,sample, tone)
        while start < (dur - sample):
            print(start,dur)
            im = o[0::1].reshape(1,256)
            im = im*256.
            img = cv2.resize(im, model.imgSize, interpolation = cv2.INTER_AREA)
            cv2.imwrite(f'dummy{start}.png',img)

            img = cv2.transpose(img)

            batch = Batch(None, [img])
            start_time = datetime.datetime.now()
            (recognized, probability) = model.inferBatch(batch, True)
            stop_time = datetime.datetime.now()
            if probability[0] > 0.0000:
                print('Recognized:', '"' + recognized[0] + '"')
                print('Probability:', probability[0])
                print('Duration:{}'.format(stop_time-start_time))
            start += 1./1
            o,dur = process_audio_file(fname,start,sample, tone)

 

def main():
    "main function"

    global remove_silence
    remove_silence = False
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    #parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    #parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    parser.add_argument("--generate", help="generate a Morse dataset of random words", action="store_true")
    parser.add_argument("--experiments", help="generate a set of experiments using config files", action="store_true")
    parser.add_argument("--silence", help="remove silence", action="store_true")
    parser.add_argument("-f", dest="filename", required=False,
                    help="input audio file ", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    config = Config('model_arrl2.yaml') #read configs for current training/validation/inference job

    decoderType = DecoderType.WordBeamSearch
    #decoderType = DecoderType.BeamSearch
    #decoderType = DecoderType.BestPath
    
    #if args.beamsearch:
    #    decoderType = DecoderType.BeamSearch
    #elif args.wordbeamsearch:
    #    decoderType = DecoderType.WordBeamSearch
    if args.experiments:
        print("*"*80)
        fnExperiments = config.value('experiment.fnExperiments')
        print(f"Looking for model files in {fnExperiments}")
        experiments = [f for f in listdir(fnExperiments) if isfile(join(fnExperiments, f))]
        print(experiments)
        for filename in experiments:
            tf.reset_default_graph()
            exp_config = Config(fnExperiments+filename)
            generate_dataset(exp_config)
            decoderType = DecoderType.WordBeamSearch
            loader = MorseDataset(exp_config)
            model = Model(exp_config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            with open(FilePaths.fnResults, 'a+') as fr:
                fr.write("\nexperiment:{} loss:{} charErrorRate:{} wordAccuracy:{}".format(filename, min(loss), min(charErrorRate), max(wordAccuracy)))
            tf.reset_default_graph()
        return
    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        decoderType = DecoderType.WordBeamSearch
        decoderType = DecoderType.BeamSearch
        loader = MorseDataset(config)
        # save characters of model for inference mode
        open(config.value("experiment.fnCharList"), 'w').write(str().join(loader.charList))
                
        # save words contained in dataset into file
        open(config.value("experiment.fnCorpus"), 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
        
        # execute training or validation
        if args.train:
            model = Model(config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            plt.figure(figsize=(20,10))
            plt.subplot(3, 1, 1)
            plt.title("Character Error Rate")
            plt.plot(charErrorRate)
            plt.subplot(3, 1, 2)
            plt.title("Word Accuracy")
            plt.plot(wordAccuracy)
            plt.subplot(3, 1, 3)
            plt.title("Loss")
            plt.plot(loss)
            plt.show()
        elif args.validate:
            model = Model(config, decoderType, mustRestore=True)
            validate(model, loader)
    elif args.generate:
        generate_dataset(config)

    # infer text on test audio file
    else:
        if args.silence:
            print(f"SILENCE REMOVAL ON")
            remove_silence = True
        config = Config('model.yaml')
        print("*"*80)
        print(open(config.value("experiment.fnAccuracy")).read())
        start_time = datetime.datetime.now()
        model = Model(config, decoderType, mustRestore=True)
        print("Loading model took:{}".format(datetime.datetime.now()-start_time))
        infer_file(model, args.filename)


if __name__ == "__main__":
    main()

