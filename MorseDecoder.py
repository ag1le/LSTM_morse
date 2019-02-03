
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
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from numpy.random import normal
import numpy as np
from morse import Morse
import yaml
from functools import reduce
import matplotlib.cm as cm


class Config():

    def __init__(self, file_name): 
        with open(file_name) as f:
            self.config = yaml.load(f.read())
    
    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)
    
    def __repr__(self):
        return str(self.config)
    

config = Config('model.yaml')




# Read WAV file containing Morse code and create 256x1 (or 16x16) tiles (256 samples/4 seconds)
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import numpy as np

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, periodogram
from peakdetect import peakdet  # download peakdetect from # https://gist.github.com/endolith/250860

def find_peak(fname):
    # Find the signal frequency and maximum value
    Fs, x = wavfile.read(fname)
    f,s = periodogram(x, Fs,'blackman',8192,'linear', False, scaling='spectrum')
    threshold = max(s)*0.9  # only 0.4 ... 1.0 of max value freq peaks included
    maxtab, mintab = peakdet(abs(s[0:int(len(s)/2-1)]), threshold,f[0:int(len(f)/2-1)] )
    return maxtab[0,0]

# Fs should be 8000 Hz 
# with decimation down to 125 Hz we get 8 msec / sample
# with WPM equals to 20 => Tdit = 1200/WPM = 60 msec   (time of 'dit')
# 4 seconds equals 256 samples ~ 66.67 Tdits 
# word 'PARIS' is 50 Tdits

def demodulate(x, Fs, freq):
    # demodulate audio signal with known CW frequency 
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
    Fs, signal = wavfile.read(fname)
    dur = len(signal)/Fs
    o = demodulate(signal[(Fs*(x)):Fs*(x+y)], Fs, tone)
    #print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur

# Read morse.wav from start_time=0 duration=4 seconds
# save demodulated/decimated signal (1,256) to morse.npy 
# options:
# decimate: Fs/16   Fs/64  Fs/64
# duration: 2        4       16
# imgsize : 32       256    1024

"""
filename = "audio/2c1018174f794091916353937fc9f518.wav"
tone = find_peak(filename)
print("tone:{}".format(tone))

o,dur = process_audio_file(filename,0,4, tone)
np.save("morse.npy", o, allow_pickle=False)
im = o[0::1].reshape(1,256)
#o[10:32] = 0.
#im = o[0::1].reshape(1,32)

plt.figure(figsize=(20,10))
plt.subplot(2, 1, 1)
plt.plot(o[0::1])
#plt.annotate('N',xy=(25, 1))
#plt.annotate('O',xy=(90, 1))
#plt.annotate('W',xy=(150, 1))
#plt.annotate('2',xy=(255, 1))
plt.ylabel('amplitude')
plt.xlabel('time')
plt.subplot(2, 1, 2)
plt.imshow(im,cmap = cm.Greys_r)
plt.xlabel('time')
plt.show()
"""


import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import write
#import sounddevice as sd
import matplotlib.pyplot as plt 


def morse(text, file_name=None, SNR_dB=20, f_code=600, Fs=8000, code_speed=20, length_N=None, play_sound=True):
    '''
    # MORSE converts text to playable morse code in wav format
    #
    # SYNTAX
    # morse(text)
    # morse(text,file_name),
    # morse(text,file_name,SNR_dB),
    # morse(text, file_name,SNR_dB,code_frequency),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate, code_speed_wpm, zero_fill_to_N),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate, code_speed_wpm, zero_fill_to_N, play_sound),
    #
    # Description:
    #
    #   If the wave file name is specified, then the funtion will output a wav
    #   file with that file name.  If only text is specified, then the function
    #   will only play the morse code wav file without saving it to a wav file.
    #   If a snr is specified, zero mean addative white Gaussian
    #   noise is added
    #
    # Examples:
    #
    #   morse('Hello'),
    #   morse('How are you doing my friend?','morsecode.wav'),
    #   morse('How are you doing my friend?','morsecode.wav', 20),
    #   morse('How are you doing my friend?','morsecode.wav', 10, 440,Fs,20),
    #   x = morse('How are you doing my friend?','morsecode.wav', 3, 440,Fs, 20, 2^20,True), #(to play the file, and make the length 2^20)
    #
    #   Copyright 2018 Mauri Niininen, AG1LE
    '''


    #t = 0:1/Fs:1.2/code_speed,  #One dit of time at w wpm is 1.2/w.

    t = np.linspace(0., 1.2/code_speed, num=int(Fs*1.2/code_speed), endpoint=True, retstep=False)
   
    Dit = np.sin(2*np.pi*f_code*t)
    ssp = np.zeros(len(Dit))
    # one Dah of time is 3 times  dit time
    t2 = np.linspace(0., 3*1.2/code_speed, num=3*int(Fs*1.2/code_speed), endpoint=True, retstep=False)
    #Dah = np.concatenate((Dit,Dit,Dit))
    Dah = np.sin(2*np.pi*f_code*t2)
    
    lsp = np.zeros(len(Dah)),    # changed size argument to function of Dah 

    # Defining Characters & Numbers
    Codebook = {
        "A": np.concatenate((Dit,ssp,Dah)),
        "B": np.concatenate((Dah,ssp,Dit,ssp,Dit,ssp,Dit)),
        "C": np.concatenate((Dah,ssp,Dit,ssp,Dah,ssp,Dit)),
        "D": np.concatenate((Dah,ssp,Dit,ssp,Dit)),
        "E": Dit,
        "F": np.concatenate((Dit,ssp,Dit,ssp,Dah,ssp,Dit)),
        "G": np.concatenate((Dah,ssp,Dah,ssp,Dit)),
        "H": np.concatenate((Dit,ssp,Dit,ssp,Dit,ssp,Dit)),
        "I": np.concatenate((Dit,ssp,Dit)),
        "J": np.concatenate((Dit,ssp,Dah,ssp,Dah,ssp,Dah)),
        "K": np.concatenate((Dah,ssp,Dit,ssp,Dah)),
        "L": np.concatenate((Dit,ssp,Dah,ssp,Dit,ssp,Dit)),
        "M": np.concatenate((Dah,ssp,Dah)),
        "N": np.concatenate((Dah,ssp,Dit)),
        "O": np.concatenate((Dah,ssp,Dah,ssp,Dah)),
        "P": np.concatenate((Dit,ssp,Dah,ssp,Dah,ssp,Dit)),
        "Q": np.concatenate((Dah,ssp,Dah,ssp,Dit,ssp,Dah)),
        "R": np.concatenate((Dit,ssp,Dah,ssp,Dit)),
        "S": np.concatenate((Dit,ssp,Dit,ssp,Dit)),
        "T": Dah,
        "U": np.concatenate((Dit,ssp,Dit,ssp,Dah)),
        "V": np.concatenate((Dit,ssp,Dit,ssp,Dit,ssp,Dah)),
        "W": np.concatenate((Dit,ssp,Dah,ssp,Dah)),
        "X": np.concatenate((Dah,ssp,Dit,ssp,Dit,ssp,Dah)),
        "Y": np.concatenate((Dah,ssp,Dit,ssp,Dah,ssp,Dah)),
        "Z": np.concatenate((Dah,ssp,Dah,ssp,Dit,ssp,Dit)),
        ".": np.concatenate((Dit,ssp,Dah,ssp,Dit,ssp,Dah,ssp,Dit,ssp,Dah)),
        ",": np.concatenate((Dah,ssp,Dah,ssp,Dit,ssp,Dit,ssp,Dah,ssp,Dah)),
        "?": np.concatenate((Dit,ssp,Dit,ssp,Dah,ssp,Dah,ssp,Dit,ssp,Dit)),
        "/": np.concatenate((Dah,ssp,Dit,ssp,Dit,ssp,Dah,ssp,Dit)),
        "1": np.concatenate((Dit,ssp,Dah,ssp,Dah,ssp,Dah,ssp,Dah)),
        "2": np.concatenate((Dit,ssp,Dit,ssp,Dah,ssp,Dah,ssp,Dah)),
        "3": np.concatenate((Dit,ssp,Dit,ssp,Dit,ssp,Dah,ssp,Dah)),
        "4": np.concatenate((Dit,ssp,Dit,ssp,Dit,ssp,Dit,ssp,Dah)),
        "5": np.concatenate((Dit,ssp,Dit,ssp,Dit,ssp,Dit,ssp,Dit)),
        "6": np.concatenate((Dah,ssp,Dit,ssp,Dit,ssp,Dit,ssp,Dit)),
        "7": np.concatenate((Dah,ssp,Dah,ssp,Dit,ssp,Dit,ssp,Dit)),
        "8": np.concatenate((Dah,ssp,Dah,ssp,Dah,ssp,Dit,ssp,Dit)),
        "9": np.concatenate((Dah,ssp,Dah,ssp,Dah,ssp,Dah,ssp,Dit)),
        "0": np.concatenate((Dah,ssp,Dah,ssp,Dah,ssp,Dah,ssp,Dah)),
      }
    text = text.upper()

    # start with pause (7 dit lengths)
    morsecode= np.concatenate((ssp,ssp,ssp,ssp,ssp,ssp,ssp))
    for ch in text:
        if ch == ' ':
            morsecode = np.concatenate((morsecode, ssp,ssp,ssp,ssp))
        elif ch == '\n':
            pass
        else:
            val = Codebook[ch]
            morsecode = np.concatenate((morsecode, val, ssp,ssp,ssp))
        
    #morsecode = np.concatenate((morsecode, lsp))

    if length_N:
        append_length = length_N - len(morsecode)
        if (append_length < 0):
            print("Length {} isn't large enough for your message, it must be > {}.\n".format(length_N,len(morsecode)))
            return morsecode
        else:
            morsecode = np.concatenate((morsecode, np.zeros(append_length)))
        
    # end with pause (14 dit lengths)
    morsecode = np.concatenate((morsecode,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp))

    #noise = randn(size(morsecode)), 
    #[noisy,noise] = addnoise(morsecode,noise,snr),
    
    if SNR_dB:
        # https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal
        # Desired SNR in dB

        # Desired linear SNR
        SNR_linear = 10.0**(SNR_dB/10.0)
        #print( "Linear snr = ", SNR_linear)

        # Measure power of signal - assume zero mean 
        power = morsecode.var()
        #print ("Power of signal = ", power)

        # Calculate required noise power for desired SNR
        noise_power = power/SNR_linear
        #print ("Noise power = ", noise_power )
        #print ("Calculated SNR = {:4.2f} dB".format(10*np.log10(power/noise_power )))

        # Generate noise with calculated power (mu=0, sigma=1)
        noise = np.sqrt(noise_power)*np.random.normal(0,1,len(morsecode))

        # Add noise to signal
        morsecode = noise + morsecode

    # Normalize before saving 
    max_n = max(morsecode),
    morsecode = morsecode/max_n
    
    if file_name:
        write(file_name, Fs, morsecode)
    if play_sound:
        #sd.play(morsecode, Fs)
        pass
    return morsecode
    
# 24487 words in alphabetical order 
# https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain 
#

import requests
import random
import uuid
import re


URL = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
fnTrain     = config.value('morse.fnTrain')
fnAudio     = config.value('morse.fnAudio')
code_speed  = config.value('morse.code_speed')
SNR_DB      = config.value('morse.SNR_dB')
count       = config.value('morse.count')
length_N    = config.value('morse.length_N')
word_max_length = config.value('morse.word_max_length')
words_in_sample = config.value('morse.words_in_sample')

def generate_dataset():
    rv = requests.get(URL)
    if rv.status_code == 200:
        with open(fnTrain,'w') as mf:
            words = rv.text.split("\n")
            wordcount = len(words)
            words = [w.upper() for w in words if len(w) <= word_max_length]
            for i in range(count):
                audio_file = fnAudio+uuid.uuid4().hex+".wav"
                sample= random.sample(words, words_in_sample)
                line = ' '.join(sample)
                phrase = re.sub(r'[\'.&]', '', line)
                code_speed = random.sample([30],1)
                SNR_DB = random.sample([10,20,30,40],1)            
                morse(phrase, audio_file, SNR_DB[0], 600, 8000, code_speed[0], length_N, False)
                mf.write(audio_file+' '+phrase+'\n')
                print(audio_file, phrase)
            print("completed {} files".format(count)) 



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

def create_image(filename, imgSize, dataAugmentation=False):
    
    # get image name from audio file name - assumes 'audio/filename.wav' format
    name = filename.split('/')
    imgname = "image/"+name[1]+".png"
    
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

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1]=='/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
    
        f=open(filePath+'morsewords.txt')
        chars = set()
        bad_samples = []

        # read all lines in the file 
        for line in f:
            # ignore comment line
            if not line or line[0]=='#':
                continue
            
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 2
            
            # filenames: audio/*.wav
            fileNameAudio = lineSplit[0]

            # Ground Truth text - open files and append to samples
            #

            gtText = self.truncateLabel(' '.join(lineSplit[1:]), maxTextLen)
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
        self.numTrainSamplesPerEpoch = 25000 
        
        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

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

    # model constants
    batchSize = config.value('model.batchSize')  # was 50 
    imgSize = config.value('model.imgSize')  # was (128,32)
    maxTextLen =  config.value('model.maxTextLen') # was 32
    

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

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
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
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
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words) 
            chars = str().join(self.charList)
            wordChars = open('model/wordCharList.txt').read().splitlines()[0]
            corpus = open('data/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


    def setupTF(self):
        "initialize TF"
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)

        sess=tf.Session() # TF session

        saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
        modelDir = 'model/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

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
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate}
        #print(feedDict)
        (_, lossVal) = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal


    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"
        
        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)
        
        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)
        print('inferBatch: probs:{} texts:{} '.format(probs, texts))
        return (texts, probs)
    

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snapID)
 




def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 20 # stop training after this number of epochs without improvement
    accLoss = []
    while True:
        epoch += 1
        print('Epoch:', epoch)

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
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate {:4.1f}% improved, save model'.format(charErrorRate))
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: {:4.1f}%'.format(charErrorRate*100.0))
        else:
            print('Character error rate {:4.1f}% not improved'.format(charErrorRate))
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since {} epochs. Training stopped.'.format(earlyStopping))
            break
    return accLoss


# In[6]:


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
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        print(recognized)
        
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
    return charErrorRate






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

def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = create_image(fnImg, Model.imgSize)
    plt.imshow(img,cmap = cm.Greys_r)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    print(recognized)
   
def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    parser.add_argument("--generate", help="generate a Morse dataset of random words", action="store_true")
    parser.add_argument("-f", dest="filename", required=False,
                    help="input audio file ", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
   
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        decoderType = DecoderType.BestPath
        loader = MorseDataset("./", Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
                
        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
        
        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            loss = train(model, loader)
            plt.plot(loss)
            plt.show()
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)
    elif args.generate:
        generate_dataset()

    # infer text on test audio file
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
        infer(model, args.filename)


if __name__ == "__main__":
    main()

