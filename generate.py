import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import write
#import sounddevice as sd
import matplotlib.pyplot as plt 

import yaml
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import sys
import os


class Config():

    def __init__(self, file_name): 
        with open(file_name) as f:
            self.config = yaml.load(f.read())
    
    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)
    
    def __repr__(self):
        return str(self.config)


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
             ' ': '_',
             '\n':'_'
                     
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
        self.parse_index = 0
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
        if len(s) == 0:
            return 0
        i = 0 
        for ch in s:
            val = self.len_chr(ch)
            i += val
            #print(ch, val, i)
        return i-3  #remove last char space at end of string

    def len_in_secs(self, s):
        dit = 1.2/self.code_speed
        len_in_dits = self.len_str(s)
        return dit*len_in_dits 


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
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"in __exit__:{exc_type} {exc_value} {traceback}")
    

    def generate_fragments(self):
        """ Yield string fragments shorter than self.length_seconds until end of self.text"""   
        mybuf = ''
        for nextchar in self.text:
            mybuf += nextchar
            len_in_secs = self.len_in_secs(mybuf)
            if len_in_secs < self.length_seconds:
                continue
            elif len_in_secs >= self.length_seconds:
                yield mybuf[:-1], self.len_in_secs(mybuf[:-1])
                mybuf = nextchar
            elif len_in_secs < 0.:
                raise ValueException("ERROR: parse_string should never have negative length strings")
   
        yield mybuf[:], self.len_in_secs(mybuf[:])

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
        

    wordcount = 0
    with open('arrl2.txt') as corpus:
        #words = corpus.read().split("\n")
        text = corpus.read()
        for speed in code_speed: # generate training material in all WPM speeds in the list
            wordcount = 0
            with Morse(text,code_speed=speed) as m1, open(fnTrain,'w') as mf:
                for line, duration in m1.generate_fragments():
                    phrase = re.sub(r'[\'&/]', '', line) # remove extra characters
                    if len(phrase) <= 1:
                        continue
                    print(f"speed:{speed} of {len(code_speed)} phrase:{phrase} dur:{duration}")
                    SNR = random.sample(SNR_DB,1)
                    audio_file = "{}SNR{}WPM{}-{}.wav".format(fnAudio, SNR[0], speed, uuid.uuid4().hex)      
                    try:
                        m = Morse(phrase, audio_file, SNR[0], 600, 8000, speed, length_seconds, 5, False)
                        m.audio()
                        mf.write(audio_file+'|'+phrase+'|\n')
                        wordcount += 1
                    except Exception as err:
                        print(f"ERROR: {audio_file} {err}")
                        error_counter += 1
                        continue
                print(f"completed {wordcount} files for speed:{speed}, with {error_counter} errors") 



def main(argv):
    if len(argv) < 2:
        print("usage: python generate.py <model-config.yaml>")
        exit(1)
    print(argv)
    configs = Config(argv[1])
    generate_dataset(configs)

if __name__ == "__main__":
    main(sys.argv)
