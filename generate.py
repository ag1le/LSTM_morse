import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import write
#import sounddevice as sd
import matplotlib.pyplot as plt 

import requests
import random
import uuid
import re
import yaml
from functools import reduce
import sys
import os
from config.Config import Config





def morse(text, file_name=None, SNR_dB=20, f_code=600, Fs=8000, code_speed=20, length_seconds=4, total_seconds=None,play_sound=True):
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
        "=": np.concatenate((Dah,ssp,Dit,ssp,Dit,ssp,Dit,ssp,Dah)),
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

    # dit duration in seconds
    dit = 1.2/code_speed
    # calculate the length of text in dit units
    txt_dits = MorseCode(text).len

    # calculate total text length in seconds
    tot_len = txt_dits * dit
    if (length_seconds - tot_len < 0):
        raise ValueError(f"text length {tot_len:.2f} exceeds audio length {length_seconds:.2f}")

    # calculate how many dits will fit in the 
    pad_dits = int((length_seconds - tot_len)/dit)
    
    # pad from start with random space to fit proper length
    morsecode = []
    pad = random.randint(0,pad_dits)
    for i in range(pad):
        morsecode = np.concatenate((morsecode,ssp))

    # concatenate all characters in text 
    for ch in text:
        if ch == ' ':
            morsecode = np.concatenate((morsecode, ssp,ssp,ssp,ssp))
        elif ch == '\n':
            pass
        else:
            val = Codebook[ch]
            morsecode = np.concatenate((morsecode, val, ssp,ssp,ssp))
        
    #morsecode = np.concatenate((morsecode, lsp))

    if total_seconds:
        append_length = Fs*total_seconds - len(morsecode)
        if (append_length < 0):
            print("Length {} isn't large enough for your message, it must be > {}.\n".format(length_N,len(morsecode)))
            return morsecode
        else:
            morsecode = np.concatenate((morsecode, np.zeros(append_length)))
        
    # end with pause (14 dit lengths)
    #morsecode = np.concatenate((morsecode,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp,ssp))

    

    # add noise to set desired SNR in dB    
    if SNR_dB is not None:
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
    max_value = max(morsecode),
    morsecode = morsecode/max_value
    
    if file_name:
        write(file_name, Fs, morsecode)
    if play_sound:
        sd.play(morsecode, Fs)
        pass
    return morsecode
    
class MorseCode():
    def __init__(self, text):
        self.code = {
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
             ' ': '_'}
        self.len = self.len_str(text)

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
        s = self.code[ch]
        return self.len_dits(s)
    
    def len_str(self, s):
        i = 0 
        for ch in s:
            val = self.len_chr(ch)
            i += val
        return i-3  #remove last char space at end of string


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
  
    with open(corpus_file) as corpus:
        words = corpus.read().split("\n")

        try: 
            os.makedirs(directory)
        except OSError:
            print("Error: cannot create ", directory)

        with open(fnTrain,'w') as mf:
            
            wordcount = len(words)
            print(f"wordcount:{wordcount}")
            words = [w.upper() for w in words if len(w) <= word_max_length]
            for i in range(count): # count of samples to generate 
                sample= random.sample(words, words_in_sample)
                line = ' '.join(sample)
                phrase = re.sub(r'[\'.&]', '', line)
                if len(phrase) <= 1:
                    continue
                speed = random.sample(code_speed,1)
                SNR = random.sample(SNR_DB,1)
                audio_file = "{}SNR{}WPM{}-{}-{}.wav".format(fnAudio, SNR[0], speed[0], phrase[:-1], uuid.uuid4().hex)      
                try:
                    #print(f"{audio_file}")
                    morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, None, False)
                    mf.write(audio_file+' '+phrase+'\n')
                    
                except Exception as err:
                    print(f"ERROR: {audio_file} {err}")
                    continue
            print("completed {} files".format(count)) 

def main(argv):
    if len(argv) < 2:
        print("usage: python generate.py <model-config.yaml>")
        exit(1)
    print(argv)
    configs = Config(argv[1])
    generate_dataset(configs)

if __name__ == "__main__":
    main(sys.argv)
