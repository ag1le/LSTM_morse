#!/usr/bin/env python
# ====================
#  MORSE DATA GENERATOR
# ====================
class Morse():
    def __init__(self, n_samples=1000, max_seq_len=32, min_seq_len=4, filename=None):
        """ Generate sequence of data with dynamic length.
        This class generates samples for training:
        - Morse char A: sequences (i.e. [1, 0, 1, 1, 1, 0, 0, 0,...])
        - Morse char B: sequences (i.e. [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,...])

        NOTICE:
        We have to pad each sequence to reach 'max_seq_len' for TensorFlow
        consistency (we cannot feed a numpy array with inconsistent
        dimensions). The dynamic calculation will then be perform thanks to
        'seqlen' attribute that records every actual sequence length.
        """
        self.Morsecode = {
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

        # read file containing training or test text
        f = open(filename,'r')
        self.text = f.read()
        f.close()
        self.data = []
        self.labels = []
        self.seqlen = []
        self.chrs = []
        for i in range(n_samples):
            ch = self.text[i]
            #print( ch)
            s,lenght = self.encode_data(ch)
            s += [[0.] for i in range(max_seq_len - lenght)]
            self.data.append(s)
            self.labels.append(self.class_labels(ch))
            self.seqlen.append(lenght)
            self.chrs.append(ch)
            
        self.batch_id = 0    

    def encode_morse(self, cws):
        """Return encoded string of symbols from Morse code book.
           Symbols: 
            dit='.'
            dah='-'
            whitespace = '_'
        """
        s=[]
        for chr in cws:
            try: # try to find CW sequence from Codebook
                s += self.Morsecode[chr]
                if chr != ' ':
                    s += ' '
            except:
                if chr == ' ' or chr =='\n':
                    s += '_'
                    continue
                print ("error: %s not in Codebook" % chr)
        return ''.join(s)

    def len_chr(self, ch):
        s = self.Morsecode[ch]
        return len_dits(s)

    def len_dits(cws):
        """Return the length of CW string in dit units, including spaces. """
        val = 0
        for ch in cws:
            if ch == '.': # dit len + el space 
                val += 2
            if ch == '-': # dah len + el space
                val += 4
            if ch==' ':   #  char space
                val += 2
            if ch=='_':   #  word space
                val += 6
        return val

    def encode_data(self, ch):
        """ Return encoded list of dit/dah values based on Morse Coodebook."""
        data = []
        s = ''
        try:
            s = self.Morsecode[ch]
            if ch != ' ':
                s += ' '
            #print(s)
        except:
            if ch == ' ' or ch =='\n':
                s += '_'
            print ("error: %s not in Codebook" % ch)
        for i in s:
            if i == '.':
                data += [[1.],[0.]]
            if i == '-':
                data += [[1.],[1.],[1.],[0.]]
            if i ==' ':   #  char space
                data += [[0.],[0.]]
            if i =='_':   #  word space
                data += [[0.],[0.],[0.],[0.],[0.],[0.]]
        return data, len(data)

    def class_labels(self, ch):
        """ Return on-hot-encoded class label for a given Morse character.
        """
        label = [0.] * len(self.Morsecode)
        try:
            e = list(self.Morsecode.keys()).index(ch)
            label[e] = 1.
            return label
        except:
            return label

    def decode(self, pos):
        """ Return character in position pos in the Morse Codebook, 
            or * if doesn't exist in the Codebook.
        """
        try:
            return self.Morsecode.keys()[pos]
        except:
            return '*'

    def number_of_classes(self):
        """ Return number of characters in the Morse codebook."""
        return len(self.Morsecode)


    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
