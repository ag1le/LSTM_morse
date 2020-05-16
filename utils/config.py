import yaml
from functools import reduce

class Config():

    def __init__(self, file_name): 
        with open(file_name) as f:
            self.config = yaml.load(f.read())
    
    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)
    
    def __repr__(self):
        return str(self.config)