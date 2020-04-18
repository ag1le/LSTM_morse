import yaml
from functools import reduce

print("in Config.py")

class Config():
    """ reads YAML config file. Use config.value('key') to get the corresponding value """

    def __init__(self, file_name): 
        with open(file_name) as f:
            self.config = yaml.load(f.read())
    
    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)
    
    def __repr__(self):
        return str(self.config)

