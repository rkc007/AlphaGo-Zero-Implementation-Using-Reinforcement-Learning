import sys
sys.path.insert(1, './')
from enum import Enum

class Colour(Enum):
    BLACK = 1
    WHITE = 2
    def __str__(self):

        if(self.value == 1):
            return 'black'
        else:
            return 'white'

    # def __str__(self):
    #     return str(self.value)
