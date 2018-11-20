#! /usr/bin/env python3

#import numpy as np
#import pandas as pd

from textgenrnn import textgenrnn


"""
Build models.
"""

print("Building model for The Man Who Would Be Queen...")
#textgen1 = textgenrnn("./weights/the_man_who_would_be_queen_weights_epoch50.hdf5")
#textgen2 = textgenrnn("./weights/the_transsexual_empire_weights_epoch50.hdf5")
textgen = textgenrnn("./weights/TERFBRAIN_weights_epoch50.hdf5")
