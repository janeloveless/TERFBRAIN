#! /usr/bin/env python3

#import numpy as np
#import pandas as pd
#import textract

from textgenrnn import textgenrnn


"""
Build and train models.
"""

num_epochs=50

print("Building model for The Man Who Would Be Queen...")
textgen = textgenrnn()
print("Training model for The Man Who Would Be Queen...")
textgen.train_from_file("./the_man_who_would_be_queen.txt", num_epochs=num_epochs)
textgen.save("./the_man_who_would_be_queen_weights_epoch" + str(num_epochs) + ".hdf5")

print("Building model for The Transsexual Empire...")
textgen = textgenrnn()
print("Training model for The Transsexual Empire...")
textgen.train_from_file("./the_transsexual_empire.txt", num_epochs=1)
textgen.save("./the_transsexual_empire_weights_epoch" + str(num_epochs) + ".hdf5")

print("Building the TERFBRAIN...")
textgen = textgenrnn()
print("Training model on The Man Who Would Be Queen and The Transsexual Empire...")
textgen.train_from_file("./corpus.txt", num_epochs=num_epochs)
textgen.save("./TERFBRAIN_weights_epoch" + str(num_epochs) + ".hdf5")
