#! /usr/bin/env python3

#import numpy as np
#import pandas as pd
#import textract

from textgenrnn import textgenrnn


"""
Build and train models.
"""

num_epochs = 50

print("Building model for The Man Who Would Be Queen...")
textgen1 = textgenrnn()
print("Training model for The Man Who Would Be Queen...")
textgen1.train_from_file("./preproc_texts/the_man_who_would_be_queen.txt", num_epochs=num_epochs)
textgen1.save("./weights/the_man_who_would_be_queen_weights_epoch" + str(num_epochs) + ".hdf5")

print("Building model for The Transsexual Empire...")
textgen2 = textgenrnn()
print("Training model for The Transsexual Empire...")
textgen2.train_from_file("./preproc_texts/the_transsexual_empire.txt", num_epochs=num_epochs)
textgen2.save("./weights/the_transsexual_empire_weights_epoch" + str(num_epochs) + ".hdf5")

print("Building the TERFBRAIN...")
textgen3 = textgenrnn()
print("Training model on The Man Who Would Be Queen and The Transsexual Empire...")
textgen3.train_from_file("./preproc_texts/corpus.txt", num_epochs=num_epochs)
textgen3.save("./weights/TERFBRAIN_weights_epoch" + str(num_epochs) + ".hdf5")
