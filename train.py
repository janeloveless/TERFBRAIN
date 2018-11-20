#! /usr/bin/env python3

#import numpy as np
#import pandas as pd
#import textract

from textgenrnn import textgenrnn


"""
Build and train models.
"""

num_epochs = 100

print("Building the TERFBRAIN...")
textgen = textgenrnn()
print("Training model on corpus...")
textgen.train_from_file("./preproc_texts/corpus.txt", num_epochs=num_epochs)
textgen.save("./weights/TERFBRAIN_weights_epoch" + str(num_epochs) + ".hdf5")
