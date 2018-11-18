#! /usr/bin/env python3

#import numpy as np
#import pandas as pd
#import textract

from textgenrnn import textgenrnn


"""
Load PDF text sources, pre-process, then save.
"""

# The Transsexual Empire: The Making of the Shemale by Janice Raymond
#print("loading pdf (The Transsexual Empire)...")
#text = textract.process("./the-transsexual-empire.pdf")
#text = text.decode("utf-8")
#text = text.lower()

#print("saving txt (The Transsexual Empire)...")
#text_file = open("./the_transsexual_empire.txt", "w")
#text_file.write(text)
#text_file.close()
# TODO edit this txt file by hand to remove all instances of "the transsexual
# empire"?


# The Man Who Would Be Queen by J. Michael Bailey
#print("loading pdf (The Man Who Would Be Queen)...")
#text = textract.process("./the-man-who-would-be-queen.pdf")
#text = text.decode("utf-8")
#text = text.lower()
#
#print("saving txt (The Man Who Would Be Queen)...")
#text_file = open("./the_man_who_would_be_queen.txt", "w")
#text_file.write(text)
#text_file.close()
# NOTE I edited this txt file by hand to remove copyright info that was listed
# on each page
# TODO edit this txt file by hand to remove all instances of "the man who would
# be queen"


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
