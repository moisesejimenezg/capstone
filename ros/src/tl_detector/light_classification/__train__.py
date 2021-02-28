import os
import pickle

from tl_classifier import TLClassifier

file_name = './../../../data/dataset.p'
assert os.path.isfile(file_name), 'data file not found'

with open(file_name, mode='rb') as f:
    data = pickle.load(f)

shape = [256, 124, 3]
classifier = TLClassifier(shape)
classifier.train(data=data)
