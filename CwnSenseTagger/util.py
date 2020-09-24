import os
import logging
import numpy as np


def positive_weight(train_generator):
    all_instance = []
    for i in train_generator:
        all_instance += i['label'].tolist()
    pos = sum(all_instance)
    neg = len(all_instance)-pos
    return neg / pos

def accuracy(label, predict):
    correct = ~(np.array(label)^np.array(predict))+2
    return correct.sum()/len(correct)
