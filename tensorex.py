import os
import math
import random
import collections
import re

import numpy as np
import tensorflow as tf

#def read_keywords(keyword_set):


data_index = 0


def generate_batch(batch_size, num_skip, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int16)
    labels = np.ndarray(shape=(batch_size,1), dype=np.int16)
    span = 2 * skip_window + 1
    buff = collections.deque(maxlen=span)

    for _ in range(span):
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span -1)
            targets_to_avoid.append(target)
            batch[i * num_skips +j] = buff[skip_window]
            labels[i * num_skips +j,0] = buff[target]
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


def generate_batch_cbow(data, batch_size, num_skips, skip_window):
    '''
        data : list of index of words
        batch_size : # of words in each mini-batch
        num_skips : # of surrounding words on both direction
        skip_window : # of words at both ends of a sentence to skip
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int16)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int16)

    span = 2 * skip_window + 1
    buff = collections.deque(maxlen=span)

    for _ in range(span):
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size):
        mask = [1] * span
        mask[skip_window] = 0
        batch[i,:] = list(compress(buff, mask))
        labels[i,0] = buff[skip_window]
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels









