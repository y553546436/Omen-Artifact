import numpy as np
import torch


def tensor_to_c_array(tensor: torch.Tensor):
    if len(tensor.shape) == 1:
        return '{' + ', '.join([str(x.item()) for x in tensor]) + '}'
    else:
        return '{' + ',\n'.join([tensor_to_c_array(x) for x in tensor]) + '}'


def generate_test_data_header(dataset: str, path: str, x: torch.Tensor, y: torch.Tensor, x_type='float', y_type='int'):
    header = f""" // This file is generated for the {dataset} dataset
#ifndef TESTDATA_H
#define TESTDATA_H

#define DATA_NUMTEST {x.size(0)}
#define DATA_NUMFEATURE {x.size(1)}
#define DATA_NUMCLASS {y.unique().size(0)}

const {x_type} x[DATA_NUMTEST][DATA_NUMFEATURE] = {tensor_to_c_array(x)};

const {y_type} y[DATA_NUMTEST] = {tensor_to_c_array(y)};

#endif
"""
    with open(path, 'w') as f:
        f.write(header)


def convert_sentences_to_array(x_test):
    sentence_lengths = np.array([len(s) for s in x_test])
    maxlen = sentence_lengths.max()
    res = np.zeros((len(x_test), maxlen + 1), dtype=np.uint32)
    for i, s in enumerate(x_test):
        res[i, 0] = sentence_lengths[i]
        for j, c in enumerate(s):
            res[i, j + 1] = ord(c)
    return res