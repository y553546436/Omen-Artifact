import argparse
import numpy as np
import pickle
import torch
from string import Template
import math

from utils import convert_sentences_to_array

from omen_header import generate_header as generate_omen_header
from LDC import generate_header as generate_ldc_header


def stringify(array: np.ndarray):
    if len(array.shape) == 1:
        return '{' + ', '.join([str(x) for x in array]) + '}'
    else:
        return '{' + ',\n'.join([stringify(x) for x in array]) + '}'


def bit_pack(array): # pack binary code into uint64_t
    res_2d = []
    for i in range(array.shape[0]):
        res_1d = []
        for j in range(0, array.shape[1], 64):
            s = '0b'
            for x in reversed(array[i, j:j+64]):
                s += '1' if x > 0 else '0'
            res_1d.append(s)
        res_2d.append(res_1d)
    return np.array(res_2d)

def convert_map_to_array(char_map):
    ords = sorted([ord(c) for c in char_map.keys()])
    char_map.keys()
    return np.array(ords)


def generate_header(args, data_dir):
    torch.manual_seed(0)
    np.random.seed(0)
    with open(f'{data_dir}/testlbl.pkl', 'rb') as f:
        testlbl = np.array(pickle.load(f))
    with open(f'{data_dir}/model.pkl', 'rb') as f:
        model = np.array(pickle.load(f))
    dataset, strategy, start, alpha = args.dataset, args.strategy, int(args.start), float(args.alpha)
    if strategy != 'onetime':
        freq = int(args.freq)
        assert start % freq == 0, 'start must be multiple of freq'
    num_class = model.shape[0]
    dim = model.shape[-1]
    print(f'Loaded model: {model.shape}, testlbl: {testlbl.shape}')
    # import the load function from the {dataset}.py file
    load = getattr(__import__(dataset), 'load')
    if not callable(load):
        raise ValueError(f'load function not found in {dataset}.py')

    print('Loading...')
    x, x_test, y, y_test = load()
    num_features = x.size(1) if args.dataset != 'language' else max(len(s) for s in x_test+x) + 1
    num_test = len(testlbl)

    if not args.binary:
        with open(f'{data_dir}/encoder_base.pkl', 'rb') as f:
            encoder_base = torch.tensor(pickle.load(f))
            print(f'encoder base loaded: {encoder_base.shape}')
        with open(f'{data_dir}/encoder_basis.pkl', 'rb') as f:
            encoder_basis = torch.tensor(pickle.load(f))
            print(f'encoder basis loaded: {encoder_basis.shape}')
        # with open(f'{data_dir}/encoder_real_basis.pkl', 'rb') as f:
        #     encoder_real_basis = torch.tensor(pickle.load(f))
        # gaussian_data = encoder_real_basis.T.cpu().numpy()
        gaussian_data = encoder_basis.T.cpu().numpy()
        offset_data = encoder_base.cpu().numpy()
        assert gaussian_data.shape == (num_features, dim), f'gaussian_data shape: {gaussian_data.shape}, expected: {(num_features, dim)}'
        assert offset_data.shape == (dim,), f'offset_data shape: {offset_data.shape}, expected: {(dim,)}'

    # generate real threshold data
    threshold_data = np.zeros(num_class - 1)
    for i in range(num_class - 1):
        alpha_ = alpha / (i+1)
        from scipy.special import ndtri
        threshold_data[i] = ndtri(1 - alpha_) ** 2

    if args.binary:
        # calculate diff2 data
        class_hv = torch.tensor(model)
        pair_diff = class_hv.unsqueeze(1) != class_hv.unsqueeze(0)
        diff2 = torch.cumsum(pair_diff, dim=-1).int()
        test_locs = torch.arange(freq-1, dim, freq)
        if test_locs[-1] != dim-1:
            test_locs = torch.cat((test_locs, torch.tensor([dim-1])))
        diff2_data = diff2[:, :, test_locs].cpu().numpy()
        diff2_data = np.transpose(diff2_data, (0, 2, 1))
        assert diff2_data.shape == (num_class, int(math.ceil(dim/freq)), num_class), f'diff2_data shape: {diff2_data.shape}, expected: {(num_class, int(dim/freq), num_class)}'
        # load encoder data
        if args.dataset != 'language':
            from levelencoder import BinaryLevelEncoder
            levelencoder = torch.load(f'{data_dir}/levelencoder.pth', map_location='cpu', weights_only=False)
            levels = levelencoder.num_level
            min_val = levelencoder.min_val
            max_val = levelencoder.max_val
            basis_data = levelencoder.basis.cpu().numpy()
            codebook_data = levelencoder.codebook.cpu().numpy()
            assert basis_data.shape == (num_features, dim), f'basis_data shape: {basis_data.shape}, expected: {(num_features, dim)}'
            assert codebook_data.shape == (levels+2, dim), f'codebook_data shape: {codebook_data.shape}, expected: {(levels+2, dim)}'
            codebook_data = bit_pack(codebook_data)
            basis_data = bit_pack(basis_data)
        else:
            from sentenceencoder import SentenceEncoder
            sentenceencoder: SentenceEncoder = torch.load(f'{data_dir}/sentenceencoder.pth', map_location='cpu', weights_only=False)
            char_codebook_data = sentenceencoder.codebook.cpu().numpy()
            char_map = sentenceencoder.char_map
            assert char_codebook_data.shape == (len(char_map), dim), f'char_codebook_data shape: {char_codebook_data.shape}, expected: {(len(char_map), dim)}'
            char_codebook_data = bit_pack(char_codebook_data)
            char_map_data = convert_map_to_array(char_map)
            num_char = len(char_map)
        # pack binary code into uint64_t
        dim = math.ceil(dim/64)
        assert freq % 64 == 0, 'freq must be multiple of 64'
        freq = freq // 64
        start = start // 64
        model = bit_pack(model)

    # read thresholds data for diff, absolute and mean strategies
    with open(f'{data_dir}/diff_threshold.pkl', 'rb') as f:
        diff_threshold = pickle.load(f)
    with open(f'{data_dir}/absolute_threshold.pkl', 'rb') as f:
        absolute_threshold = pickle.load(f)
    with open(f'{data_dir}/mean_thresholds.pkl', 'rb') as f:
        mean_thresholds = pickle.load(f)
    if not args.binary:
        # generate squared class hvs for diff, absolute and mean strategies
        test_locs = torch.arange(freq-1, dim, freq)
        if test_locs[-1] != dim-1:
            test_locs = torch.cat((test_locs, torch.tensor([dim-1])))
        class_hv = torch.tensor(model)
        class_squared_sum_data = torch.cumsum(class_hv ** 2, dim=-1)[:, test_locs].cpu().numpy()
        class_squared_sum_data = np.transpose(class_squared_sum_data, (1, 0))
        assert class_squared_sum_data.shape == (int(math.ceil(dim/freq)), num_class), f'class_squared_sum_data shape: {class_squared_sum_data.shape}, expected: {(int(math.ceil(dim/freq)), num_class)}'

    # check data shapes
    if args.dataset != 'language':
        assert x_test.shape == (num_test, num_features), f'x_test shape: {x_test.shape}, expected: {(num_test, num_features)}'
    assert model.shape == (num_class, dim), f'model shape: {model.shape}, expected: {(num_class, dim)}'
    assert threshold_data.shape == (num_class - 1,), f'threshold_data shape: {threshold_data.shape}, expected: {(num_class - 1,)}'


    with open(f'{args.template}', 'r') as f:
        header_template = Template(f.read())

    data_mapping = {
        'num_feature': num_features,
        'start': start,
        'freq': freq,
        'num_dim': dim,
        'num_class': num_class,
        'num_test': num_test,
        'data_type': 'REAL' if not args.binary else 'BINARY',
        'test_data': stringify(x_test.cpu().numpy() if args.dataset != 'language' else convert_sentences_to_array(x_test)),
        'class_hv_data': stringify(model.T),
        'threshold_data': stringify(threshold_data),
        'test_label': stringify(testlbl),
        'diff_threshold': diff_threshold,
        'absolute_threshold': absolute_threshold,
        'mean_thresholds': stringify(mean_thresholds),
    }
    if args.binary:
        data_mapping['diff2_data'] = stringify(diff2_data)
        if args.dataset != 'language':
            data_mapping['levels'] = levels
            data_mapping['min_val'] = min_val
            data_mapping['max_val'] = max_val
            data_mapping['basis_data'] = stringify(basis_data)
            data_mapping['codebook_data'] = stringify(codebook_data)
            # empty char_codebook_data, char_map_data, sentence_len_data, num_char
            data_mapping['char_codebook_data'] = '{}'
            data_mapping['char_map_data'] = '{}'
            data_mapping['sentence_len_data'] = '{}'
            data_mapping['num_char'] = 0
        else:
            data_mapping['char_codebook_data'] = stringify(char_codebook_data)
            data_mapping['char_map_data'] = stringify(char_map_data)
            data_mapping['num_char'] = num_char
            # empty levels, min_val, max_val, basis_data, codebook_data
            data_mapping['levels'] = 0
            data_mapping['min_val'] = 0
            data_mapping['max_val'] = 0
            data_mapping['basis_data'] = '{}'
            data_mapping['codebook_data'] = '{}'
        # empty gaussian_data and offset_data
        data_mapping['gaussian_data'] = '{}'
        data_mapping['offset_data'] = '{}'
        # empty class_squared_sum_data
        data_mapping['class_squared_sum_data'] = '{}'
    else:
        data_mapping['gaussian_data'] = stringify(gaussian_data)
        data_mapping['offset_data'] = stringify(offset_data)
        data_mapping['class_squared_sum_data'] = stringify(class_squared_sum_data)
        # empty binary data
        data_mapping['diff2_data'] = '{}'
        data_mapping['levels'] = 0
        data_mapping['min_val'] = 0
        data_mapping['max_val'] = 0
        data_mapping['basis_data'] = '{}'
        data_mapping['codebook_data'] = '{}'
        data_mapping['char_codebook_data'] = '{}'
        data_mapping['char_map_data'] = '{}'
        data_mapping['sentence_len_data'] = '{}'
        data_mapping['num_char'] = 0

    preambles = ''
    if args.dataset == 'language':
        preambles += '#define LANGUAGE\n'
    if args.alpha == '0.01':
        preambles += '#define HEURISTICS\n'

    header_content = preambles + header_template.substitute(data_mapping)
    with open(f'{args.header}', 'w') as f:
        f.write(header_content)
    print('Header file generated at', args.header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate performance of Omen')
    parser.add_argument('--dataset', type=str, help='Dataset name: mnist, isolet, ucihar')
    parser.add_argument('--data', type=str, help='Directory containing model and test data')
    parser.add_argument('-b', "--binary", help="Use binary hypervectors", action='store_true')
    parser.add_argument('--template', type=str, help='Path to header file template', default='../performance/hdc/include/hdc.h.template')
    parser.add_argument('--header', type=str, help='Path to save header file', default='../performance/hdc/include/hdc.h')
    parser.add_argument('--strategy', type=str, help='Omen strategy: linear, exponential, onetime', default='linear')
    parser.add_argument('--alpha', type=str, help='Omen confidence threshold alpha')
    parser.add_argument('--start', type=str, help='First dimension Omen conducts a test')
    parser.add_argument('--freq', type=str, help='Interval between two consecutive tests (linear), or ratio between two consecutive tests (exponential)')
    parser.add_argument('--BLDC', help='Use BLDC', action='store_true')
    args = parser.parse_args()
    if args.dataset == 'language' and not args.binary:
        raise ValueError('Language dataset must use binary hypervectors')
    if args.alpha is None:
        args.alpha = '0.05' # default confidence threshold
    if args.data is None and args.dataset is not None:
        args.data = f'data-{args.dataset}'
    if args.data is None and args.dataset is None:
        print('Using default data directory: data')
        args.data = 'data'
    print('Using template', args.template)
    print('Output header file will be saved at', args.header)
    if args.BLDC:
        define = '#define BLDC\n'
        if args.alpha == '0.01':
            define += '#define HEURISTICS\n'
        omen_header = generate_omen_header(args, args.data)
        ldc_header = generate_ldc_header(args, args.data)
        with open(args.header, 'w') as f:
            f.write(define + omen_header + ldc_header)
    else:
        generate_header(args, args.data)
