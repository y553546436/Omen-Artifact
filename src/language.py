import torch
import numpy as np

from utils import convert_sentences_to_array, generate_test_data_header

# loads simple mnist dataset
def load():
    torch.manual_seed(0)
    np.random.seed(0)
    # fetches data
    lang_list = ['dutch', 'english', 'franch', 'german', 'italian']
    x, x_test, y, y_test = [], [], [], []
    for i, lang in enumerate(lang_list):
        with open('../language_dataset/' + lang + '.txt') as f:
            lines = [line.strip().split('\t')[1] for line in f.readlines()]
        np.random.shuffle(lines)
        x += lines[:len(lines) * 6 // 7]
        x_test += lines[len(lines) * 6 // 7:]
        y += [i] * (len(lines) * 6 // 7)
        y_test += [i] * (len(lines) - len(lines) * 6 // 7)
    y = torch.from_numpy(np.array(y))
    y_test = torch.from_numpy(np.array(y_test))

    return x, x_test, y, y_test


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Generate header file or numpy data for mnist dataset')
    parser.add_argument('--header', type=str, help='Output header file')
    parser.add_argument('--numpy', type=str, help='Output numpy file')
    args = parser.parse_args()
    x, x_test, y, y_test = load()
    print(x[:10])
    # get the maximum length of the sentence
    max_len = max([len(sentence) for sentence in x] + [len(sentence) for sentence in x_test])
    print(f'max len: {max_len}')
    x_test = convert_sentences_to_array(x_test)
    x_test = torch.from_numpy(x_test).int()
    if args.header:
        print(f'Generating header file at {args.header}')
        print('Test data shape:', x_test.shape)
        print('Test label shape:', y_test.shape)
        generate_test_data_header('language', args.header, x_test, y_test, x_type='int')
        print(f'Header file generated at {args.header}')
    if args.numpy:
        print(f'Saving numpy data in {args.numpy}')
        x_test = x_test.numpy()
        y_test = y_test.numpy()
        os.makedirs(args.numpy, exist_ok=True)
        np.save(os.path.join(args.numpy, 'test.npy'), x_test)
        np.save(os.path.join(args.numpy, 'testlbl.npy'), y_test)
        print('Test data saved')


