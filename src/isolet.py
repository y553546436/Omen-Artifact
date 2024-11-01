import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
from utils import generate_test_data_header

def load():
    # fetches data
    x, y = sklearn.datasets.fetch_openml('isolet', version=1, return_X_y=True, parser='auto')
    x = np.array(x)
    y = np.array(y)
    x = x.astype(float)
    y = y.astype(int) - 1

    # split and normalize
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y, random_state=0, stratify=y, test_size=1/5)
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    return x, x_test, y, y_test


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Generate header file or numpy data for mnist dataset')
    parser.add_argument('--header', type=str, help='Output header file')
    parser.add_argument('--numpy', type=str, help='Output numpy file')
    args = parser.parse_args()
    _, x_test, _, y_test = load()
    if args.header:
        print(f'Generating header file at {args.header}')
        print('Test data shape:', x_test.shape)
        print('Test label shape:', y_test.shape)
        generate_test_data_header('isolet', args.header, x_test, y_test)
        print(f'Header file generated at {args.header}')
    if args.numpy:
        print(f'Saving numpy data in {args.numpy}')
        x_test = x_test.numpy()
        y_test = y_test.numpy()
        os.makedirs(args.numpy, exist_ok=True)
        np.save(os.path.join(args.numpy, 'test.npy'), x_test)
        np.save(os.path.join(args.numpy, 'testlbl.npy'), y_test)
        print('Test data saved')