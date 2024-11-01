import pandas as pd
import sklearn.preprocessing
import torch
import os
import shutil
import requests
import zipfile
from tqdm import tqdm
import numpy as np
from utils import generate_test_data_header

dataset_url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
dataset_file_name = dataset_url.split('/')[-1]

def load():
    # download dataset if not found
    if not os.path.exists('UCI HAR Dataset'):
        # dataset not found, download
        print('Downloading dataset...')
        response = requests.get(dataset_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc=dataset_file_name)
        with open(dataset_file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    
        # extract dataset
        with zipfile.ZipFile(dataset_file_name, 'r') as zip_ref:
            zip_ref.extractall('UCI HAR Dataset tmp')
        # extract dataset
        with zipfile.ZipFile('UCI HAR Dataset tmp/UCI HAR Dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('UCI HAR Dataset tmp2')
        # remove tmp folder
        os.remove(dataset_file_name)
        shutil.rmtree('UCI HAR Dataset tmp')
        shutil.move('UCI HAR Dataset tmp2/UCI HAR Dataset', 'UCI HAR Dataset')
        shutil.rmtree('UCI HAR Dataset tmp2')
    
    x = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\\s+', header=None).to_numpy().astype(float)
    y = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None).to_numpy().ravel().astype(int)
    y = y - 1  # make labels 0-indexed
    
    x_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep='\\s+', header=None).to_numpy().astype(float)
    y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None).to_numpy().ravel().astype(int)
    y_test = y_test - 1  # make labels 0-indexed
    
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    
    # move data to torch tensors
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
        generate_test_data_header('ucihar', args.header, x_test, y_test)
        print(f'Header file generated at {args.header}')
    if args.numpy:
        print(f'Saving numpy data in {args.numpy}')
        x_test = x_test.numpy()
        y_test = y_test.numpy()
        os.makedirs(args.numpy, exist_ok=True)
        np.save(os.path.join(args.numpy, 'test.npy'), x_test)
        np.save(os.path.join(args.numpy, 'testlbl.npy'), y_test)
        print('Test data saved')