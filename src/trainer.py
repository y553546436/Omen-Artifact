from time import time
import torch
import numpy as np
import pickle
import os
from LDC import LDC, bipolar_to_binary
from NN import LinearHDC, BinaryLinearHD
from LearningHD import LearningHD
from levelencoder import BinaryLevelEncoder
from binary_onlinhd import BinaryOnlineHD
from other_termination import get_mean_thresholds, get_diff_threshold, get_absolute_threshold

def main(dataset: str, dir='data', dim=10000, args=None):
    # import the load function from the {dataset}.py file
    load = getattr(__import__(dataset), 'load')
    if not callable(load):
        raise ValueError(f'load function not found in {dataset}.py')

    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f'Using NVIDIA GPU: {torch.cuda.get_device_name()}.')
        else:
            device = 'cpu'
            print('CUDA is not available. Using CPU.')
    elif args.device == 'cpu':
        device = 'cpu'
        print('Using CPU.')
    else:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f'Using NVIDIA GPU: {torch.cuda.get_device_name()}.')
        else:
            device = 'cpu'
            print('Using CPU.')

    torch.manual_seed(0)
    np.random.seed(0)

    print('Loading...')
    x, x_test, y, y_test = load()
    classes = y.unique().size(0)
    if dataset == 'language':
        assert args.binary and args.trainer == 'OnlineHD', 'Language dataset only supports binary hypervectors with OnlineHD'
    else:
        features = x.size(1)

    if args.binary and args.trainer == 'OnlineHD':
        if dataset == 'language':
            from sentenceencoder import SentenceEncoder
            # get a set of unique characters in the dataset
            chars = set()
            for s in x + x_test:
                chars.update(set(s))
            encoder = SentenceEncoder(dim, chars).to(device)
        else:
            encoder = BinaryLevelEncoder(features, dim, args.levels, args.min, args.max).to(device)
        model = BinaryOnlineHD(classes, dim=dim, encoder=encoder).to(device)
    elif args.trainer == 'LDC':
        if args.binary:
            model = LDC(features, classes, feature_hv_dim=args.fd, value_hv_dim=args.vd)
            model = model.to(device)
        else:
            model = LearningHD(classes, features, dim)
            model.to(device)
    elif args.binary and args.trainer == 'LeHDC':
        encoder = BinaryLevelEncoder(features, dim, args.levels, args.min, args.max).to(device)
        model = BinaryLinearHD(dim, classes, encoder=encoder)
    elif args.trainer == 'LeHDC' or args.trainer == 'OnlineHD':
        from onlinehd import OnlineHD
        # OnlineHD encoding and trainer
        model = OnlineHD(classes, features, dim=dim)
        model = model.to(device)
    else:
        raise ValueError(f'Not implemented trainer: {args.trainer}')

    if dataset != 'language':
        # for language dataset, list of strings, not tensor
        x = x.to(device)
        x_test = x_test.to(device)
    y = y.to(device)
    y_test = y_test.to(device)

    print(f'Classes: {classes}')
    if dataset != 'language':
        # for language dataset, number of features (trigrams) is varying
        print(f'Features: {features}')
        # for language dataset, list of strings, not tensor
        print(f'x shape: {x.shape}')
        print(f'x_test shape: {x_test.shape}')
    print(f'y shape: {y.shape}')
    print(f'y_test shape: {y_test.shape}')

    if args.trainer == 'LeHDC' or args.trainer == 'OnlineHD':
        print('Encoding...')
        t = time()
        enc_x = model.encode(x)
        enc_x_test = model.encode(x_test)
        print('Encoding time:', round(time() - t, 2))
        print()
        print('Encoded X shape:', enc_x.size())
        print('Encoded X:', enc_x)

    print('Training...')

    if args.trainer == 'LeHDC':
        if not args.binary:
            # change to use LinearHDC's trainer
            model = LinearHDC(dim, classes, encoder=model.encoder)
        else:
            # change binary hypervectors to bipolar
            enc_x = - 2 * enc_x + 1
            enc_x = enc_x.float()
            enc_x_test = - 2 * enc_x_test + 1
            enc_x_test = enc_x_test.float()
        model.to(device)
        print(f"encoded x dtype: {enc_x.dtype}")
        model.fit(enc_x, y, enc_x_test, y_test, epochs=args.epochs)
    elif args.trainer == 'OnlineHD':
        if args.binary:
            model.fit(enc_x, y, enc_x_test, y_test, epochs=args.epochs, batch_size=args.batch_size)
        else:
            model.fit(enc_x, y, encoded=True)
    elif args.trainer == 'LDC':
        if args.binary:
            model.fit(x, y, x_test, y_test, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        else:
            model.fit(x, y, x_test, y_test, epochs=args.epochs)
    else:
        raise ValueError(f'Not implemented trainer: {args.trainer}')

    print('Training completed.')
    print('Testing...')
    if args.trainer == 'LeHDC':
        print(f'encoded x test dtype: {enc_x_test.dtype}')
        y_pred = model.predict(enc_x_test)
        if args.binary:
            enc_x_test = bipolar_to_binary(enc_x_test)
            if args.export_mean or args.find_diff_threshold or args.find_absolute_threshold:
                enc_x = bipolar_to_binary(enc_x)
    elif args.trainer == 'OnlineHD':
        if args.binary:
            y_pred = model.predict(enc_x_test)
        else:
            y_pred = model.predict(enc_x_test, encoded=True)
    elif args.trainer == 'LDC':
        y_pred = model.predict(x_test)
    else:
        raise ValueError(f'Not implemented trainer: {args.trainer}')
    print(f'y_pred: {y_pred}')
    acc = (y_pred == y_test).float().mean().item()
    num_correct = (y_pred == y_test).sum().item()
    num_total = y_test.size(0)
    print(f'Accuracy: {acc} ({num_correct} / {num_total})')


    if not os.path.exists(dir):
        os.makedirs(dir)

    print('Exporting model...', end='')

    if args.trainer == 'LeHDC':
        if args.binary:
            class_hvs = bipolar_to_binary(model.binary_layer.weight.sign()).cpu().detach().numpy()
            with open(f'{dir}/model.pkl', 'wb') as f:
                pickle.dump(class_hvs, f)
            encoder = model.encoder
            torch.save(encoder, f'{dir}/levelencoder.pth')
        else:
            class_hvs = model.weight.cpu().detach().numpy()
            encoder_basis = model.encoder.basis.cpu().detach().numpy()
            encoder_base = model.encoder.base.cpu().detach().numpy()
    elif args.trainer == 'OnlineHD':
        if args.binary:
            class_hvs = model.quantized_model().cpu().detach().numpy()
            with open(f'{dir}/model.pkl', 'wb') as f:
                pickle.dump(class_hvs, f)
            if dataset != 'language':
                # with open(f'{dir}/encoder_basis.pkl', 'wb') as f:
                #     pickle.dump(model.encoder.basis.cpu().detach().numpy(), f)
                # with open(f'{dir}/encoder_level_codebook.pkl', 'wb') as f:
                #     pickle.dump(model.encoder.codebook.cpu().detach().numpy(), f)
                encoder = model.encoder
                torch.save(encoder, f'{dir}/levelencoder.pth')
            else:
                # with open(f'{dir}/char_map.pkl', 'wb') as f:
                #     pickle.dump(model.encoder.char_map, f)
                # with open(f'{dir}/codebook.pkl', 'wb') as f:
                #     pickle.dump(model.encoder.codebook.cpu().detach().numpy(), f)
                torch.save(model.encoder, f'{dir}/sentenceencoder.pth')
        else:
            class_hvs = model.model.cpu().detach().numpy()
            # normalize class hvs because Omen uses dot product similarity metric, not cosine similarity
            # after normalization, because test hvs have the same expected norm, the dot product behaves similarly as cosine similarity (low accuracy difference)
            class_hvs /= np.linalg.norm(class_hvs, axis=1).reshape(-1, 1)
            encoder_basis = model.encoder.basis.cpu().detach().numpy()
            encoder_base = model.encoder.base.cpu().detach().numpy()
    elif args.trainer == 'LDC':
        if args.binary:
            class_hvs = model.class_layer.weight
            # convert bipolar model to binary
            class_hvs = bipolar_to_binary(class_hvs).cpu().detach().numpy()
            # print(f'class_hvs shape: {class_hvs.shape}')
            # print(f'class_hvs: {class_hvs}')
            model_state = model.state_dict()
            # LDC custom export
            with open(f'{dir}/model.pkl', 'wb') as f:
                pickle.dump(class_hvs, f)
            torch.save(model_state, f'{dir}/ldc_model_state.pth')
            # save model args for future loading from the state dict
            model_args = {
                'features': features,
                'classes': classes,
                'feature_hv_dim': args.fd,
                'value_hv_dim': args.vd
            }
            with open(f'{dir}/ldc_model_args.pkl', 'wb') as f:
                pickle.dump(model_args, f)
        else:
            class_hvs = model.model.weight.cpu().detach().numpy()
            encoder_basis = model.encoder.basis.cpu().detach().numpy()
            encoder_base = model.encoder.base.cpu().detach().numpy()
    else:
        raise ValueError(f'Not implemented trainer: {args.trainer}')

    if not args.binary:
        with open(f'{dir}/model.pkl', 'wb') as f:
            pickle.dump(class_hvs, f)
        with open(f'{dir}/encoder_basis.pkl', 'wb') as f:
            pickle.dump(encoder_basis, f)
        with open(f'{dir}/encoder_base.pkl', 'wb') as f:
            pickle.dump(encoder_base, f)

    print('done.')

    # export encoded data
    print('Exporting encoded data...', end='')

    if not (args.trainer == 'LeHDC' or args.trainer == 'OnlineHD'):
        with torch.no_grad():
            enc_x_test = model.encode(x_test)
            if args.export_mean or args.find_diff_threshold or args.find_absolute_threshold:
                enc_x = model.encode(x)

    if args.trainer == 'LDC' and args.binary:
        # convert encoded bipolar vectors to binary
        enc_x_test = bipolar_to_binary(enc_x_test).long()
        if args.export_mean or args.find_diff_threshold or args.find_absolute_threshold:
            enc_x = bipolar_to_binary(enc_x).long()
        # print(enc_x_test[:5])


    with open(f'{dir}/testvec.pkl', 'wb') as f:
        pickle.dump(enc_x_test.cpu().detach().numpy(), f)
    with open(f'{dir}/testlbl.pkl', 'wb') as f:
        pickle.dump(y_test.cpu().detach().numpy(), f)

    if args.export_mean:
        # mean_thresholds = get_mean_thresholds(enc_x, y, class_hvs, args.binary)
        mean_thresholds = get_mean_thresholds(enc_x, y, torch.from_numpy(class_hvs).to(device), args.binary)
        with open(f'{dir}/mean_thresholds.pkl', 'wb') as f:
            pickle.dump(mean_thresholds, f)
    if args.find_diff_threshold:
        print('Finding diff threshold...')
        diff_threshold = get_diff_threshold(enc_x, y, torch.from_numpy(class_hvs).to(device), args.binary)
        print(f'diff_threshold: {diff_threshold}')
        with open(f'{dir}/diff_threshold.pkl', 'wb') as f:
            pickle.dump(diff_threshold, f)
    if args.find_absolute_threshold:
        print('Finding absolute threshold...')
        absolute_threshold = get_absolute_threshold(enc_x, y, torch.from_numpy(class_hvs).to(device), args.binary)
        print(f'absolute_threshold: {absolute_threshold}')
        with open(f'{dir}/absolute_threshold.pkl', 'wb') as f:
            pickle.dump(absolute_threshold, f)

    print('done.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset to train on", type=str, required=True)
    parser.add_argument('--trainer', help="Trainer to use (OnlineHD/LeHDC/LDC)", type=str, required=True)
    parser.add_argument('--dir', help="Directory to store encoded test data", type=str)
    parser.add_argument('--dim', help="Dimension of the hypervectors", type=int, default=10000)
    parser.add_argument('-b', "--binary", help="Use binary hypervectors", action='store_true')
    # Binary OnlineHD specific arguments
    parser.add_argument('-l', "--levels", help="Number of levels for Level Encoding", type=int)
    parser.add_argument('-m', "--min", help="Minimum value for Level Encoding", type=float)
    parser.add_argument('-M', "--max", help="Maximum value for Level Encoding", type=float)
    # Binary LDC specific arguments
    parser.add_argument('--fd', help="Feature Dimension for LDC", type=int, default=100)
    parser.add_argument('--vd', help="Value Dimension for LDC", type=int, default=4)
    parser.add_argument('--epochs', help="Number of epochs to train", type=int, default=500)
    parser.add_argument('--batch_size', help="Batch size for training", type=int, default=1000)
    parser.add_argument('--lr', help="Learning rate for training", type=float, default=1e-4)
    parser.add_argument('--device', help="Device to use (cpu/cuda)", type=str)
    # export mean thresholds for termination heuristic "mean"
    parser.add_argument('--export_mean', help="Export mean thresholds for termination heuristic 'mean'", action='store_true', default=True)
    parser.add_argument('--find_diff_threshold', help="Find threshold for termination heuristic 'diff'", action='store_true', default=True)
    parser.add_argument('--find_absolute_threshold', help="Find threshold for termination heuristic 'absolute'", action='store_true', default=True)
    args = parser.parse_args()
    if args.dir is None:
        args.dir = f'data-{args.dataset}'
    if args.binary and (args.trainer == 'OnlineHD' or args.trainer == 'LeHDC'):
        # check if min max are provided (required for other datasets than language)
        if args.dataset != 'language' and (args.levels is None or args.min is None or args.max is None):
            raise ValueError('min, max, and levels must be provided for Level Encoding in Binary mode')
    main(args.dataset, args.dir, args.dim, args)
