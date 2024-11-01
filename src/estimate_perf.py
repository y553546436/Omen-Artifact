import argparse
import numpy as np
import pickle


num_features = {
    'mnist': 28*28,
    'isolet': 617,
    'ucihar': 561,
    'language': 0, # do not estimate performance for language dataset
} # number of features
num_classes = {
    'mnist': 10,
    'isolet': 26,
    'ucihar': 6,
    'language': 5, # do not estimate performance for language dataset
} # number of classes

def estimate_performance(args, data_dir, output_dir, dim=10048):
    with open(f'{data_dir}/testlbl.pkl', 'rb') as f:
        testlbl = np.array(pickle.load(f))
    with open(f'{output_dir}/cand.pkl', 'rb') as f:
        cand = np.array(pickle.load(f).cpu())
    with open(f'{output_dir}/confidence.pkl', 'rb') as f:
        confidence = np.array(pickle.load(f).cpu())
    dataset, strategy, start = args.dataset, args.strategy, int(args.start)
    if strategy != 'onetime':
        freq = int(args.freq)
    num_feature = num_features[dataset]
    num_class = num_classes[dataset]
    naive_flops = (dim * num_feature * 2 + dim * 4) + (num_class * (dim * 2)) # encoding + inference # flops per test
    # print(cand.shape, testlbl.shape)
    naive_acc = sum(cand[:, -1] == testlbl) / len(testlbl)
    num_correct = sum(cand[:, -1] == testlbl)
    print(f'Naive accuracy: {naive_acc} ({num_correct} / {testlbl.size}), naive flops per test: {naive_flops}')

    num_test = len(testlbl)
    if strategy == 'linear':
        test_locs = np.array(list(range(start-1, dim, freq)))
    elif strategy == 'exponential':
        test_locs = [start-1]
        while (test_locs[-1]+1) * freq <= dim:
            test_locs.append((test_locs[-1]+1) * freq - 1)
        test_locs = np.array(test_locs)
    elif strategy == 'onetime':
        test_locs = np.array([start-1])
    test_locs = np.append(test_locs, dim-1) # terminate at dim if not confident
    # confidence = np.append(confidence, np.full((num_test, 1), True), axis=1)
    confidence[:, dim-1] = True
    stop_indices = np.argmax(confidence[:, test_locs], axis=1)
    stop_locs = test_locs[stop_indices]

    omen_acc = sum(cand[np.arange(num_test), stop_locs] == testlbl) / num_test
    omen_flops = (stop_locs * num_feature * 2 + stop_locs * 4) + (num_class * (stop_locs * 2)) # encoding and inference cost total
    omen_flops = omen_flops.sum()
    for i in range(num_test): # calculate test costs
        diff2_table = np.zeros(num_class)
        for loc in test_locs[:stop_indices[i]]:
            omen_flops += (loc - diff2_table[cand[i, loc]]) * (num_class-1) * 3 # paired test for cand class
            diff2_table[cand[i, loc]] = loc
    omen_flops = omen_flops / num_test
    omen_dim = stop_locs.mean() + 1
    print(f'Average dimension used: {omen_dim}')
    print('Warning: flops estimation only works for real-value MAP and Omen')
    print(f'Accuracy: {omen_acc}, flops per test: {omen_flops}')
    print(f'Flops reduction: {naive_flops - omen_flops}')
    print(f'Speedup: {naive_flops / omen_flops}')
    return omen_acc, omen_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate performance of Omen')
    parser.add_argument('--dataset', type=str, help='Dataset name: mnist, isolet, ucihar')
    parser.add_argument('--data', type=str, help='Directory containing model and test data')
    parser.add_argument('--output', type=str, help='Directory to save output')
    parser.add_argument('--strategy', type=str, help='Omen strategy: linear, exponential, onetime')
    parser.add_argument('--start', type=str, help='First dimension Omen conducts a test')
    parser.add_argument('--freq', type=str, help='Interval between two consecutive tests (linear), or ratio between two consecutive tests (exponential)')
    parser.add_argument('--dim', type=int, help='Dimension of the hypervectors', default=10048)
    args = parser.parse_args()
    if args.data is None and args.dataset is not None:
        args.data = f'data-{args.dataset}'
    if args.output is None and args.dataset is not None:
        args.output = f'output-{args.dataset}'
    if args.data is None and args.dataset is None:
        print('Using default data directory: data')
        args.data = 'data'
    if args.output is None and args.dataset is None:
        print('Using default output directory: output')
        args.output = 'output'
    estimate_performance(args, args.data, args.output, args.dim)
