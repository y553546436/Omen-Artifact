from src.partial import load_and_test
from src.estimate_perf import estimate_performance


mcu_dir = 'mcu-output'
local_dir = 'output'


def convert_to_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    for key, value in args.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args([])


def get_acc_dim(partial_args, ep_args):
    partial_args = convert_to_args(partial_args)
    load_and_test(partial_args.data, partial_args.output, partial_args.alpha, partial_args.binary, partial_args)
    ep_args = convert_to_args(ep_args)
    acc, dim_used = estimate_performance(ep_args, ep_args.data, ep_args.output, ep_args.dim)
    print(f'{partial_args.dataset}, {partial_args.ber}, {partial_args.strategy}, {partial_args.alpha}, {acc}, {dim_used}')
    return acc, dim_used


def main():
    trainer = 'LDC'
    dtype = 'binary'
    dim = 256
    datasets = ['ucihar', 'isolet', 'mnist']
    bers = [0.0215, 0.1273]
    alphas = [0.01, 0.05, 0.10]
    for dataset in datasets:
        partial_args_template = {
            'dataset': dataset,
            'data': f'src/model_{dataset}_{trainer}_{dtype}',
            'output': f'src/output_{dataset}_{trainer}_{dtype}',
            'alpha': 0.05,
            'binary': dtype == 'binary',
            'strategy': 'omen',
            'numtest': -1,
            'threshold': None,
            'cutoff': None,
            'ber': 0,
        }
        ep_args_template = {
            'dataset': dataset,
            'data': f'src/model_{dataset}_{trainer}_{dtype}',
            'output': f'src/output_{dataset}_{trainer}_{dtype}',
            'strategy': 'linear',
            'start': 16,
            'freq': 4,
            'dim': dim
        }
        for ber in bers:
            partial_args_template['ber'] = ber
            output = []
            output_file = f'output/{dataset}_{trainer}_{dtype}_ber{ber}.csv'
            from copy import deepcopy
            # unoptimized baseline
            partial_args = deepcopy(partial_args_template)
            partial_args['strategy'] = 'cutoff'
            partial_args['cutoff'] = dim
            ep_args = deepcopy(ep_args_template)
            ep_args['strategy'] = 'onetime'
            ep_args['start'] = dim
            acc, dim_used = get_acc_dim(partial_args, ep_args)
            output.append(str(acc))
            output.append(str(dim_used))
            # omen
            for alpha in alphas:
                partial_args = deepcopy(partial_args_template)
                partial_args['alpha'] = alpha
                acc, dim_used = get_acc_dim(partial_args, ep_args_template)
                output.append(str(acc))
                output.append(str(dim_used))
            # diff, absolute, mean
            for strategy in ['diff', 'absolute', 'mean']:
                partial_args = deepcopy(partial_args_template)
                partial_args['strategy'] = strategy
                acc, dim_used = get_acc_dim(partial_args, ep_args_template)
                output.append(str(acc))
                output.append(str(dim_used))
            with open(output_file, 'w') as f:
                f.write(','.join(output) + '\n')

if __name__ == '__main__':
    main()