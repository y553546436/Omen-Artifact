from parse_results import load_csv
from src.partial import load_and_test
from src.estimate_perf import estimate_performance


mcu_dir = 'mcu-output'
local_dir = 'output'


def get_omen_dim(dataset, trainer, dtype, local):
    start = ('512' if dtype == 'binary' else '128') if trainer in ['LeHDC', 'OnlineHD'] else '16'
    freq = '64' if trainer in ['LeHDC', 'OnlineHD'] else '4'
    omen_file = f'{local_dir if local else mcu_dir}/{dataset}_{trainer}_{dtype}_linear_s{start}_f{freq}_a005.csv'
    bit_packed = (dtype == 'binary' and trainer != 'LDC')
    omen_data = load_csv(omen_file, bit_packed=bit_packed)
    return int(omen_data['omen_dim'])


def convert_to_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    for key, value in args.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args([])


def main():
    trainers = ['OnlineHD', 'LeHDC', 'LDC']
    dtypes = ['binary', 'real']
    datasets = ['language', 'ucihar', 'isolet', 'mnist']
    for dataset in datasets:
        for trainer in trainers:
            if trainer == 'LeHDC':
                dim = 5056
            elif trainer == 'OnlineHD':
                dim = 10048
            else:
                dim = 256
            for dtype in dtypes:
                if dataset == 'language' and trainer != 'OnlineHD':
                    continue
                if dataset == 'language' and dtype == 'real':
                    continue
                local = (trainer in ['LeHDC', 'OnlineHD'] and dtype == 'real')
                omen_dim = get_omen_dim(dataset, trainer, dtype, local)
                print(f'{dataset}, {trainer}, {dtype}, dim: {omen_dim}')
                partial_args = {
                    'dataset': dataset,
                    'data': f'src/model_{dataset}_{trainer}_{dtype}',
                    'output': f'src/output_{dataset}_{trainer}_{dtype}',
                    'alpha': 0.05,
                    'binary': dtype == 'binary',
                    'strategy': 'cutoff',
                    'numtest': -1,
                    'threshold': None,
                    'cutoff': omen_dim,
                    'ber': 0,
                }
                partial_args = convert_to_args(partial_args)
                load_and_test(partial_args.data, partial_args.output, partial_args.alpha, partial_args.binary, partial_args)
                ep_args = {
                    'dataset': dataset,
                    'data': f'src/model_{dataset}_{trainer}_{dtype}',
                    'output': f'src/output_{dataset}_{trainer}_{dtype}',
                    'strategy': 'onetime',
                    'start': omen_dim,
                    'freq': None,
                    'dim': dim
                }
                ep_args = convert_to_args(ep_args)
                acc, dim = estimate_performance(ep_args, ep_args.data, ep_args.output, ep_args.dim)
                print(f'{dataset}, {trainer}, {dtype}, acc: {acc}')
                output_file = f'output/{dataset}_{trainer}_{dtype}_sv_baseline_acc.csv'
                with open(output_file, 'w') as f:
                    f.write(f'{acc}\n')

if __name__ == '__main__':
    main()