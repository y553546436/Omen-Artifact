import os
import re
import numpy as np
import pandas as pd


mcu_dir = 'mcu-output'
local_dir = 'output'

def load_csv(file, bit_packed=False):
    if not os.path.exists(file):
        print(f'File {file} not found')
        return {
            'normal_acc': 0,
            'omen_acc': 0,
            'diff_acc': 0,
            'absolute_acc': 0,
            'mean_acc': 0,
            'omen_dim': 0,
            'diff_dim': 0,
            'absolute_dim': 0,
            'mean_dim': 0,
            'normal_time': 0,
            'omen_time': 0,
            'diff_time': 0,
            'absolute_time': 0,
            'mean_time': 0,
        }
    dim_scale = 64 if bit_packed else 1
    data = np.loadtxt(file, delimiter=',')
    normal_acc = (data[:, 0] == data[:, -1]).mean() * 100
    omen_acc = (data[:, 3] == data[:, -1]).mean() * 100
    diff_acc = (data[:, 9] == data[:, -1]).mean() * 100
    absolute_acc = (data[:, 6] == data[:, -1]).mean() * 100
    mean_acc = (data[:, 12] == data[:, -1]).mean() * 100
    omen_dim = data[:, 2].mean() * dim_scale
    diff_dim = data[:, 8].mean() * dim_scale
    absolute_dim = data[:, 5].mean() * dim_scale
    mean_dim = data[:, 11].mean() * dim_scale
    normal_time = data[:, 1].mean()
    omen_time = data[:, 4].mean()
    diff_time = data[:, 10].mean()
    absolute_time = data[:, 7].mean()
    mean_time = data[:, 13].mean()
    return {
        'normal_acc': normal_acc,
        'omen_acc': omen_acc,
        'diff_acc': diff_acc,
        'absolute_acc': absolute_acc,
        'mean_acc': mean_acc,
        'omen_dim': omen_dim,
        'diff_dim': diff_dim,
        'absolute_dim': absolute_dim,
        'mean_dim': mean_dim,
        'normal_time': normal_time,
        'omen_time': omen_time,
        'diff_time': diff_time,
        'absolute_time': absolute_time,
        'mean_time': mean_time,
    }


def shortname(trainer, dtype, dataset):
    n = ''
    if trainer == 'LeHDC':
        n = 'LeH'
    elif trainer == 'OnlineHD':
        n = 'OHD'
    elif trainer == 'LDC':
        n = 'LDC'
    n += '-'
    if dtype == 'binary':
        # n += 'BSC'
        n += 'B'
    else:
        # n += 'MAP'
        n += 'M'
    n += '-'
    if dataset == 'language':
        n += 'lang'
    else:
        n += dataset
    return n


def mcutime(time):
    return f'\mcutime{{{time:.3f}}}'


def localtime(time):
    return f'\localtime{{{time/1000:.3f}}}'


def parse_row(trainer, dtype, dataset, local=False):
    start = ('512' if dtype == 'binary' else '128') if trainer in ['LeHDC', 'OnlineHD'] else '16'
    freq = '64' if trainer in ['LeHDC', 'OnlineHD'] else '4'
    alphas = ['001', '005', '010']
    files = [f'{local_dir if local else mcu_dir}/{dataset}_{trainer}_{dtype}_linear_s{start}_f{freq}_a{alpha}.csv' for alpha in alphas]
    bit_packed = (dtype == 'binary' and trainer != 'LDC')
    if trainer == 'LeHDC':
        dim = 5056
    elif trainer == 'OnlineHD':
        dim = 10048
    else:
        dim = 256
    results = [load_csv(file, bit_packed=bit_packed) for file in files]
    # with open(f'{local_dir}/{dataset}_{trainer}_{dtype}_sv_baseline_acc.csv', 'r') as f:
    #     sv_baseline_acc = float(f.read()) * 100
    sv_file = f'{local_dir if local else mcu_dir}/sv_{dataset}_{trainer}_{dtype}_linear_s{start}_f{freq}_a010.csv'
    sv_result = load_csv(sv_file, bit_packed=bit_packed)
    timedec = localtime if local else mcutime
    row = [shortname(trainer, dtype, dataset), 
           f'{results[0]["normal_acc"]:.1f}', f'{dim:.0f}', timedec(results[0]["normal_time"]), 
           f'{results[0]["omen_acc"]:.1f}', f'{results[0]["omen_dim"]:.0f}', timedec(results[0]["omen_time"]), 
           f'{results[1]["omen_acc"]:.1f}', f'{results[1]["omen_dim"]:.0f}', timedec(results[1]["omen_time"]), #f'{sv_baseline_acc:.1f}',
           f'{sv_result["normal_acc"]:.1f}', timedec(sv_result["normal_time"]),
           f'{results[2]["omen_acc"]:.1f}', f'{results[2]["omen_dim"]:.0f}', timedec(results[2]["omen_time"]), 
           f'{results[0]["diff_acc"]:.1f}', f'{results[0]["diff_dim"]:.0f}', timedec(results[0]["diff_time"]), 
           f'{results[0]["absolute_acc"]:.1f}', f'{results[0]["absolute_dim"]:.0f}', timedec(results[0]["absolute_time"]), 
           f'{results[0]["mean_acc"]:.1f}', f'{results[0]["mean_dim"]:.0f}', timedec(results[0]["mean_time"])]
    return row


# Function to extract numeric value from runtime strings
def extract_time(runtime_str):
    match = re.search(r'\{([0-9.]+)\}', runtime_str)
    return float(match.group(1)) if match else None


def parse_table(all_local=False, csv=False):
    trainers = ['OnlineHD', 'LeHDC', 'LDC']
    dtypes = ['binary', 'real']
    datasets = ['language', 'ucihar', 'isolet', 'mnist']
    rows = []
    for dataset in datasets:
        for trainer in trainers:
            for dtype in dtypes:
                if dataset == 'language' and trainer != 'OnlineHD':
                    continue
                if dataset == 'language' and dtype == 'real':
                    continue
                local = all_local or (trainer in ['LeHDC', 'OnlineHD'] and dtype == 'real')
                row = parse_row(trainer, dtype, dataset, local)
                rows.append(row)
    # parse rows to pandas dataframe
    columns =  ['Configuration', 
                'Normal Acc', 'Normal Dim', 'Normal Time', 
                'Omen Acc(a=0.01)', 'Omen Dim(a=0.01)', 'Omen Time(a=0.01)',
                'Omen Acc(a=0.05)', 'Omen Dim(a=0.05)', 'Omen Time(a=0.05)', #'SV Baseline Acc',
                'SV Baseline Acc', 'SV Baseline Time',
                'Omen Acc(a=0.10)', 'Omen Dim(a=0.10)', 'Omen Time(a=0.10)',
                'Diff Acc', 'Diff Dim', 'Diff Time',
                'Absolute Acc', 'Absolute Dim', 'Absolute Time',
                'Mean Acc', 'Mean Dim', 'Mean Time']
    df = pd.DataFrame(rows, columns=columns)
    # parse runtime strings to numeric values
    df['Normal Time'] = df['Normal Time'].apply(extract_time)
    df['Omen Time(a=0.01)'] = df['Omen Time(a=0.01)'].apply(extract_time)
    df['Omen Time(a=0.05)'] = df['Omen Time(a=0.05)'].apply(extract_time)
    df['Omen Time(a=0.10)'] = df['Omen Time(a=0.10)'].apply(extract_time)
    df['Diff Time'] = df['Diff Time'].apply(extract_time)
    df['Absolute Time'] = df['Absolute Time'].apply(extract_time)
    df['Mean Time'] = df['Mean Time'].apply(extract_time)
    df['SV Baseline Time'] = df['SV Baseline Time'].apply(extract_time)
    print(df)
    df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric)
    # calculate stats
    df['Omen Acc Drop(a=0.01)'] = df['Normal Acc'] - df['Omen Acc(a=0.01)']
    df['Omen Acc Drop(a=0.05)'] = df['Normal Acc'] - df['Omen Acc(a=0.05)']
    df['SV Baseline Acc Drop'] = df['Normal Acc'] - df['SV Baseline Acc']
    df['Omen Acc Drop(a=0.10)'] = df['Normal Acc'] - df['Omen Acc(a=0.10)']
    df['Diff Acc Drop'] = df['Normal Acc'] - df['Diff Acc']
    df['Absolute Acc Drop'] = df['Normal Acc'] - df['Absolute Acc']
    df['Mean Acc Drop'] = df['Normal Acc'] - df['Mean Acc']

    df['Omen Dim Reduction(a=0.01)'] = df['Normal Dim'] / df['Omen Dim(a=0.01)']
    df['Omen Dim Reduction(a=0.05)'] = df['Normal Dim'] / df['Omen Dim(a=0.05)']
    df['Omen Dim Reduction(a=0.10)'] = df['Normal Dim'] / df['Omen Dim(a=0.10)']
    df['Diff Dim Reduction'] = df['Normal Dim'] / df['Diff Dim']
    df['Absolute Dim Reduction'] = df['Normal Dim'] / df['Absolute Dim']
    df['Mean Dim Reduction'] = df['Normal Dim'] / df['Mean Dim']

    df['Omen Speedup(a=0.01)'] = df['Normal Time'] / df['Omen Time(a=0.01)']
    df['Omen Speedup(a=0.05)'] = df['Normal Time'] / df['Omen Time(a=0.05)']
    df['Omen Speedup(a=0.10)'] = df['Normal Time'] / df['Omen Time(a=0.10)']
    df['Diff Speedup'] = df['Normal Time'] / df['Diff Time']
    df['Absolute Speedup'] = df['Normal Time'] / df['Absolute Time']
    df['Mean Speedup'] = df['Normal Time'] / df['Mean Time']
    df['SV Baseline Speedup'] = df['Normal Time'] / df['SV Baseline Time']
    print(df)
    # calculate min, max for stats
    stats = df.describe().loc[['min', 'max']]
    print(stats)

    # build another relative table
    rel_columns = ['Configuration',
                    'Normal Acc', 'Normal Dim', 'Normal Time',
                    'Omen Acc Drop(a=0.01)', 'Omen Dim Reduction(a=0.01)', 'Omen Speedup(a=0.01)',
                    'Omen Acc Drop(a=0.05)', 'Omen Dim Reduction(a=0.05)', 'Omen Speedup(a=0.05)', # 'SV Baseline Acc Drop',
                    'SV Baseline Acc Drop', 'SV Baseline Speedup',
                    'Omen Acc Drop(a=0.10)', 'Omen Dim Reduction(a=0.10)', 'Omen Speedup(a=0.10)',
                    'Diff Acc Drop', 'Diff Dim Reduction', 'Diff Speedup',
                    'Absolute Acc Drop', 'Absolute Dim Reduction', 'Absolute Speedup',
                    'Mean Acc Drop', 'Mean Dim Reduction', 'Mean Speedup']
    rel_df = df[rel_columns]
    stats = rel_df.describe().loc[['min', 'max']]
    # save the min max stats to a file
    stats.to_csv('stats.csv')
    # collect all speedup / dim reduction for Omens
    speedup_columns = ['Omen Speedup(a=0.01)', 'Omen Speedup(a=0.05)', 'Omen Speedup(a=0.10)']
    dim_reduction_columns = ['Omen Dim Reduction(a=0.01)', 'Omen Dim Reduction(a=0.05)', 'Omen Dim Reduction(a=0.10)']
    # concatenate all dim reduction vs speedup data points
    speedup_data = pd.concat([rel_df['Configuration'], rel_df[speedup_columns]], axis=1)
    dim_reduction_data = pd.concat([rel_df['Configuration'], rel_df[dim_reduction_columns]], axis=1)
    print(speedup_data)
    print(dim_reduction_data)
    # remove all LDC-BSC configurations
    speedup_data = speedup_data[~speedup_data['Configuration'].str.contains('LDC-BSC')]
    dim_reduction_data = dim_reduction_data[~dim_reduction_data['Configuration'].str.contains('LDC-BSC')]
    print(speedup_data)
    print(dim_reduction_data)
    # put in a same table for fitting a linear regression of speedup vs dim reduction
    all_speedup = pd.concat([speedup_data['Omen Speedup(a=0.01)'], speedup_data['Omen Speedup(a=0.05)'], speedup_data['Omen Speedup(a=0.10)']])
    all_dim_reduction = pd.concat([dim_reduction_data['Omen Dim Reduction(a=0.01)'], dim_reduction_data['Omen Dim Reduction(a=0.05)'], dim_reduction_data['Omen Dim Reduction(a=0.10)']])
    speedup_dim_reduction = pd.concat([all_speedup, all_dim_reduction], axis=1)
    print(speedup_dim_reduction)
    # linear regression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(all_dim_reduction.values.reshape(-1, 1), all_speedup.values.reshape(-1, 1))
    print(lr.coef_, lr.intercept_)
    # print the linear regression equation
    print(f'Speedup = {lr.coef_[0][0]:.3f} * Dim Reduction + {lr.intercept_[0]:.3f}')
    # calculate mean squared error
    from sklearn.metrics import mean_squared_error
    y_pred = lr.predict(all_dim_reduction.values.reshape(-1, 1))
    mse = mean_squared_error(all_speedup.values.reshape(-1, 1), y_pred)
    print(f'Mean Squared Error: {mse:.3f}')
    # label the local time and mcutime
    if not csv:
        for col in rel_columns:
            if 'Time' in col or 'Speedup' in col:
                for i, row in rel_df.iterrows():
                    if row[col] is not None:
                        if not all_local:
                            if ('OHD' in row['Configuration'] or 'LeH' in row['Configuration']) and '-M-' in row['Configuration']:
                                rel_df.at[i, col] = f'\localtime{{{row[col]:.2f}}}'
                            else:
                                rel_df.at[i, col] = f'\mcutime{{{row[col]:.2f}}}'
                        else:
                            rel_df.at[i, col] = f'\localtime{{{row[col]:.2f}}}'
    # format the table for latex
    # rel_table = []
    row_strs = []
    for _, row in rel_df.iterrows():
        row_str = []
        for col in rel_columns:
            if 'Acc' in col:
                row_str.append(f'{row[col]:.1f}')
            else:
                row_str.append(f'{row[col]:.2f}' if isinstance(row[col], float) else str(row[col]))
        # rel_table.append(' & '.join(row_str) + ' \\\\\\hline')
        row_strs.append(row_str)
            
    # print(rel_table)
    return row_strs


def print_table(all_local=False, csv=False):
    row_strs = parse_table(all_local, csv)
    header = ['Benchmark', 'Unoptimized Acc', 'Unoptimized Dim', 'Unoptimized Time', 'Omen Acc Loss(a=0.01)', 'Omen Dim Reduction(a=0.01)', 'Omen Speedup(a=0.01)', 'Omen Acc Loss(a=0.05)', 'Omen Dim Reduction(a=0.05)', 'Omen Speedup(a=0.05)', 'SV Baseline Acc Loss', 'SV Baseline Speedup', 'Omen Acc Loss(a=0.10)', 'Omen Dim Reduction(a=0.10)', 'Omen Speedup(a=0.10)', 'Diff Acc Loss', 'Diff Dim Reduction', 'Diff Speedup', 'Absolute Acc Loss', 'Absolute Dim Reduction', 'Absolute Speedup', 'Mean Acc Loss', 'Mean Dim Reduction', 'Mean Speedup']
    if csv:
        print(','.join(header))
    else:
        print(' & '.join(header) + ' \\\\\\hline')
    for row_str in row_strs:
        if csv:
            print(','.join(row_str))
        else:
            print(' & '.join(row_str) + ' \\\\\\hline')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse results and generate latex or csv table')
    parser.add_argument('--local', action='store_true', help='Use all local time')
    parser.add_argument('--csv', action='store_true', help='Generate csv table')
    args = parser.parse_args()
    if args.csv:
        print_table(args.local, csv=True)
    else:
        print_table(args.local)
