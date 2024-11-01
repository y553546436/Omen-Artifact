def acc_format(acc):
    return f'{acc*100:.1f}'


def drr_format(drr):
    return f'{drr:.2f}'


def main():
    datasets = ['ucihar', 'isolet', 'mnist']
    bers = [0.0215, 0.1273]
    ber_names = ['2BPC', '3BPC']
    trainer = 'LDC'
    dtype = 'binary'
    for dataset in datasets:
        for i, ber in enumerate(bers):
            print(f'{dataset}-{ber_names[i]} ', end='')
            with open(f'output/{dataset}_{trainer}_{dtype}_ber{ber}.csv', 'r') as f:
                line = f.readlines()[0]
            data = list(map(float, line.split(',')))
            acc, dim = data[0], data[1]
            print(f'& {acc_format(acc)} & {int(dim)}', end='')
            for i in range(2, len(data), 2):
                al, drr = acc-data[i], dim/data[i+1]
                print(f'& {acc_format(al)} & {drr_format(drr)}', end='')
            print('\\\\\\hline')

if __name__ == '__main__':
    main()