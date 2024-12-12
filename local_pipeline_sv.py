import os
import subprocess
import argparse
import serial
from rich.console import Console

from smaller_vector_baseline import get_omen_dim

console = Console()

parser = argparse.ArgumentParser(description='Run Omen experiments')
args = parser.parse_args()

# datasets: mnist ucihar isolet language
# trainers: LeHDC LDC OnlineHD
# dtypes: binary real
# do not run real dtype on LeHDC and OnlineHD
# LeHDC dim: 5056
# OnlineHD dim: 10048
# LDC dim: 256 (both binary and real)
# LeHDC and OnlineHD binary start: 512
# LeHDC and OnlineHD binary freq: 64
# LDC binary start: 16
# LDC binary freq: 4
# LDC real start: 16
# LDC real freq: 4
# alphas: 0.01 0.05 0.10
# strategy: linear
# language dataset only run binary OnlineHD

# all configurations
def trainer_args(trainer, dtype):
    trainer_args_templates = {
        'LeHDC': f'--dim 5056 {"-b -m -1 -M 1 -l 2" if dtype == "binary" else ""}',
        'LDC': f'{"--fd 256 --vd 4 -b" if dtype == "binary" else "--dim 256"} --epochs 100',
        'OnlineHD': f'--dim 10048 {"-b -m -1 -M 1 -l 2" if dtype == "binary" else ""}',
    }
    return f"'{trainer_args_templates[trainer]}'"


def make_command(dataset, dtype, trainer, start, freq, alpha, strategy):
    cmd = f'make -f local_sv.mk DATASET={dataset} DTYPE={dtype} TRAINER={trainer} TRAINER_ARGS={trainer_args(trainer, dtype)} START={start} FREQ={freq} ALPHA={alpha} STRATEGY={strategy} CUTOFF={get_omen_dim(dataset, trainer, dtype)}'
    return cmd


def generate_configs():
    # generate all configurations from the description above
    # return a list of commands
    commands = []
    datasets = ['language', 'ucihar', 'isolet', 'mnist']
    trainers = ['LeHDC', 'OnlineHD', 'LDC']
    dtypes = ['real', 'binary']
    alphas = ['0.01', '0.05', '0.10']
    strategies = ['linear']
    for trainer in trainers:
        for dataset in datasets:
            for dtype in dtypes:
                if dataset == 'language' and trainer != 'OnlineHD':
                    continue
                if dataset == 'language' and dtype == 'real':
                    continue
                start = ('512' if dtype == 'binary' else '128') if trainer in ['LeHDC', 'OnlineHD'] else '16'
                freq = '64' if trainer in ['LeHDC', 'OnlineHD'] else '4'
                for alpha in alphas:
                    for strategy in strategies:
                        commands.append(make_command(dataset, dtype, trainer, start, freq, alpha, strategy))
    return commands


def run_command(cmd):
    # run the command and print the output
    success = True
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        activate_cmd = f'eval "$(conda shell.bash hook)" && conda activate {conda_prefix.split("/")[-1]}'
        full_cmd = f'{activate_cmd} && {cmd}'
    else:
        full_cmd = cmd
    console.print(f'Running command: {full_cmd}', style='bold blue')
    try:
        subprocess.run(full_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f'Error running command: {cmd}: {e}', style='bold yellow')
        success = False
    except serial.SerialException as e:
        console.print(f"Serial error in configuration {cmd}: {e}", style='bold yellow')
        success = False
    except Exception as e:
        console.print(f"Unexpected error in configuration {cmd}: {e}", style='bold yellow')
        success = False
    finally:
        console.print(f'Command {cmd} {"succeeded" if success else "failed"}', style='bold green' if success else 'bold red')
    return success


if __name__ == '__main__':
    commands = generate_configs()
    failed = []
    for cmd in commands:
        if not run_command(cmd):
            failed.append(cmd)
    print(f'Failed configurations: {failed}')
    if failed:
        print('Writing failed configurations to failed_configs.txt')
        print(f'num failed: {len(failed)}')
        with open('failed_configs.txt', 'w') as f:
            f.write('\n'.join(failed))
    else:
        print('All configurations succeeded')