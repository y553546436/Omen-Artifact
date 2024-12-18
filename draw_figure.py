import os
import re
import numpy as np
import pandas as pd
from parse_results import parse_row, extract_time
import matplotlib.pyplot as plt

def get_data(all_local=False):
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
    return rel_df


titlefontsize = 30
axislabelsize = 15
ticklabelsize = 15
symlog_linthresh = 1
fig_scale_ratio = 0.6

def draw_scatter(rel_df, dataset, x_option='Speedup'):
    plt.figure(figsize=(8, 6))

    # Filter to only keep isolet dataset
    rel_df = rel_df[rel_df['Configuration'].str.contains(dataset)]
    
    # Filter points with y > threshold for each method
    threshold = 5
    if dataset == 'ucihar':
        threshold = 10
    omen_mask = rel_df['Omen Acc Drop(a=0.10)'] <= threshold
    diff_mask = rel_df['Diff Acc Drop'] <= threshold
    abs_mask = rel_df['Absolute Acc Drop'] <= threshold
    mean_mask = rel_df['Mean Acc Drop'] <= threshold
    sv_mask = rel_df['SV Baseline Acc Drop'] <= threshold

    # print(rel_df)
    
    # Define markers for different configurations
    # Define colors for different configurations
    colors = {
        f'OHD-B-{dataset}': '#e74c3c',  # red
        f'OHD-M-{dataset}': '#3498db',  # blue  
        f'LeH-B-{dataset}': '#2ecc71',  # green
        f'LeH-M-{dataset}': '#9b59b6',  # purple
        f'LDC-B-{dataset}': '#7f8c8d',  # grey
        f'LDC-M-{dataset}': '#34495e'   # black
    }
    # only one configuration for language dataset
    if dataset == 'lang':
        colors = {
            f'OHD-B-{dataset}': '#e74c3c',  # red
        }

    markers = {
        'omen': 'o',
        'diff': 'X',
        'absolute': '>',
        'mean': 'D',
        'sv': '*'
    }
    
    # Define markers for different methods
    marker_size = 100
    # Plot Omen points for each configuration and alpha
    for config in colors:
        config_mask = rel_df['Configuration'].str.contains(config) & omen_mask
        plt.scatter(rel_df.loc[config_mask, f'Omen {x_option}(a=0.01)'],
                   rel_df.loc[config_mask, 'Omen Acc Drop(a=0.01)'],
                   color=colors[config], marker=markers['omen'], label=f'{config} (Omen)', s=marker_size)
        plt.scatter(rel_df.loc[config_mask, f'Omen {x_option}(a=0.05)'],
                   rel_df.loc[config_mask, 'Omen Acc Drop(a=0.05)'],
                   color=colors[config], marker=markers['omen'], label='_', s=marker_size)
        plt.scatter(rel_df.loc[config_mask, f'Omen {x_option}(a=0.10)'],
                   rel_df.loc[config_mask, 'Omen Acc Drop(a=0.10)'],
                   color=colors[config], marker=markers['omen'], label='_', s=marker_size)
        
        # Draw connecting lines for each configuration
        for idx in rel_df[config_mask].index:
            x_coords = [rel_df.loc[idx, f'Omen {x_option}(a=0.01)'],
                       rel_df.loc[idx, f'Omen {x_option}(a=0.05)'],
                       rel_df.loc[idx, f'Omen {x_option}(a=0.10)']
                      ]
            y_coords = [rel_df.loc[idx, 'Omen Acc Drop(a=0.01)'],
                       rel_df.loc[idx, 'Omen Acc Drop(a=0.05)'],
                       rel_df.loc[idx, 'Omen Acc Drop(a=0.10)']
                      ]
            plt.plot(x_coords, y_coords, color=colors[config], alpha=0.7, linestyle='solid', linewidth=2)
    
    # Plot other methods with method-specific markers
    for config in colors:
        config_mask = rel_df['Configuration'].str.contains(config)
        plt.scatter(rel_df.loc[config_mask & diff_mask, f'Diff {x_option}'],
                   rel_df.loc[config_mask & diff_mask, 'Diff Acc Drop'],
                   color=colors[config], marker=markers['diff'], label=f'{config} (Diff)', s=marker_size)
        plt.scatter(rel_df.loc[config_mask & abs_mask, f'Absolute {x_option}'],
                   rel_df.loc[config_mask & abs_mask, 'Absolute Acc Drop'],
                   color=colors[config], marker=markers['absolute'], label=f'{config} (Absolute)', s=marker_size)
        plt.scatter(rel_df.loc[config_mask & mean_mask, f'Mean {x_option}'],
                   rel_df.loc[config_mask & mean_mask, 'Mean Acc Drop'],
                   color=colors[config], marker=markers['mean'], label=f'{config} (Mean)', s=marker_size)
        plt.scatter(rel_df.loc[config_mask & sv_mask, f'SV Baseline {x_option}'],
                   rel_df.loc[config_mask & sv_mask, 'SV Baseline Acc Drop'],
                   color=colors[config], marker=markers['sv'], label=f'{config} (SV)', s=marker_size)
    plt.xlabel('Speedup' if x_option == 'Speedup' else 'Dimension Reduction', fontsize=axislabelsize)
    plt.ylabel('Accuracy Loss (%)', fontsize=axislabelsize)
    # plt.title(f'{dataset}', fontsize=titlefontsize)
    plt.grid(False)
    # plt.legend()
    plt.xscale('log')
    plt.yscale('symlog', linthresh=symlog_linthresh)
    max_y = max([max(rel_df.loc[omen_mask, 'Omen Acc Drop(a=0.01)'].max(),
                     rel_df.loc[omen_mask, 'Omen Acc Drop(a=0.05)'].max(),
                     rel_df.loc[omen_mask, 'Omen Acc Drop(a=0.10)'].max(),
                     rel_df.loc[diff_mask, 'Diff Acc Drop'].max(),
                     rel_df.loc[abs_mask, 'Absolute Acc Drop'].max(), 
                     rel_df.loc[mean_mask, 'Mean Acc Drop'].max(),
                     rel_df.loc[sv_mask, 'SV Baseline Acc Drop'].max())])
    if dataset == 'lang' or 'ucihar':
        max_y += 1 # add 1 to the max y value for better visualization
    plt.ylim(max_y, -0.05)  # Reversed y-axis from max value to 0

    # identify pareto-optimal points for each config
    for config in colors:
        config_mask = rel_df['Configuration'].str.contains(config)
        
        # Get all points for this config
        points_x = []
        points_y = []
        
        # Add Omen points
        for alpha in ['0.01', '0.05', '0.10']:
            if any(config_mask & omen_mask):
                points_x.append(rel_df.loc[config_mask & omen_mask, f'Omen {x_option}(a={alpha})'].values[0])
                points_y.append(rel_df.loc[config_mask & omen_mask, f'Omen Acc Drop(a={alpha})'].values[0])
        
        # Add other method points
        if any(config_mask & diff_mask):
            points_x.append(rel_df.loc[config_mask & diff_mask, f'Diff {x_option}'].values[0])
            points_y.append(rel_df.loc[config_mask & diff_mask, 'Diff Acc Drop'].values[0])
            
        if any(config_mask & abs_mask):
            points_x.append(rel_df.loc[config_mask & abs_mask, f'Absolute {x_option}'].values[0])
            points_y.append(rel_df.loc[config_mask & abs_mask, 'Absolute Acc Drop'].values[0])
            
        if any(config_mask & mean_mask):
            points_x.append(rel_df.loc[config_mask & mean_mask, f'Mean {x_option}'].values[0])
            points_y.append(rel_df.loc[config_mask & mean_mask, 'Mean Acc Drop'].values[0])

        if any(config_mask & sv_mask):
            points_x.append(rel_df.loc[config_mask & sv_mask, f'SV Baseline {x_option}'].values[0])
            points_y.append(rel_df.loc[config_mask & sv_mask, 'SV Baseline Acc Drop'].values[0])
        
        # Find pareto-optimal points
        for i in range(len(points_x)):
            is_pareto = True
            for j in range(len(points_x)):
                if i != j:
                    # Check if point j dominates point i
                    if points_x[j] >= points_x[i] and points_y[j] <= points_y[i]:
                        if points_x[j] > points_x[i] or points_y[j] < points_y[i]:
                            is_pareto = False
                            break
            
            if is_pareto:
                # Plot a hollow rectangle marker behind the point
                plt.scatter(points_x[i], points_y[i], marker='s', facecolors='#f1c40f', 
                          edgecolors='#f39c12', alpha=0.7, s=marker_size+100, 
                          linewidth=2, zorder=0, linestyle='solid')
    
    # Generate linear tick positions
    xticks = [1, 2, 3, 4, 5, 6, 7, 8]
    yticks = [0, 1, 2, 3, 4, 5]
    if dataset == 'lang':
        yticks = [0, 1, 2, 3, 4, 5]
        xticks = [2, 4, 6, 8, 10, 12, 14]
    if dataset == 'ucihar':
        yticks = [0, 2, 4, 6, 8]
    if dataset == 'isolet':
        yticks = [0, 1, 2, 3]
    if dataset == 'mnist':
        yticks = [0, 1, 2, 3]
        xticks = [1, 2, 3, 4, 5]
    
    # Get current axes
    ax = plt.gca()
    
    # Set major ticks with labels
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=ticklabelsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=ticklabelsize)
    
    # Remove minor ticks
    ax.minorticks_off()

    # Adjust figure height to be shorter
    fig = plt.gcf()
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1]*fig_scale_ratio)

    plt.savefig(f'scatter_{dataset}_{x_option}.pdf', bbox_inches='tight', format='pdf', dpi=300, transparent=False)
    plt.clf()
    plt.close()


def draw_su_dim_fit_line(rel_df):
    # collect all speedup / dim reduction for Omens
    speedup_columns = ['Omen Speedup(a=0.01)', 'Omen Speedup(a=0.05)', 'Omen Speedup(a=0.10)']
    dim_reduction_columns = ['Omen Dim Reduction(a=0.01)', 'Omen Dim Reduction(a=0.05)', 'Omen Dim Reduction(a=0.10)']
    # concatenate all dim reduction vs speedup data points
    speedup_data = pd.concat([rel_df['Configuration'], rel_df[speedup_columns]], axis=1)
    dim_reduction_data = pd.concat([rel_df['Configuration'], rel_df[dim_reduction_columns]], axis=1)
    # print(speedup_data)
    # print(dim_reduction_data)
    
    # separate LDC-B and non-LDC-B configurations
    ldc_b_speedup = speedup_data[speedup_data['Configuration'].str.contains('LDC-B')]
    ldc_b_dim = dim_reduction_data[dim_reduction_data['Configuration'].str.contains('LDC-B')]
    non_ldc_b_speedup = speedup_data[~speedup_data['Configuration'].str.contains('LDC-B')]
    non_ldc_b_dim = dim_reduction_data[~dim_reduction_data['Configuration'].str.contains('LDC-B')]

    # put in a same table for fitting a linear regression of speedup vs dim reduction
    non_ldc_b_all_speedup = pd.concat([non_ldc_b_speedup['Omen Speedup(a=0.01)'], non_ldc_b_speedup['Omen Speedup(a=0.05)'], non_ldc_b_speedup['Omen Speedup(a=0.10)']])
    non_ldc_b_all_dim_reduction = pd.concat([non_ldc_b_dim['Omen Dim Reduction(a=0.01)'], non_ldc_b_dim['Omen Dim Reduction(a=0.05)'], non_ldc_b_dim['Omen Dim Reduction(a=0.10)']])
    
    ldc_b_speedup_all = pd.concat([ldc_b_speedup['Omen Speedup(a=0.01)'], ldc_b_speedup['Omen Speedup(a=0.05)'], ldc_b_speedup['Omen Speedup(a=0.10)']])
    ldc_b_dim_all = pd.concat([ldc_b_dim['Omen Dim Reduction(a=0.01)'], ldc_b_dim['Omen Dim Reduction(a=0.05)'], ldc_b_dim['Omen Dim Reduction(a=0.10)']])
    
    # Create scatter plot of actual data points
    marker_size = 200
    plt.figure(figsize=(8, 6))
    plt.scatter(non_ldc_b_all_dim_reduction, non_ldc_b_all_speedup, color='#2980b9', alpha=0.5, s=marker_size, label='Non-LDC-B points')
    plt.scatter(ldc_b_dim_all, ldc_b_speedup_all, color='#e67e22', alpha=0.5, s=marker_size, label='LDC-B points')
    
    # linear regression
    from sklearn.linear_model import LinearRegression
    
    # Fit line for non-LDC-B points
    lr_non_ldc = LinearRegression()
    lr_non_ldc.fit(non_ldc_b_all_dim_reduction.values.reshape(-1, 1), non_ldc_b_all_speedup.values.reshape(-1, 1))
    
    # Fit line for LDC-B points
    lr_ldc = LinearRegression()
    lr_ldc.fit(ldc_b_dim_all.values.reshape(-1, 1), ldc_b_speedup_all.values.reshape(-1, 1))
    
    # Generate points for fitted lines using shared x range
    x_min = min(non_ldc_b_all_dim_reduction.min(), ldc_b_dim_all.min())
    x_max = max(non_ldc_b_all_dim_reduction.max(), ldc_b_dim_all.max())
    x_range = np.linspace(x_min, x_max, 100)
    
    # Generate predictions
    y_pred_non_ldc = lr_non_ldc.predict(x_range.reshape(-1, 1))
    y_pred_ldc = lr_ldc.predict(x_range.reshape(-1, 1))
    
    # Plot fitted lines
    linewidth = 5
    linestyle = 'dashed'
    plt.plot(x_range, y_pred_non_ldc, color='#34495e', linewidth=linewidth, linestyle=linestyle,
             label=f'Non-LDC-B fitted line\ny = {lr_non_ldc.coef_[0][0]:.3f}x + {lr_non_ldc.intercept_[0]:.3f}')
    plt.plot(x_range, y_pred_ldc, color='#c0392b', linewidth=linewidth, linestyle=linestyle,
             label=f'LDC-B fitted line\ny = {lr_ldc.coef_[0][0]:.3f}x + {lr_ldc.intercept_[0]:.3f}')
    plt.xlabel('Dimension Reduction Ratio', fontsize=axislabelsize)
    plt.ylabel('Speedup', fontsize=axislabelsize)
    plt.xticks(fontsize=ticklabelsize)
    plt.yticks(fontsize=ticklabelsize)
    
    # Adjust figure height to be shorter
    fig = plt.gcf()
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1]*fig_scale_ratio)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('speedup_vs_dim_reduction.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    
    print("Non-LDC-B case:")
    print(lr_non_ldc.coef_, lr_non_ldc.intercept_)
    print(f'Speedup = {lr_non_ldc.coef_[0][0]:.3f} * Dim Reduction + {lr_non_ldc.intercept_[0]:.3f}')
    print("\nLDC-B case:")
    print(lr_ldc.coef_, lr_ldc.intercept_)
    print(f'Speedup = {lr_ldc.coef_[0][0]:.3f} * Dim Reduction + {lr_ldc.intercept_[0]:.3f}')
    # calculate mean squared error for both cases
    from sklearn.metrics import mean_squared_error
    
    # MSE for non-LDC-B case
    y_pred_non_ldc_mse = lr_non_ldc.predict(non_ldc_b_all_dim_reduction.values.reshape(-1, 1))
    mse_non_ldc = mean_squared_error(non_ldc_b_all_speedup.values.reshape(-1, 1), y_pred_non_ldc_mse)
    print(f'Mean Squared Error (Non-LDC-B): {mse_non_ldc:.3f}')
    
    # MSE for LDC-B case
    y_pred_ldc_mse = lr_ldc.predict(ldc_b_dim_all.values.reshape(-1, 1))
    mse_ldc = mean_squared_error(ldc_b_speedup_all.values.reshape(-1, 1), y_pred_ldc_mse)
    print(f'Mean Squared Error (LDC-B): {mse_ldc:.3f}')


def print_unoptimized_table(rel_df):
    # print the unoptimized table
    normal_accs = rel_df['Normal Acc']
    print('\\textbf{Accuracy (\\%)} & ', end='')
    print(' & '.join([f'{acc:.1f}' for acc in normal_accs]), end=' \\\\ \\hline\n')
    normal_dims = rel_df['Normal Dim']
    print('\\textbf{\\# Dimensions} & ', end='')
    print(' & '.join([f'{dim}' for dim in normal_dims]), end=' \\\\ \\hline\n')
    normal_times = rel_df['Normal Time']
    print('\\textbf{Time} & ', end='')
    print(' & '.join([f'\\localtime{{{time:.1f}}}' if ('OHD' in config or 'LeH' in config) and '-M-' in config else f'\\mcutime{{{time:.1f}}}' for time, config in zip(normal_times, rel_df['Configuration'])]), end=' \\\\ \\hline\n')


if __name__ == '__main__':
    rel_df = get_data()
    draw_scatter(rel_df, 'isolet', 'Speedup')
    draw_scatter(rel_df, 'ucihar', 'Speedup')
    draw_scatter(rel_df, 'lang', 'Speedup')
    draw_scatter(rel_df, 'mnist', 'Speedup')
    draw_su_dim_fit_line(rel_df)
    print_unoptimized_table(rel_df)