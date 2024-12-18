import pandas as pd
import matplotlib.pyplot as plt


def acc_format(acc):
    return f'{acc*100:.1f}'


def drr_format(drr):
    return f'{drr:.2f}'


bers = [0.0215, 0.1273]
fig_scale_ratio = 0.6


def print_ber_table():
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



def get_data(ber):
    datasets = ['ucihar', 'isolet', 'mnist']
    trainer = 'LDC'
    dtype = 'binary'
    rows = []
    for dataset in datasets:
        with open(f'output/{dataset}_{trainer}_{dtype}_ber{ber}.csv', 'r') as f:
            line = f.readlines()[0]
        data = list(map(float, line.split(',')))
        rows.append([dataset] + data)
    rel_df = pd.DataFrame(rows, columns=['Dataset', 'Normal Acc', 'Normal Dim', 'Omen Acc(a=0.01)', 'Omen Dim(a=0.01)',
                                     'Omen Acc(a=0.05)', 'Omen Dim(a=0.05)', 'Omen Acc(a=0.10)', 'Omen Dim(a=0.10)',
                                     'SV Baseline Acc', 'SV Baseline Dim', 'Diff Acc', 'Diff Dim',
                                     'Absolute Acc', 'Absolute Dim', 'Mean Acc', 'Mean Dim'])
    rel_df['Omen Acc Drop(a=0.01)'] = 100 * (rel_df['Normal Acc'] - rel_df['Omen Acc(a=0.01)'])
    rel_df['Omen Acc Drop(a=0.05)'] = 100 * (rel_df['Normal Acc'] - rel_df['Omen Acc(a=0.05)'])
    rel_df['Omen Acc Drop(a=0.10)'] = 100 * (rel_df['Normal Acc'] - rel_df['Omen Acc(a=0.10)'])
    rel_df['Diff Acc Drop'] = 100 * (rel_df['Normal Acc'] - rel_df['Diff Acc'])
    rel_df['Absolute Acc Drop'] = 100 * (rel_df['Normal Acc'] - rel_df['Absolute Acc'])
    rel_df['Mean Acc Drop'] = 100 * (rel_df['Normal Acc'] - rel_df['Mean Acc'])
    rel_df['SV Baseline Acc Drop'] = 100 * (rel_df['Normal Acc'] - rel_df['SV Baseline Acc'])
    rel_df['Omen Dim Reduction(a=0.01)'] = rel_df['Normal Dim'] / rel_df['Omen Dim(a=0.01)']
    rel_df['Omen Dim Reduction(a=0.05)'] = rel_df['Normal Dim'] / rel_df['Omen Dim(a=0.05)']
    rel_df['Omen Dim Reduction(a=0.10)'] = rel_df['Normal Dim'] / rel_df['Omen Dim(a=0.10)']
    rel_df['Diff Dim Reduction'] = rel_df['Normal Dim'] / rel_df['Diff Dim']
    rel_df['Absolute Dim Reduction'] = rel_df['Normal Dim'] / rel_df['Absolute Dim']
    rel_df['Mean Dim Reduction'] = rel_df['Normal Dim'] / rel_df['Mean Dim']
    rel_df['SV Baseline Dim Reduction'] = rel_df['Normal Dim'] / rel_df['SV Baseline Dim']
    return rel_df


def print_unoptimized_table():
    rel_df = get_data(bers[0])
    normal_acc1 = rel_df['Normal Acc']
    rel_df = get_data(bers[1])
    normal_acc2 = rel_df['Normal Acc']
    print('\\textbf{Accuracy}', end='')
    for i in range(len(normal_acc1)):
        print(f' & {acc_format(normal_acc1[i])} & {acc_format(normal_acc2[i])}', end='')
    print('\\\\ \\hline')


titlefontsize = 30
axislabelsize = 15
ticklabelsize = 15
symlog_linthresh = 1

def draw_ber_figure(ber):
    rel_df = get_data(ber)
    plt.figure(figsize=(8, 6))
    
    # Filter points with y > threshold for each method
    threshold = 10
    omen_mask = rel_df['Omen Acc Drop(a=0.10)'] <= threshold
    diff_mask = rel_df['Diff Acc Drop'] <= threshold
    abs_mask = rel_df['Absolute Acc Drop'] <= threshold
    mean_mask = rel_df['Mean Acc Drop'] <= threshold
    sv_mask = rel_df['SV Baseline Acc Drop'] <= threshold

    # print(rel_df)
    
    # Define colors for different datasets
    colors = {
        'ucihar': '#e74c3c',  # red
        'isolet': '#3498db',  # blue
        'mnist': '#34495e'   # black
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
    # Plot Omen points for each dataset and alpha
    for dataset in colors:
        dataset_mask = rel_df['Dataset'].str.contains(dataset) & omen_mask
        plt.scatter(rel_df.loc[dataset_mask, f'Omen Dim Reduction(a=0.01)'],
                   rel_df.loc[dataset_mask, 'Omen Acc Drop(a=0.01)'],
                   color=colors[dataset], marker=markers['omen'], label=f'{dataset} (Omen)', s=marker_size)
        plt.scatter(rel_df.loc[dataset_mask, f'Omen Dim Reduction(a=0.05)'],
                   rel_df.loc[dataset_mask, 'Omen Acc Drop(a=0.05)'],
                   color=colors[dataset], marker=markers['omen'], label='_', s=marker_size)
        plt.scatter(rel_df.loc[dataset_mask, f'Omen Dim Reduction(a=0.10)'],
                   rel_df.loc[dataset_mask, 'Omen Acc Drop(a=0.10)'],
                   color=colors[dataset], marker=markers['omen'], label='_', s=marker_size)
        
        # Draw connecting lines for each dataset
        for idx in rel_df[dataset_mask].index:
            x_coords = [rel_df.loc[idx, f'Omen Dim Reduction(a=0.01)'],
                       rel_df.loc[idx, f'Omen Dim Reduction(a=0.05)'],
                       rel_df.loc[idx, f'Omen Dim Reduction(a=0.10)']
                      ]
            y_coords = [rel_df.loc[idx, 'Omen Acc Drop(a=0.01)'],
                       rel_df.loc[idx, 'Omen Acc Drop(a=0.05)'],
                       rel_df.loc[idx, 'Omen Acc Drop(a=0.10)']
                      ]
            plt.plot(x_coords, y_coords, color=colors[dataset], alpha=0.7, linestyle='solid', linewidth=2)
    
    # Plot other methods with method-specific markers
    for dataset in colors:
        dataset_mask = rel_df['Dataset'].str.contains(dataset)
        plt.scatter(rel_df.loc[dataset_mask & diff_mask, f'Diff Dim Reduction'],
                   rel_df.loc[dataset_mask & diff_mask, 'Diff Acc Drop'],
                   color=colors[dataset], marker=markers['diff'], label=f'{dataset} (Diff)', s=marker_size)
        plt.scatter(rel_df.loc[dataset_mask & abs_mask, f'Absolute Dim Reduction'],
                   rel_df.loc[dataset_mask & abs_mask, 'Absolute Acc Drop'],
                   color=colors[dataset], marker=markers['absolute'], label=f'{dataset} (Absolute)', s=marker_size)
        plt.scatter(rel_df.loc[dataset_mask & mean_mask, f'Mean Dim Reduction'],
                   rel_df.loc[dataset_mask & mean_mask, 'Mean Acc Drop'],
                   color=colors[dataset], marker=markers['mean'], label=f'{dataset} (Mean)', s=marker_size)
        plt.scatter(rel_df.loc[dataset_mask & sv_mask, f'SV Baseline Dim Reduction'],
                   rel_df.loc[dataset_mask & sv_mask, 'SV Baseline Acc Drop'],
                   color=colors[dataset], marker=markers['sv'], label=f'{dataset} (SV)', s=marker_size)
    plt.xlabel('Dimension Reduction', fontsize=axislabelsize)
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
    # if dataset == 'lang' or 'ucihar':
    #     max_y += 1 # add 1 to the max y value for better visualization
    plt.ylim(max_y, -0.05)  # Reversed y-axis from max value to 0

    # identify pareto-optimal points for each dataset
    for dataset in colors:
        dataset_mask = rel_df['Dataset'].str.contains(dataset)
        
        # Get all points for this config
        points_x = []
        points_y = []
        
        # Add Omen points
        for alpha in ['0.01', '0.05', '0.10']:
            if any(dataset_mask & omen_mask):
                points_x.append(rel_df.loc[dataset_mask & omen_mask, f'Omen Dim Reduction(a={alpha})'].values[0])
                points_y.append(rel_df.loc[dataset_mask & omen_mask, f'Omen Acc Drop(a={alpha})'].values[0])
        
        # Add other method points
        if any(dataset_mask & diff_mask):
            points_x.append(rel_df.loc[dataset_mask & diff_mask, f'Diff Dim Reduction'].values[0])
            points_y.append(rel_df.loc[dataset_mask & diff_mask, 'Diff Acc Drop'].values[0])
            
        if any(dataset_mask & abs_mask):
            points_x.append(rel_df.loc[dataset_mask & abs_mask, f'Absolute Dim Reduction'].values[0])
            points_y.append(rel_df.loc[dataset_mask & abs_mask, 'Absolute Acc Drop'].values[0])
            
        if any(dataset_mask & mean_mask):
            points_x.append(rel_df.loc[dataset_mask & mean_mask, f'Mean Dim Reduction'].values[0])
            points_y.append(rel_df.loc[dataset_mask & mean_mask, 'Mean Acc Drop'].values[0])

        if any(dataset_mask & sv_mask):
            points_x.append(rel_df.loc[dataset_mask & sv_mask, f'SV Baseline Dim Reduction'].values[0])
            points_y.append(rel_df.loc[dataset_mask & sv_mask, 'SV Baseline Acc Drop'].values[0])
        
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
    if ber == 0.0215:
        xticks = [1, 1.5, 2, 2.5, 3]
    else:
        xticks = [1, 1.5, 2]
    yticks = [0, 2, 4, 6, 8]
    
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

    plt.savefig(f'scatter_ber{ber}.pdf', bbox_inches='tight', format='pdf', dpi=300, transparent=False)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    for ber in bers:
        draw_ber_figure(ber)
    print_unoptimized_table()