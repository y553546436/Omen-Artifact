import matplotlib.pyplot as plt
import numpy as np

axis_label_size = 20
tick_label_size = 20
legend_size = 17
tick_length = 10
tick_width = 2

def draw_latency_breakdown(latencies1, latencies2, label1, label2):
    if not latencies1 or not latencies2:
        raise ValueError("The lists of latencies cannot be empty.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#34495e', '#3498db', '#27ae60']
    labels = ['Encoding', 'Distance Computation', 'Statistical Tests']
    
    def plot_bar(y, latencies, label):
        left = 0
        for i, (latency, color) in enumerate(zip(latencies, colors)):
            ax.barh(y, latency, left=left, height=0.3, align='center', color=color, label=labels[i])
            left += latency
        ax.text(-0.05, y, label, ha='right', va='center', fontsize=axis_label_size, fontweight='bold')
    
    plot_bar(0.2, latencies1, label1)
    plot_bar(-0.2, latencies2, label2)
    
    ax.set_xlabel('Latency (ms)', fontsize=axis_label_size, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, max(sum(latencies1), sum(latencies2)))
    ax.tick_params(axis='x', which='major', length=tick_length, width=tick_width, labelsize=tick_label_size)
    ax.tick_params(axis='y', which='major', length=tick_length, width=tick_width, labelsize=tick_label_size)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), 
              bbox_to_anchor=(0.4, -0.4), loc='lower center', ncol=max(len(latencies1), len(latencies2)),
              fontsize=legend_size)
    # Adjust font weight of legend text
    for text in ax.legend_.get_texts():
        text.set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the bottom margin to make room for the legend
    return fig


def print_table(unoptimized, omen):
    print("\\textbf{Baseline} & Encode & Distance Compute & Statistical Tests & \\textbf{Total} \\\\")
    print("\\hline")
    
    # Unoptimized row
    print(f"\\texttt{{Unoptimized}} & {unoptimized[0]:.2f} & {unoptimized[1]:.2f} & - & {sum(unoptimized):.2f} \\\\")
    print("\\hline")
    
    # Omen row
    print(f"\\tool{{}} $\\alpha=5\%$ & {omen[0]:.2f} & {omen[1]:.2f} & {omen[2]:.2f} & {sum(omen):.2f} \\\\")
    print("\\hline")

with open('breakdown_ucihar_OnlineHD_binary_linear_s512_f64_a005.csv', 'r') as f:
# with open('breakdown_mnist_LDC_binary_linear_s16_f4_a005.csv', 'r') as f:
    import csv
    reader = csv.DictReader(f)
    data = list(reader)
    n = len(data)

unoptimized_encode = sum(float(d['Normal Encode Time']) for d in data) / n
unoptimized_distance = sum(float(d['Normal Distance Time']) for d in data) / n
omen_encode = sum(float(d['Omen Encode Time']) for d in data) / n
omen_distance = sum(float(d['Omen Distance Time']) for d in data) / n
omen_test = sum(float(d['Omen Test Time']) for d in data) / n
print(f'unoptimized_encode: {unoptimized_encode}, percentage: {unoptimized_encode / (unoptimized_encode + unoptimized_distance) * 100:.2f}%')
print(f'unoptimized_distance: {unoptimized_distance}, percentage: {unoptimized_distance / (unoptimized_encode + unoptimized_distance) * 100:.2f}%')
print(f'omen_encode: {omen_encode}, percentage: {omen_encode / (omen_encode + omen_distance + omen_test) * 100:.2f}%')
print(f'omen_distance: {omen_distance}, percentage: {omen_distance / (omen_encode + omen_distance + omen_test) * 100:.2f}%')
print(f'omen_test: {omen_test}, percentage: {omen_test / (omen_encode + omen_distance + omen_test) * 100:.2f}%')
unoptimized = [unoptimized_encode, unoptimized_distance]
omen = [omen_encode, omen_distance, omen_test]
print_table(unoptimized, omen)
# fig = draw_latency_breakdown(unoptimized, omen, "Unoptimized", "Omen \u03B1=5%")
# plt.savefig('latency_breakdown_ucihar_OnlineHD_binary.pdf', dpi=300)
# plt.savefig('latency_breakdown_mnist_LDC_binary.pdf', dpi=300)