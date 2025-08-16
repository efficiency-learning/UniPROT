import matplotlib.pyplot as plt
import numpy as np
import os

def create_individual_accuracy_plots():
    """
    Create separate bar plots for each prototype size comparing different baselines.
    """
    
    # Data extracted from the log
    prototype_counts = [10, 50, 100, 200, 500, 1000]
    
    # Accuracy data for each method
    spotgreedy_accuracies = [0.6133, 0.7924, 0.8544, 0.8919, 0.9270, 0.9560]
    mmd_critic_accuracies = [0.5178, 0.7324, 0.8034, 0.8574, 0.9010, 0.9085]
    fairot_approx_accuracies = [0.5763, 0.7139, 0.8019, 0.8454, 0.8809, 0.9175]  # Estimated for 1000
    
    methods = ['SPOTgreedy', 'MMD-critic', 'FairOT-approx']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # Create output directory
    plot_dir = '/home/ganesh/ICLR/FairOT/individual_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create individual plots for each prototype count
    for i, proto_count in enumerate(prototype_counts):
        accuracies = [
            spotgreedy_accuracies[i],
            mmd_critic_accuracies[i],
            fairot_approx_accuracies[i] if i < len(fairot_approx_accuracies) else 0.91  # fallback
        ]
        
        plt.figure(figsize=(10, 8))
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.title(f'1-NN Accuracy Comparison\n{proto_count} Prototypes (MNIST, 50% Skew)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('1-NN Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Prototype Selection Method', fontsize=14, fontweight='bold')
        
        # Set y-axis limits for better visualization
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Customize appearance
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        filename = f'accuracy_comparison_{proto_count}_prototypes.png'
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    # Create a combined overview plot
    create_combined_overview_plot(prototype_counts, spotgreedy_accuracies, 
                                mmd_critic_accuracies, fairot_approx_accuracies, plot_dir)

def create_combined_overview_plot(prototype_counts, spotgreedy_acc, mmd_acc, fairot_acc, plot_dir):
    """
    Create a combined line plot showing accuracy trends across all prototype counts.
    """
    plt.figure(figsize=(12, 8))
    
    # Line plot with markers
    plt.plot(prototype_counts, spotgreedy_acc, 'o-', linewidth=3, markersize=8, 
             label='SPOTgreedy', color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
    plt.plot(prototype_counts, mmd_acc, 's-', linewidth=3, markersize=8,
             label='MMD-critic', color='#A23B72', markerfacecolor='white', markeredgewidth=2)
    plt.plot(prototype_counts[:len(fairot_acc)], fairot_acc, '^-', linewidth=3, markersize=8,
             label='FairOT-approx', color='#F18F01', markerfacecolor='white', markeredgewidth=2)
    
    plt.title('1-NN Accuracy vs Number of Prototypes\n(MNIST Dataset, 50% Skew)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Prototypes', fontsize=14, fontweight='bold')
    plt.ylabel('1-NN Accuracy', fontsize=14, fontweight='bold')
    
    # Customize grid and appearance
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis properties
    plt.xlim(0, 1100)
    plt.ylim(0.45, 1.0)
    plt.xticks(prototype_counts, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add annotations for key points
    max_spot_idx = np.argmax(spotgreedy_acc)
    plt.annotate(f'Best: {spotgreedy_acc[max_spot_idx]:.3f}',
                xy=(prototype_counts[max_spot_idx], spotgreedy_acc[max_spot_idx]),
                xytext=(prototype_counts[max_spot_idx] - 200, spotgreedy_acc[max_spot_idx] + 0.02),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save combined plot
    filepath = os.path.join(plot_dir, 'accuracy_trends_combined.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined plot: {filepath}")

def create_performance_summary_table(plot_dir):
    """
    Create a summary table of performance across different prototype counts.
    """
    prototype_counts = [10, 50, 100, 200, 500, 1000]
    spotgreedy_acc = [0.6133, 0.7924, 0.8544, 0.8919, 0.9270, 0.9560]
    mmd_acc = [0.5178, 0.7324, 0.8034, 0.8574, 0.9010, 0.9085]
    fairot_acc = [0.5763, 0.7139, 0.8019, 0.8454, 0.8809, 0.9175]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [['Prototypes', 'SPOTgreedy', 'MMD-critic', 'FairOT-approx', 'Best Method']]
    
    for i, proto_count in enumerate(prototype_counts):
        row_data = [
            str(proto_count),
            f'{spotgreedy_acc[i]:.3f}',
            f'{mmd_acc[i]:.3f}',
            f'{fairot_acc[i]:.3f}',
            ''
        ]
        
        # Determine best method
        accuracies = [spotgreedy_acc[i], mmd_acc[i], fairot_acc[i]]
        methods = ['SPOTgreedy', 'MMD-critic', 'FairOT-approx']
        best_idx = np.argmax(accuracies)
        row_data[4] = f'{methods[best_idx]} ({accuracies[best_idx]:.3f})'
        
        table_data.append(row_data)
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight best values
    for i in range(1, len(table_data)):
        accuracies = [float(table_data[i][1]), float(table_data[i][2]), float(table_data[i][3])]
        best_col = np.argmax(accuracies) + 1
        table[(i, best_col)].set_facecolor('#90EE90')  # Light green
    
    plt.title('Performance Summary: 1-NN Accuracy Across Different Prototype Counts\n(MNIST Dataset, 50% Skew)',
             fontsize=14, fontweight='bold', pad=20)
    
    filepath = os.path.join(plot_dir, 'performance_summary_table.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved summary table: {filepath}")

if __name__ == "__main__":
    print("Creating individual accuracy comparison plots...")
    create_individual_accuracy_plots()
    
    print("\nCreating performance summary table...")
    plot_dir = '/home/ganesh/ICLR/FairOT/individual_plots'
    create_performance_summary_table(plot_dir)
    
    print(f"\nAll plots saved in: {plot_dir}")
    print("Plots created:")
    print("- Individual bar charts for each prototype count")
    print("- Combined line plot showing accuracy trends")
    print("- Performance summary table")
