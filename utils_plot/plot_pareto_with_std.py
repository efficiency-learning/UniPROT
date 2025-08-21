import matplotlib.pyplot as plt
import numpy as np

# Pareto summary data (from pareto.txt)
methods = ['FairOT (epsilon)', 'Uniform', 'SpotGreedy']
auc_means = [0.7504, 0.6834, 0.6257]
auc_stds = [0.0191, 0.0204, 0.0711]
dpd_means = [0.0321, 0.0692, 0.0083]
dpd_stds = [0.0409, 0.0552, 0.0144]
colors = ['purple', 'blue', 'green']

plt.figure(figsize=(7, 6))
for i in range(len(methods)):
    # Plot mean point
    plt.scatter(dpd_means[i], auc_means[i], color=colors[i], s=100, label=methods[i])
    # Plot error bars
    plt.errorbar(dpd_means[i], auc_means[i], xerr=dpd_stds[i], yerr=auc_stds[i], fmt='o', color=colors[i], alpha=1)
    # Plot shaded region for std
    plt.fill_betweenx(
        [auc_means[i] - auc_stds[i], auc_means[i] + auc_stds[i]],
        dpd_means[i] - dpd_stds[i],
        dpd_means[i] + dpd_stds[i],
        color=colors[i], alpha=0.2
    )

plt.xlabel('Demographic Parity Difference (lower is better)')
plt.ylabel('AUC (higher is better)')
plt.title('Pareto Curve with Std Shaded Region (Mean ± Std over 5 seeds)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pareto_curve_with_std.png')
plt.show()
