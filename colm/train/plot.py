import re
import os
import matplotlib.pyplot as plt

# List of log file paths
log_files = [
    "/home/ganesh/CoLM/colm/train/logs/spot_10k.log",
    "/home/ganesh/CoLM/colm/train/logs/fairot_10k.log",
    "/home/ganesh/CoLM/colm/train/logs/fairotmmulti_10k.log"
]


# Output image path
output_image_path = "/home/ganesh/CoLM/colm/train/logs/multiple_loss_plot.png"

# EMA smoothing factor (0.9 = more smoothing, 0.1 = more responsive)
ema_alpha = 0.01

# Function to extract loss values from a log file
def extract_losses(filepath):
    losses = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                match = re.search(r"'loss': ([\d.]+)", line)
                if match:
                    losses.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
    return losses

# Apply Exponential Moving Average
def compute_ema(values, alpha=0.1):
    if not values:
        return []
    ema_values = [values[0]]  # Start with first value
    for val in values[1:]:
        ema = alpha * val + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)
    return ema_values

# Plotting
plt.figure(figsize=(10, 6))
for file in log_files:
    losses = extract_losses(file)
    if losses:
        ema_losses = compute_ema(losses, alpha=ema_alpha)
        label = os.path.basename(file)
        plt.plot(ema_losses, label=f"{label} (EMA)")

plt.title("Smoothed Loss Curves (EMA) from Multiple Log Files")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_image_path)

print(f"EMA-smoothed loss plot saved to: {output_image_path}")
