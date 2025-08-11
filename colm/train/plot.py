import re
import os
import matplotlib.pyplot as plt

# List of log file paths
log_files2 = [
    # "/home/ganesh/CoLM/colm/train/logs/spot_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairot_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairotunw_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairotmultifast_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairotexactcosinefast_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/colm_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairotunwfast_10k.log",
    # "/home/ganesh/CoLM/colm/train/logs/fairotmmulti_10k.log",
    "/home/ganesh/CoLM/colm/train/logs/aug2_colm1k.log",
    "/home/ganesh/CoLM/colm/train/logs/aug2_greats_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug2_spot_cosine.log",
    # "/home/ganesh/CoLM/colm/train/logs/aug2_colm1k.log",
    # "/home/ganesh/CoLM/colm/train/logs/aug2_fairotmulti1k.log",
    "/home/ganesh/CoLM/colm/train/logs/aug2_fairotmulti1k_cosine.log",
    # "/home/ganesh/CoLM/colm/train/logs/aug2_fairot1k_cosine.log",
    
]

# math eff
log_files123 = [
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmeff_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultieff_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatseff_cosine.log",
]

# math normal, mezo
log_files = [
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmnorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_mezo_cosine.log",
]

# math normal, grad
log_files23 = [
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmnorm_grad_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm_grad_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairot_grad_repNorm_cosine.log",
    # "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmnorm_grad_wt_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm_unw_grad_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_grad_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_grad_repNone_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm_grad_repNorm_sgd_cosine.log",
]



log_files = [
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm10k_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmnorm10k_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm10k_grad_repNorm_cosine.log",
]


log_files = [
    "/home/ganesh/CoLM/colm/train/logs/aug3_greatsnorm_b64_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_colmnorm_b64_grad_repNorm_cosine.log",
    "/home/ganesh/CoLM/colm/train/logs/aug3_fairotmultinorm_b64_grad_repNorm_cosine.log",
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
