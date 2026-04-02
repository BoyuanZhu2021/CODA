"""
Generate Figure: Confidence Threshold Sensitivity Analysis

Creates a dual-axis plot showing:
- Left Y-axis: Accuracy (%) for FakeTT and FakeSV
- Right Y-axis: LLM Usage (%)
- X-axis: Threshold τ values
- Vertical dashed line at optimal τ=0.75
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from Table 7
tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
fakett_acc = [85.94, 86.59, 87.33, 87.89, 88.15, 88.27, 87.52, 86.78, 85.31, 82.84]
fakesv_acc = [84.75, 85.86, 86.97, 87.43, 87.24, 87.01, 85.67, 84.10, 82.16, 79.35]
llm_usage = [0.00, 3.55, 8.30, 11.43, 16.13, 21.48, 28.07, 36.37, 47.25, 62.44]

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(8, 5))

# Set up the second y-axis
ax2 = ax1.twinx()

# Plot bar chart for LLM usage (background, lowest zorder)
bar_width = 0.035
bars = ax2.bar(tau_values, llm_usage, bar_width, 
               color='lightgray', alpha=0.6, label='LLM Usage (%)', zorder=1)

# Add vertical dashed line at τ=0.75 (middle layer)
ax1.axvline(x=0.75, color='green', linestyle='--', linewidth=2.5, alpha=0.9, zorder=5)

# Plot accuracy lines (foreground, highest zorder)
line1, = ax1.plot(tau_values, fakett_acc, 'o-', color='#2E86AB', 
                  linewidth=2.5, markersize=9, label='FakeTT Accuracy', zorder=10)
line2, = ax1.plot(tau_values, fakesv_acc, 's-', color='#E94F37', 
                  linewidth=2.5, markersize=9, label='FakeSV Accuracy', zorder=10)

# Add annotation for optimal threshold (moved to upper right)
ax1.annotate('Optimal τ=0.75\n(88.27%, 87.01%)', 
             xy=(0.75, 88.27), xytext=(0.88, 90.0),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.95),
             zorder=15)

# Axis labels and formatting
ax1.set_xlabel('Confidence Threshold τ', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12, color='black')
ax2.set_ylabel('LLM Usage (%)', fontsize=12, color='gray')

# Set axis limits
ax1.set_ylim(78, 91)
ax2.set_ylim(0, 70)

# Set x-axis ticks
ax1.set_xticks(tau_values)
ax1.set_xticklabels([f'{t:.2f}' for t in tau_values], fontsize=10)

# Grid (behind everything)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Combined legend (moved to upper left)
from matplotlib.patches import Patch
legend_elements = [
    line1,
    line2,
    Patch(facecolor='lightgray', alpha=0.6, label='LLM Usage (%)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Title
plt.title('Effect of Confidence Threshold τ on Accuracy and LLM Usage', fontsize=13, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save figure
output_dir = 'latex_paper/draw'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'threshold_sensitivity.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PNG for preview
png_path = os.path.join(output_dir, 'threshold_sensitivity.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"PNG preview saved to: {png_path}")

plt.show()

