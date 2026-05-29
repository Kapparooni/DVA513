import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('benchmark_latency_ereno.csv', sep=';')

# Extract relevant columns - Powerduck MCC (Saved-mcc) vs Ereno MCC (mcc_recomputed_ErenoDATA)
# Also get sparsity and architecture info
df_filtered = df[['model', 'final_sparsity_saved', 'actual_sparsity_saved', 
                  'Saved-mcc', 'mcc_recomputed_ErenoDATA', 'architecture']].copy()

# Clean up sparsity values (replace comma with dot for decimal)
df_filtered['sparsity'] = df_filtered['final_sparsity_saved'].str.replace(',', '.').astype(float)

# Remove any rows with missing MCC values
df_filtered = df_filtered.dropna(subset=['Saved-mcc', 'mcc_recomputed_ErenoDATA'])

# Sort by sparsity for clean plotting
df_filtered = df_filtered.sort_values('sparsity')

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot Powerduck MCC (tested on Powerduck dataset)
ax.plot(df_filtered['sparsity'] * 100, df_filtered['Saved-mcc'], 
        'o-', linewidth=2.5, markersize=6, label='Powerduck Dataset', 
        color='#2E86AB', alpha=0.8)

# Plot Ereno MCC (tested on Ereno dataset)
ax.plot(df_filtered['sparsity'] * 100, df_filtered['mcc_recomputed_ErenoDATA'], 
        's-', linewidth=2.5, markersize=6, label='Ereno Dataset', 
        color='#A23B72', alpha=0.8)

# Customize the plot
ax.set_xlabel('Sparsity (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MCC (Matthews Correlation Coefficient)', fontsize=14, fontweight='bold')
ax.set_title('MCC vs Pruning Sparsity: Powerduck vs Ereno Dataset Performance', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Set axis limits
ax.set_xlim(-2, 102)
ax.set_ylim(-0.2, 1.0)

# Add horizontal line at y=0 and y=0.7 for reference
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Good performance threshold (0.7)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Acceptable threshold (0.5)')

# Add legend
ax.legend(loc='best', fontsize=11, framealpha=0.95)

# Find best performance for each dataset
best_powerduck = df_filtered.loc[df_filtered['Saved-mcc'].idxmax()]
best_ereno = df_filtered.loc[df_filtered['mcc_recomputed_ErenoDATA'].idxmax()]

# Mark best points
ax.plot(best_powerduck['sparsity'] * 100, best_powerduck['Saved-mcc'], 
        'ro', markersize=12, zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)
ax.plot(best_ereno['sparsity'] * 100, best_ereno['mcc_recomputed_ErenoDATA'], 
        'ro', markersize=12, zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)

# Add annotations for best points
ax.annotate(f'Best Powerduck\nMCC={best_powerduck["Saved-mcc"]:.3f}\nat {best_powerduck["sparsity"]*100:.0f}%',
            xy=(best_powerduck['sparsity'] * 100, best_powerduck['Saved-mcc']),
            xytext=(15, 10), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

ax.annotate(f'Best Ereno\nMCC={best_ereno["mcc_recomputed_ErenoDATA"]:.3f}\nat {best_ereno["sparsity"]*100:.0f}%',
            xy=(best_ereno['sparsity'] * 100, best_ereno['mcc_recomputed_ErenoDATA']),
            xytext=(15, -20), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

# Add shaded regions to show performance zones
ax.fill_between([0, 100], 0.7, 1.0, alpha=0.1, color='green', label='High performance zone')
ax.fill_between([0, 100], 0.5, 0.7, alpha=0.1, color='yellow', label='Moderate performance zone')
ax.fill_between([0, 100], -0.2, 0.5, alpha=0.1, color='red', label='Poor performance zone')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('mcc_comparison_powerduck_vs_ereno.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed summary
print("=" * 80)
print("PERFORMANCE COMPARISON: POWERDUCK vs ERENO DATASET")
print("=" * 80)

print(f"\n{'Dataset':<15} {'Best MCC':<12} {'At Sparsity':<12} {'Architecture':<15} {'Model':<40}")
print("-" * 95)

# Best Powerduck
pd_best_row = df_filtered.loc[df_filtered['Saved-mcc'].idxmax()]
print(f"{'Powerduck':<15} {pd_best_row['Saved-mcc']:.4f}{'':<8} "
      f"{pd_best_row['sparsity']*100:.1f}%{'':<8} "
      f"{pd_best_row['architecture']:<15} "
      f"{pd_best_row['model'][:40]:<40}")

# Best Ereno
er_best_row = df_filtered.loc[df_filtered['mcc_recomputed_ErenoDATA'].idxmax()]
print(f"{'Ereno':<15} {er_best_row['mcc_recomputed_ErenoDATA']:.4f}{'':<8} "
      f"{er_best_row['sparsity']*100:.1f}%{'':<8} "
      f"{er_best_row['architecture']:<15} "
      f"{er_best_row['model'][:40]:<40}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS:")
print("=" * 80)

# Find sparsity where MCC drops below 0.7
pd_above_07 = df_filtered[df_filtered['Saved-mcc'] > 0.7]
er_above_07 = df_filtered[df_filtered['mcc_recomputed_ErenoDATA'] > 0.7]

if len(pd_above_07) > 0:
    max_sp_pd = pd_above_07['sparsity'].max()
    print(f"\n✓ Powerduck maintains MCC > 0.7 up to {max_sp_pd*100:.0f}% sparsity")
else:
    print(f"\n✗ Powerduck never achieves MCC > 0.7")

if len(er_above_07) > 0:
    max_sp_er = er_above_07['sparsity'].max()
    print(f"✓ Ereno maintains MCC > 0.7 up to {max_sp_er*100:.0f}% sparsity")
else:
    print(f"✗ Ereno never achieves MCC > 0.7")

# Find best architecture for each
print(f"\n✓ Best performing architecture for Powerduck: {pd_best_row['architecture']}")
print(f"  (MCC = {pd_best_row['Saved-mcc']:.3f} at {pd_best_row['sparsity']*100:.0f}% sparsity)")
print(f"\n✓ Best performing architecture for Ereno: {er_best_row['architecture']}")
print(f"  (MCC = {er_best_row['mcc_recomputed_ErenoDATA']:.3f} at {er_best_row['sparsity']*100:.0f}% sparsity)")

# Compare robustness
print("\n" + "=" * 80)
print("ROBUSTNESS TO PRUNING:")
print("=" * 80)

# Calculate performance drop at high sparsity
high_sparsity = df_filtered[df_filtered['sparsity'] >= 0.8]
if len(high_sparsity) > 0:
    pd_high_avg = high_sparsity['Saved-mcc'].mean()
    er_high_avg = high_sparsity['mcc_recomputed_ErenoDATA'].mean()
    print(f"\nAt ≥80% sparsity:")
    print(f"  Powerduck average MCC: {pd_high_avg:.3f}")
    print(f"  Ereno average MCC: {er_high_avg:.3f}")
    
    if pd_high_avg > er_high_avg:
        print(f"  → Powerduck is more robust at high sparsity")
    else:
        print(f"  → Ereno is more robust at high sparsity")