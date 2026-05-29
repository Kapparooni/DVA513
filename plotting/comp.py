import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('inputs/benchmark_latency_ereno.csv', sep=';')

# Extract and clean the data
df_filtered = df[['model', 'final_sparsity_saved', 'Saved-mcc', 'mcc_recomputed_ErenoDATA']].copy()

# Convert values
df_filtered['sparsity'] = df_filtered['final_sparsity_saved'].astype(str).str.replace(',', '.').astype(float)
df_filtered['Saved-mcc'] = df_filtered['Saved-mcc'].astype(str).str.replace(',', '.').astype(float)
df_filtered['mcc_recomputed_ErenoDATA'] = df_filtered['mcc_recomputed_ErenoDATA'].astype(str).str.replace(',', '.').astype(float)

# Extract architecture number from model name
df_filtered['architecture'] = df_filtered['model'].str.extract(r'Architecture(\d+)').astype(float)

# Remove any rows with NaN
df_filtered = df_filtered.dropna()

print("Architectures found:", df_filtered['architecture'].unique())
print(f"\nTotal data points: {len(df_filtered)}")

# Create a figure with subplots for each architecture
architectures = sorted(df_filtered['architecture'].unique())
n_arch = len(architectures)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, arch in enumerate(architectures):
    if idx < len(axes):
        ax = axes[idx]
        arch_data = df_filtered[df_filtered['architecture'] == arch]
        
        # Sort by sparsity
        arch_data = arch_data.sort_values('sparsity')
        
        # Plot both datasets
        ax.plot(arch_data['sparsity'] * 100, arch_data['Saved-mcc'], 
                'o-', linewidth=2.5, markersize=6, label='Powerduck Dataset', 
                color='#2E86AB', alpha=0.8)
        ax.plot(arch_data['sparsity'] * 100, arch_data['mcc_recomputed_ErenoDATA'], 
                's-', linewidth=2.5, markersize=6, label='Ereno Dataset', 
                color='#A23B72', alpha=0.8)
        
        # Customize subplot
        ax.set_xlabel('Sparsity (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('MCC', fontsize=10, fontweight='bold')
        ax.set_title(f'Architecture {int(arch)}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-2, 102)
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.legend(loc='best', fontsize=8)
        
        # Add best point annotations
        best_pd = arch_data.loc[arch_data['Saved-mcc'].idxmax()]
        best_er = arch_data.loc[arch_data['mcc_recomputed_ErenoDATA'].idxmax()]
        
        ax.plot(best_pd['sparsity'] * 100, best_pd['Saved-mcc'], 
                'ro', markersize=8, zorder=5)
        ax.plot(best_er['sparsity'] * 100, best_er['mcc_recomputed_ErenoDATA'], 
                'ro', markersize=8, zorder=5)

plt.suptitle('MCC vs Pruning Sparsity: Powerduck vs Ereno Dataset by Architecture', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('mcc_comparison_by_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create a combined plot where we can see all architectures together
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['o', 's', '^', 'D']

for idx, arch in enumerate(architectures):
    arch_data = df_filtered[df_filtered['architecture'] == arch].sort_values('sparsity')
    
    # Plot Powerduck (solid lines)
    ax.plot(arch_data['sparsity'] * 100, arch_data['Saved-mcc'], 
            marker=markers[idx], linestyle='-', linewidth=2, markersize=5,
            label=f'Powerduck - Arch {int(arch)}', color=colors[idx], alpha=0.8)
    
    # Plot Ereno (dashed lines)
    ax.plot(arch_data['sparsity'] * 100, arch_data['mcc_recomputed_ErenoDATA'], 
            marker=markers[idx], linestyle='--', linewidth=2, markersize=5,
            label=f'Ereno - Arch {int(arch)}', color=colors[idx], alpha=0.8)

ax.set_xlabel('Sparsity (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MCC', fontsize=14, fontweight='bold')
ax.set_title('MCC vs Sparsity: All Architectures Comparison', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-2, 102)
ax.set_ylim(-0.2, 1.0)
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Good performance (0.7)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Acceptable (0.5)')
ax.legend(loc='best', ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig('mcc_comparison_all_architectures.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics by architecture
print("\n" + "=" * 100)
print("PERFORMANCE SUMMARY BY ARCHITECTURE")
print("=" * 100)

for arch in architectures:
    arch_data = df_filtered[df_filtered['architecture'] == arch]
    print(f"\n{'='*50}")
    print(f"ARCHITECTURE {int(arch)}")
    print(f"{'='*50}")
    
    best_pd = arch_data.loc[arch_data['Saved-mcc'].idxmax()]
    best_er = arch_data.loc[arch_data['mcc_recomputed_ErenoDATA'].idxmax()]
    
    print(f"\nPowerduck Dataset:")
    print(f"  Best MCC: {best_pd['Saved-mcc']:.4f} at {best_pd['sparsity']*100:.1f}% sparsity")
    print(f"  Average MCC: {arch_data['Saved-mcc'].mean():.4f}")
    print(f"  Maintains >0.7 up to: {arch_data[arch_data['Saved-mcc']>0.7]['sparsity'].max()*100 if len(arch_data[arch_data['Saved-mcc']>0.7])>0 else 0:.0f}%")
    
    print(f"\nEreno Dataset:")
    print(f"  Best MCC: {best_er['mcc_recomputed_ErenoDATA']:.4f} at {best_er['sparsity']*100:.1f}% sparsity")
    print(f"  Average MCC: {arch_data['mcc_recomputed_ErenoDATA'].mean():.4f}")
    print(f"  Maintains >0.7 up to: {arch_data[arch_data['mcc_recomputed_ErenoDATA']>0.7]['sparsity'].max()*100 if len(arch_data[arch_data['mcc_recomputed_ErenoDATA']>0.7])>0 else 0:.0f}%")
    
    # Which dataset performs better for this architecture?
    if best_pd['Saved-mcc'] > best_er['mcc_recomputed_ErenoDATA']:
        print(f"\n→ Powerduck performs better on Architecture {int(arch)}")
    else:
        print(f"\n→ Ereno performs better on Architecture {int(arch)}")