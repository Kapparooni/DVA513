import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np

# Define folders
input_dir = 'inputs'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load files
df9 = pd.read_csv(os.path.join(input_dir, 'threshold_selected_ereno9.csv'))
df13 = pd.read_csv(os.path.join(input_dir, 'threshold_selected_ereno13.csv'))
df6 = pd.read_csv(os.path.join(input_dir, 'threshold_selected_ereno6.csv'))

# Add feature count
df9['feature_count'] = 9
df13['feature_count'] = 13
df6['feature_count'] = 6

# Combine ALL data (12 models total)
df = pd.concat([df9, df13, df6], ignore_index=True)

# Extract architecture
df['architecture'] = df['model'].str.extract(r'Architecture(\d+)').astype(int)

# Calculate accuracy preserved
df['accuracy_preserved'] = df['Saved-mcc'] / df['Baseline_MCC']

# Rename columns for cleaner labels
df = df.rename(columns={
    'actual_sparsity_saved': 'Sparsity',
    'latency_seconds': 'Latency (s)',
    'Saved-mcc': 'MCC-Post',
    'Baseline_MCC': 'MCC-Base',
    'Allowed_Drop': 'Allowed Drop',
    'initial_parameters': 'Initial Params',
    'parameters_final': 'Final Params',
    'feature_count': 'Feature Count',
    'architecture': 'Architecture'
})

print("=" * 60)
print("CORRELATION MATRIX - ALL 12 MODELS (MCC Drop Removed)")
print("=" * 60)
print(f"Total models: {len(df)}")
print(f"Feature counts: {sorted(df['Feature Count'].unique())}")
print(f"Architectures: {sorted(df['Architecture'].unique())}")

# Select key numerical columns for correlation (MCC Drop REMOVED)
corr_cols = [
    'Feature Count',
    'Architecture', 
    'Sparsity',
    'Latency (s)',
    'MCC-Post',
    'MCC-Base',
    'Allowed Drop',
    'Initial Params',
    'Final Params',
    'accuracy_preserved'
]

# Compute correlation matrix
corr_matrix = df[corr_cols].corr()

# Calculate p-values
def calculate_pvalues(df, cols):
    """Calculate p-values for correlation matrix"""
    pvalues = pd.DataFrame(index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i == j:
                pvalues.loc[i, j] = 0.0
            else:
                mask = df[[i, j]].dropna()
                if len(mask) > 2:
                    _, p = stats.pearsonr(mask[i], mask[j])
                    pvalues.loc[i, j] = p
                else:
                    pvalues.loc[i, j] = 1.0
    return pvalues

pvalues = calculate_pvalues(df, corr_cols)

# Print correlation matrix
print("\n📊 CORRELATION MATRIX (n=12):")
print(corr_matrix.round(3))

# Print significance matrix
print("\n📊 P-VALUES (significant if < 0.05):")
print(pvalues.round(4))

# Save to CSV
corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix_no_MCCdrop.csv'))
pvalues.to_csv(os.path.join(output_dir, 'correlation_pvalues_no_MCCdrop.csv'))

print(f"\n✅ Saved correlation matrix to: {output_dir}/correlation_matrix_no_MCCdrop.csv")
print(f"✅ Saved p-values to: {output_dir}/correlation_pvalues_no_MCCdrop.csv")

# Create heatmap with significance stars
fig, ax = plt.subplots(figsize=(12, 10))

# Create custom annotations with stars
annot_with_stars = []
for i in range(len(corr_cols)):
    row = []
    for j in range(len(corr_cols)):
        if i >= j:  # Lower triangle including diagonal
            r = corr_matrix.iloc[i, j]
            p = pvalues.iloc[i, j]
            if i == j:
                row.append(f"{r:.2f}")
            else:
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                row.append(f"{r:.2f}{stars}")
        else:
            row.append('')  # Upper triangle empty
    annot_with_stars.append(row)

# Plot heatmap
sns.heatmap(
    corr_matrix, 
    annot=annot_with_stars, 
    fmt='',
    cmap='coolwarm', 
    center=0, 
    square=True, 
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
    annot_kws={'size': 9},
    vmin=-1, vmax=1,
    ax=ax
)

ax.set_title('Correlation Matrix (MCC Drop Removed)\n* p<0.05, ** p<0.01, *** p<0.001 | n=12', 
             fontsize=12, fontweight='bold')
ax.set_xticklabels(corr_cols, rotation=45, ha='right')
ax.set_yticklabels(corr_cols, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix_no_MCCdrop.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# HIGHLIGHT THE MOST IMPORTANT FINDINGS
# ============================================================
print("\n" + "=" * 60)
print("🔍 KEY FINDINGS (n=12, MCC Drop Removed)")
print("=" * 60)

def add_significance_stars(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return ''

# Key relationships to examine (MCC Drop removed)
relationships = [
    ('Sparsity', 'Latency (s)'),
    ('Sparsity', 'MCC-Post'),
    ('Sparsity', 'accuracy_preserved'),
    ('Feature Count', 'Sparsity'),
    ('Feature Count', 'Latency (s)'),
    ('Feature Count', 'MCC-Post'),
    ('Feature Count', 'accuracy_preserved'),
    ('Architecture', 'Latency (s)'),
    ('Architecture', 'Sparsity'),
    ('Architecture', 'MCC-Post'),
    ('MCC-Base', 'MCC-Post'),
    ('Allowed Drop', 'accuracy_preserved'),
    ('Sparsity', 'Allowed Drop')
]

print("\n📈 Correlation Analysis:")
significant_count = 0
for x, y in relationships:
    r = corr_matrix.loc[x, y]
    p = pvalues.loc[x, y]
    sig = add_significance_stars(p)
    
    if p < 0.05:
        print(f"\n✅ {x} vs {y}: r = {r:.3f} (p={p:.4f}{sig}) - SIGNIFICANT")
        significant_count += 1
    else:
        # Still print but note not significant
        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        direction = "positive" if r > 0 else "negative"
        print(f"\n   {x} vs {y}: r = {r:.3f} (p={p:.4f}) - {strength} {direction} trend (not significant)")

print("\n" + "=" * 60)
print("📊 SIGNIFICANT CORRELATIONS (p < 0.05)")
print("=" * 60)

significant_pairs = []
for i in range(len(corr_cols)):
    for j in range(i+1, len(corr_cols)):
        if pvalues.iloc[i, j] < 0.05:
            significant_pairs.append((corr_cols[i], corr_cols[j], 
                                     corr_matrix.iloc[i, j], 
                                     pvalues.iloc[i, j]))
            print(f"{corr_cols[i]} ↔ {corr_cols[j]}: r = {corr_matrix.iloc[i, j]:.3f}, p = {pvalues.iloc[i, j]:.4f}{add_significance_stars(pvalues.iloc[i, j])}")

if len(significant_pairs) == 0:
    print("No statistically significant correlations found at p < 0.05")
    print("With n=12, need |r| > 0.576 for significance")
else:
    print(f"\nFound {len(significant_pairs)} significant correlations")

# Create simplified bar chart of key correlations
fig, ax = plt.subplots(figsize=(10, 6))

key_metrics = [
    'Sparsity → Latency',
    'Sparsity → Accuracy', 
    'Sparsity → Acc Preserved',
    'Features → Sparsity',
    'Features → Latency',
    'Features → Accuracy',
    'Arch → Latency',
    'Allowed Drop → Acc Preserved'
]

key_values = [
    corr_matrix.loc['Sparsity', 'Latency (s)'],
    corr_matrix.loc['Sparsity', 'MCC-Post'],
    corr_matrix.loc['Sparsity', 'accuracy_preserved'],
    corr_matrix.loc['Feature Count', 'Sparsity'],
    corr_matrix.loc['Feature Count', 'Latency (s)'],
    corr_matrix.loc['Feature Count', 'MCC-Post'],
    corr_matrix.loc['Architecture', 'Latency (s)'],
    corr_matrix.loc['Allowed Drop', 'accuracy_preserved']
]

# Add significance stars to bar labels
key_significance = [
    add_significance_stars(pvalues.loc['Sparsity', 'Latency (s)']),
    add_significance_stars(pvalues.loc['Sparsity', 'MCC-Post']),
    add_significance_stars(pvalues.loc['Sparsity', 'accuracy_preserved']),
    add_significance_stars(pvalues.loc['Feature Count', 'Sparsity']),
    add_significance_stars(pvalues.loc['Feature Count', 'Latency (s)']),
    add_significance_stars(pvalues.loc['Feature Count', 'MCC-Post']),
    add_significance_stars(pvalues.loc['Architecture', 'Latency (s)']),
    add_significance_stars(pvalues.loc['Allowed Drop', 'accuracy_preserved'])
]

key_metrics_with_stars = [f"{m} {s}" for m, s in zip(key_metrics, key_significance)]

colors = []
for val in key_values:
    if val > 0.5:
        colors.append('darkred')
    elif val > 0.3:
        colors.append('red')
    elif val > 0:
        colors.append('lightcoral')
    elif val < -0.5:
        colors.append('darkblue')
    elif val < -0.3:
        colors.append('blue')
    else:
        colors.append('gray')

bars = ax.barh(key_metrics_with_stars, key_values, color=colors)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0.576, color='red', linestyle='--', alpha=0.5, label='Significance threshold (|r| > 0.576)')
ax.axvline(x=-0.576, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Correlation Strength')
ax.set_title('Key Correlations (MCC Drop Removed)\n* p<0.05, ** p<0.01, *** p<0.001', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'key_findings_no_MCCdrop.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"✅ All outputs saved to: {output_dir}")
print("=" * 60)
print("\nGenerated files:")
print("  📊 correlation_matrix_no_MCCdrop.png - Heatmap with significance stars")
print("  📄 correlation_matrix_no_MCCdrop.csv - Raw correlations")
print("  📄 correlation_pvalues_no_MCCdrop.csv - Statistical significance")
print("  📊 key_findings_no_MCCdrop.png - Simplified summary")

print("\n📌 INTERPRETATION GUIDE:")
print("   * = p < 0.05  (95% confidence)")
print("   ** = p < 0.01 (99% confidence)")
print("   *** = p < 0.001 (99.9% confidence)")
print("   |r| > 0.576 needed for significance with n=12")
print("\n📌 REMOVED: MCC Drop (redundant with accuracy_preserved, r = -1.00)")