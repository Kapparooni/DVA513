from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# CSV FILES
# ==========================================

csv_files = [
    ("6 Features", INPUT_DIR / "threshold_selected_rows_6.csv"),
    ("9 Features", INPUT_DIR / "threshold_selected_rows_9.csv"),
    ("13 Features", INPUT_DIR / "threshold_selected_rows_13.csv"),
]

# ==========================================
# LOAD + COMBINE
# ==========================================

all_data = []

for feature_label, file_path in csv_files:

    df = pd.read_csv(file_path)

    df["feature_set"] = feature_label

    # architecture extraction
    df["architecture"] = (
        df["model"]
        .astype(str)
        .str.extract(r"(Architecture\d)")
    )

    # numeric conversion
    for col in [
        "Saved-mcc",
        "Baseline_MCC",
        "actual_sparsity_saved"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    all_data.append(df)

# merge everything
df = pd.concat(all_data, ignore_index=True)

# ==========================================
# SORTING
# ==========================================

feature_order = [
    "6 Features",
    "9 Features",
    "13 Features"
]

architecture_order = [
    "Architecture1",
    "Architecture2",
    "Architecture3",
    "Architecture4"
]

# ==========================================
# PLOTTING
# ==========================================

fig, ax = plt.subplots(figsize=(18, 8))

bar_width = 0.35

x_positions = []
x_labels = []

current_x = 0

for arch in architecture_order:

    arch_df = df[
        df["architecture"] == arch
    ]

    for feature in feature_order:

        row = arch_df[
            arch_df["feature_set"] == feature
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        baseline = row["Baseline_MCC"]
        pruned = row["Saved-mcc"]
        sparsity = row["actual_sparsity_saved"]

        # positions
        x_base = current_x
        x_pruned = current_x + bar_width

        # bars
        ax.bar(
            x_base,
            baseline,
            width=bar_width,
            label="Baseline MCC" if current_x == 0 else ""
        )

        ax.bar(
            x_pruned,
            pruned,
            width=bar_width,
            label="Pruned MCC" if current_x == 0 else ""
        )

        # ==================================
        # MCC VALUES ON TOP
        # ==================================

        ax.text(
            x_base,
            baseline + 0.01,
            f"{baseline:.3f}",
            ha="center",
            fontsize=9
        )

        ax.text(
            x_pruned,
            pruned + 0.01,
            f"{pruned:.3f}",
            ha="center",
            fontsize=9
        )

        # ==================================
        # SPARSITY UNDER PRUNED BAR
        # ==================================

        ax.text(
            x_pruned,
            -0.05,
            f"S={sparsity:.2f}",
            ha="center",
            fontsize=8,
            rotation=90
        )

        # center tick
        x_positions.append(
            current_x + bar_width / 2
        )

        x_labels.append(
            f"{arch}\n{feature}"
        )

        current_x += 1.2

    # spacing between architectures
    current_x += 0.8

# ==========================================
# FINAL STYLING
# ==========================================

ax.set_xticks(x_positions)
ax.set_xticklabels(
    x_labels,
    rotation=0
)

ax.set_ylabel("MCC")
ax.set_title(
    "Baseline vs Pruned MCC Across Architectures and Feature Sets"
)

ax.set_ylim(0, 1.05)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.legend()

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

output_path = (
    OUTPUT_DIR /
    "mcc_comparison_plot.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved plot to:\n{output_path}")