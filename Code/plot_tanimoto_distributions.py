#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def load_matrix(path):
    print(f"Loading matrix: {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"  shape = {df.shape}")
    return df

def upper_triangle_values(df, include_diagonal=False):
    """
    Return the upper triangle of a square matrix as a 1D array.
    If include_diagonal=False, the diagonal is excluded.
    """
    arr = df.values.astype(float)
    n_rows, n_cols = arr.shape
    if n_rows != n_cols:
        raise ValueError("Matrix is not square, cannot take upper triangle for within-set pairs.")
    k = 0 if include_diagonal else 1
    tri_idx = np.triu_indices(n_rows, k=k)
    return arr[tri_idx]

def flatten_values(df):
    """Return all values of a rectangular matrix as a 1D array."""
    return df.values.astype(float).ravel()

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    base_dir = Path(__file__).resolve().parent.parent  # ~/Desktop/Revision
    results_dir = base_dir / "Code" / "tanimoto_results"

    aa_path = results_dir / "tanimoto_active_vs_active.csv"
    ii_path = results_dir / "tanimoto_inactive_vs_inactive.csv"
    ai_path = results_dir / "tanimoto_active_vs_inactive.csv"

    # Load matrices
    aa_df = load_matrix(aa_path)
    ii_df = load_matrix(ii_path)
    ai_df = load_matrix(ai_path)

    # Extract distributions
    tan_aa = upper_triangle_values(aa_df, include_diagonal=False)
    tan_ii = upper_triangle_values(ii_df, include_diagonal=False)
    tan_ai = flatten_values(ai_df)

    print(f"\n#values AA: {len(tan_aa)}")
    print(f"#values II: {len(tan_ii)}")
    print(f"#values AI: {len(tan_ai)}\n")

    # Build a long-form DataFrame for easier plotting if needed
    data = pd.DataFrame({
        "tanimoto": np.concatenate([tan_aa, tan_ii, tan_ai]),
        "pair_type": (["Active–Active"] * len(tan_aa)
                      + ["Inactive–Inactive"] * len(tan_ii)
                      + ["Active–Inactive"] * len(tan_ai))
    })

    # -------------------------------------------------------
    # 1) Overlaid histogram
    # -------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, 1.0, 41)  # 0–1 in steps of 0.025

    plt.hist(tan_aa, bins=bins, alpha=0.5, density=True,
             label="Active–Active", edgecolor="none")
    plt.hist(tan_ii, bins=bins, alpha=0.5, density=True,
             label="Inactive–Inactive", edgecolor="none")
    plt.hist(tan_ai, bins=bins, alpha=0.5, density=True,
             label="Active–Inactive", edgecolor="none")

    plt.xlabel("Tanimoto similarity")
    plt.ylabel("Density")
    plt.title("Tanimoto distributions for active/active, inactive/inactive, active/inactive")
    plt.legend()
    plt.tight_layout()

    out1 = results_dir / "tanimoto_hist_overlaid.svg"
    plt.savefig(out1, format="svg")
    plt.close()
    print(f"Saved overlaid histogram to: {out1}")

    # -------------------------------------------------------
    # 2) 3-panel histogram (one per pair type)
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    axes[0].hist(tan_aa, bins=bins, density=True, edgecolor="black")
    axes[0].set_title("Active–Active")
    axes[0].set_xlabel("Tanimoto")
    axes[0].set_ylabel("Density")

    axes[1].hist(tan_ii, bins=bins, density=True, edgecolor="black")
    axes[1].set_title("Inactive–Inactive")
    axes[1].set_xlabel("Tanimoto")

    axes[2].hist(tan_ai, bins=bins, density=True, edgecolor="black")
    axes[2].set_title("Active–Inactive")
    axes[2].set_xlabel("Tanimoto")

    fig.suptitle("Tanimoto similarity histograms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out2 = results_dir / "tanimoto_hist_three_panel.svg"
    plt.savefig(out2, format="svg")
    plt.close()
    print(f"Saved 3-panel histogram to: {out2}")

    # -------------------------------------------------------
    # 3) Violin + jitter plot
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    group_labels = ["Active–Active", "Inactive–Inactive", "Active–Inactive"]
    data_groups = [
        data.loc[data["pair_type"] == "Active–Active", "tanimoto"].values,
        data.loc[data["pair_type"] == "Inactive–Inactive", "tanimoto"].values,
        data.loc[data["pair_type"] == "Active–Inactive", "tanimoto"].values,
    ]

    positions = [1, 2, 3]

    # Violin plot
    vparts = ax.violinplot(data_groups,
                           positions=positions,
                           showmeans=True,
                           showextrema=False,
                           widths=0.7)

    # Optional: slightly adjust violin face alpha
    for vp in vparts['bodies']:
        vp.set_alpha(0.4)

    # Jittered points on top
    rng = np.random.default_rng(42)
    for i, y in enumerate(data_groups, start=1):
        # Subsample if you want to avoid crazy dense plots:
        # here, we randomly take up to 2000 points from each group
        if len(y) > 2000:
            y_plot = rng.choice(y, size=2000, replace=False)
        else:
            y_plot = y

        x_jitter = rng.normal(loc=i, scale=0.04, size=len(y_plot))
        ax.scatter(x_jitter, y_plot, s=5, alpha=0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=20)
    ax.set_ylabel("Tanimoto similarity")
    ax.set_title("Tanimoto similarity: violin + jitter")

    plt.tight_layout()
    out3 = results_dir / "tanimoto_violin_jitter.svg"
    plt.savefig(out3, format="svg")
    plt.close()
    print(f"Saved violin+jitter plot to: {out3}")

    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
