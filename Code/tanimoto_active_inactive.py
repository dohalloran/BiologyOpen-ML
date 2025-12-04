#!/usr/bin/env python

"""
Compute Tanimoto similarity between active and inactive datasets.

Assumes folder structure:

Revision/
  Code/
    tanimoto_active_inactive.py  (this file)
  active/
    active_synonyms.csv
  inactive/
    inactive_synonyms.csv

Each CSV is assumed to have at least columns:
  - SMILES
  - CID
"""

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]  # .../Revision
ACTIVE_CSV = BASE / "active" / "active_synonyms.csv"
INACTIVE_CSV = BASE / "inactive" / "inactive_synonyms.csv"
OUT_DIR = BASE / "Code" / "tanimoto_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_smiles_table(path, smiles_col="SMILES", id_col="CID"):
    print(f"\nLoading: {path}")
    df = pd.read_csv(path)

    # Be a bit defensive about column names
    cols = {c.lower(): c for c in df.columns}
    if smiles_col.lower() not in cols:
        raise ValueError(f"Could not find a '{smiles_col}' column in {path}. "
                         f"Available columns: {list(df.columns)}")
    if id_col.lower() not in cols:
        raise ValueError(f"Could not find a '{id_col}' column in {path}. "
                         f"Available columns: {list(df.columns)}")

    smiles_col_real = cols[smiles_col.lower()]
    id_col_real = cols[id_col.lower()]

    df = df[[smiles_col_real, id_col_real]].copy()
    df.rename(columns={smiles_col_real: "SMILES", id_col_real: "CID"}, inplace=True)

    print(f"  Loaded {len(df)} rows from {path.name}")
    return df


def smiles_to_mols(df):
    mols = []
    keep_idx = []

    for i, row in df.iterrows():
        smi = str(row["SMILES"]).strip()
        if not smi or smi.lower() == "nan":
            print(f"  Skipping empty SMILES at row {i}")
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  !! Failed to parse SMILES at row {i}: {smi}")
            continue
        mols.append(mol)
        keep_idx.append(i)

    df_ok = df.loc[keep_idx].reset_index(drop=True)
    print(f"  Parsed {len(mols)} valid molecules (out of {len(df)})")
    return df_ok, mols


def mols_to_morgan_fps(mols, radius=2, nbits=2048):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
           for m in mols]
    return fps


def tanimoto_matrix(fps_a, fps_b):
    """Return numpy array [len(a) x len(b)] of Tanimoto similarities."""
    n_a = len(fps_a)
    n_b = len(fps_b)
    mat = np.zeros((n_a, n_b), dtype=float)

    for i, fa in enumerate(fps_a):
        # BulkTanimotoSimilarity compares fa to each fingerprint in fps_b
        sims = DataStructs.BulkTanimotoSimilarity(fa, fps_b)
        mat[i, :] = sims

    return mat


def summarize_matrix(mat, row_ids, col_ids, label):
    print(f"\n=== {label} Tanimoto summary ===")
    print(f"  Shape: {mat.shape[0]} x {mat.shape[1]}")
    print(f"  Overall mean   : {mat.mean():.3f}")
    print(f"  Overall median : {np.median(mat):.3f}")
    print(f"  Overall min    : {mat.min():.3f}")
    print(f"  Overall max    : {mat.max():.3f}")

    # For each row, max similarity
    max_per_row = mat.max(axis=1)
    print(f"  Mean of per-{label.split()[0].lower()} maxima: {max_per_row.mean():.3f}")
    print(f"  Median of per-{label.split()[0].lower()} maxima: {np.median(max_per_row):.3f}")

    # Top 10 pairs
    flat = mat.ravel()
    top_n = min(10, flat.size)
    top_idx = np.argpartition(-flat, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-flat[top_idx])]  # sort descending

    print(f"\n  Top {top_n} highest pairs:")
    for k, idx in enumerate(top_idx, start=1):
        i, j = divmod(idx, mat.shape[1])
        print(f"    #{k:2d}: {row_ids[i]}  vs  {col_ids[j]}   Tanimoto = {mat[i, j]:.3f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Load data
    act_df_raw = load_smiles_table(ACTIVE_CSV)
    inact_df_raw = load_smiles_table(INACTIVE_CSV)

    # SMILES -> RDKit molecules
    act_df, act_mols = smiles_to_mols(act_df_raw)
    inact_df, inact_mols = smiles_to_mols(inact_df_raw)

    act_ids = act_df["CID"].astype(str).tolist()
    inact_ids = inact_df["CID"].astype(str).tolist()

    # Fingerprints
    print("\nGenerating Morgan fingerprints (radius=2, nBits=2048)...")
    act_fps = mols_to_morgan_fps(act_mols, radius=2, nbits=2048)
    inact_fps = mols_to_morgan_fps(inact_mols, radius=2, nbits=2048)

    # -----------------------------------------------------------------
    # Active vs inactive
    # -----------------------------------------------------------------
    print("\nComputing active vs inactive Tanimoto matrix...")
    avs_mat = tanimoto_matrix(act_fps, inact_fps)

    avs_df = pd.DataFrame(avs_mat, index=act_ids, columns=inact_ids)
    out_avs = OUT_DIR / "tanimoto_active_vs_inactive.csv"
    avs_df.to_csv(out_avs)
    print(f"Active vs inactive matrix written to: {out_avs}")

    summarize_matrix(avs_mat, act_ids, inact_ids, "Active vs Inactive")

    # -----------------------------------------------------------------
    # Active vs active (self-similarity, excluding diagonal)
    # -----------------------------------------------------------------
    print("\nComputing active vs active Tanimoto matrix...")
    aaa_mat = tanimoto_matrix(act_fps, act_fps)
    np.fill_diagonal(aaa_mat, np.nan)  # ignore self = 1.0 for summary

    out_aaa = OUT_DIR / "tanimoto_active_vs_active.csv"
    pd.DataFrame(aaa_mat, index=act_ids, columns=act_ids).to_csv(out_aaa)
    print(f"Active vs active matrix written to: {out_aaa}")

    summarize_matrix(np.nan_to_num(aaa_mat, nan=0.0), act_ids, act_ids, "Active vs Active")

    # -----------------------------------------------------------------
    # Inactive vs inactive (self-similarity, excluding diagonal)
    # -----------------------------------------------------------------
    print("\nComputing inactive vs inactive Tanimoto matrix...")
    iii_mat = tanimoto_matrix(inact_fps, inact_fps)
    np.fill_diagonal(iii_mat, np.nan)

    out_iii = OUT_DIR / "tanimoto_inactive_vs_inactive.csv"
    pd.DataFrame(iii_mat, index=inact_ids, columns=inact_ids).to_csv(out_iii)
    print(f"Inactive vs inactive matrix written to: {out_iii}")

    summarize_matrix(np.nan_to_num(iii_mat, nan=0.0), inact_ids, inact_ids, "Inactive vs Inactive")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
