#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# List of RDKit descriptor names and functions
DESC_LIST = list(Descriptors._descList)
DESC_NAMES = [name for name, func in DESC_LIST]
DESC_FUNCS = {name: func for name, func in DESC_LIST}


def calc_desc_for_smiles(smiles: str):
    """Return dict of descriptor_name -> value (NaN if SMILES fails)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: math.nan for name in DESC_NAMES}
    return {name: func(mol) for name, func in DESC_LIST}


def build_rdkit_table(meta_path: str, out_path: str, id_col: str = "DrugBank ID"):
    """
    Read a metadata CSV containing at least SMILES and DrugBank ID,
    compute RDKit descriptors, and write them to out_path.

    The output CSV will have the DrugBank ID as the FIRST column,
    followed by the RDKit descriptors.
    """
    print(f"Reading metadata from: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # Check required columns
    if "SMILES" not in df_meta.columns:
        raise ValueError(f"'SMILES' column not found in {meta_path}")
    if id_col not in df_meta.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in {meta_path}. "
            "Please update id_col in build_rdkit_table() to match your file."
        )

    smiles_list = df_meta["SMILES"].tolist()
    print(f"  Found {len(smiles_list)} rows")

    desc_rows = []
    for i, smi in enumerate(smiles_list, start=1):
        if pd.isna(smi) or str(smi).strip() == "":
            print(f"  Row {i}: empty SMILES -> all NaN descriptors")
            desc_rows.append({name: math.nan for name in DESC_NAMES})
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            print(f"  Row {i}: RDKit could not parse SMILES '{smi}' -> all NaN")
            desc_rows.append({name: math.nan for name in DESC_NAMES})
        else:
            desc_rows.append({name: func(mol) for name, func in DESC_LIST})

    # Build descriptor DataFrame
    rdkit_df = pd.DataFrame(desc_rows, columns=DESC_NAMES)

    # Insert DrugBank ID as the first column
    rdkit_df.insert(0, id_col, df_meta[id_col].values)

    print(f"  Writing RDKit descriptors to: {out_path}")
    rdkit_df.to_csv(out_path, index=False)
    print(f"  Done. Wrote {len(rdkit_df)} rows.\n")


if __name__ == "__main__":
    # Run for DrugBank approved set
    build_rdkit_table(
        meta_path="DrugBank_approved.csv",
        out_path="DrugBank_approved_rdkit.csv",
        id_col="DrugBank ID"  # change if your ID column name is different
    )

    print("All RDKit descriptor tables rebuilt.")
