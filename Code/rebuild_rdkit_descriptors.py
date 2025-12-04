#!/usr/bin/env python

import os
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

def build_rdkit_table(meta_path: str, out_path: str):
    print(f"Reading metadata from: {meta_path}")
    df_meta = pd.read_csv(meta_path)
    if "SMILES" not in df_meta.columns:
        raise ValueError(f"'SMILES' column not found in {meta_path}")

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

    rdkit_df = pd.DataFrame(desc_rows, columns=DESC_NAMES)
    print(f"  Writing RDKit descriptors to: {out_path}")
    rdkit_df.to_csv(out_path, index=False)
    print(f"  Done. Wrote {len(rdkit_df)} rows.\n")

if __name__ == "__main__":
    # Run for active
    build_rdkit_table(
        meta_path="active/active_synonyms.csv",
        out_path="active/active_rdkit.csv"
    )

    # Run for inactive
    build_rdkit_table(
        meta_path="inactive/inactive_synonyms.csv",
        out_path="inactive/inactive_rdkit.csv"
    )

    print("All RDKit descriptor tables rebuilt.")
