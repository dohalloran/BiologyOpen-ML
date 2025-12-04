import pubchempy as pcp
import csv

input_file = "inactive.smi"
output_file = "inactive_synonyms.csv"

rows = []

with open(input_file) as f:
    for line in f:
        smi = line.strip()
        if not smi:
            continue

        cid = ""
        preferred_name = ""
        iupac_name = ""
        primary_synonym = ""
        all_synonyms = ""

        try:
            compounds = pcp.get_compounds(smi, namespace="smiles")
            if compounds:
                c = compounds[0]

                # CID
                cid = c.cid

                # IUPAC name
                iupac_name = c.iupac_name or ""

                # Synonyms list (may be None)
                syns = c.synonyms or []

                # Primary synonym (first one if exists)
                primary_synonym = syns[0] if syns else ""

                # All synonyms (limit to first 10 to keep file sane)
                if syns:
                    all_synonyms = "; ".join(syns[:10])

                # Preferred name:
                #  - Prefer IUPAC name if available
                #  - Else fall back to primary synonym
                if iupac_name:
                    preferred_name = iupac_name
                elif primary_synonym:
                    preferred_name = primary_synonym

        except Exception as e:
            # On any error we just leave fields empty for this SMILES
            pass

        rows.append(
            (
                smi,
                cid,
                preferred_name,
                iupac_name,
                primary_synonym,
                all_synonyms,
            )
        )

# write CSV
with open(output_file, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(
        ["SMILES", "CID", "PreferredName", "IUPACName", "PrimarySynonym", "Synonyms"]
    )
    w.writerows(rows)

print("Wrote", output_file)
