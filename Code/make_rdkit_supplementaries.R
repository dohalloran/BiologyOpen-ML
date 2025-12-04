############################################################
# make_rdkit_supplementaries.R
#
# Run from the Revision directory:
#   cd ~/Desktop/Revision
#   Rscript Code/make_rdkit_supplementaries.R
#
# Inputs:
#   active/active_synonyms.csv
#   inactive/inactive_synonyms.csv
#   active/active_rdkit.csv
#   inactive/inactive_rdkit.csv
#
# Outputs:
#   Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv
#   Supplementary_Table_S2_RDKit_active_descriptors.csv
#   Supplementary_Table_S3_RDKit_inactive_descriptors.csv
############################################################

library(dplyr)

cat("=== Loading RDKit input files ===\n")

# 1) Metadata / synonyms
active_list <- read.csv(
  "active/active_synonyms.csv",
  stringsAsFactors = FALSE, check.names = FALSE
)
inactive_list <- read.csv(
  "inactive/inactive_synonyms.csv",
  stringsAsFactors = FALSE, check.names = FALSE
)

# 2) RDKit descriptors
active_rdkit <- read.csv(
  "active/active_rdkit.csv",
  stringsAsFactors = FALSE, check.names = FALSE
)
inactive_rdkit <- read.csv(
  "inactive/inactive_rdkit.csv",
  stringsAsFactors = FALSE, check.names = FALSE
)

cat("Files loaded.\n\n")

# --- sanity: row counts must match ---
if (nrow(active_list) != nrow(active_rdkit)) {
  stop("Active list vs RDKit rows mismatch: ",
       nrow(active_list), " vs ", nrow(active_rdkit))
}
if (nrow(inactive_list) != nrow(inactive_rdkit)) {
  stop("Inactive list vs RDKit rows mismatch: ",
       nrow(inactive_list), " vs ", nrow(inactive_rdkit))
}

# --- check required metadata columns ---
needed_cols <- c("SMILES", "CID", "PreferredName",
                 "IUPACName", "PrimarySynonym", "Synonyms")

for (c in needed_cols) {
  if (!c %in% names(active_list)) {
    stop("Column '", c, "' not found in active metadata file.")
  }
  if (!c %in% names(inactive_list)) {
    stop("Column '", c, "' not found in inactive metadata file.")
  }
}

# --- construct RDKit IDs based on row order ---
cat("Constructing RDKit_IDs...\n")

active_ids   <- paste0("RDKit_active_",   seq_len(nrow(active_list)))
inactive_ids <- paste0("RDKit_inactive_", seq_len(nrow(inactive_list)))

# ==========================================================
# S1: labeled set (metadata only, RDKit-based)
# ==========================================================
cat("Building RDKit S1 (labeled dataset)...\n")

active_s1 <- data.frame(
  set_label      = "Active",
  RDKit_ID       = active_ids,
  CID            = active_list$CID,
  PreferredName  = active_list$PreferredName,
  IUPACName      = active_list$IUPACName,
  PrimarySynonym = active_list$PrimarySynonym,
  Synonyms       = active_list$Synonyms,
  SMILES         = active_list$SMILES,
  source         = "Known_anthelmintic",
  stringsAsFactors = FALSE
)

inactive_s1 <- data.frame(
  set_label      = "Inactive",
  RDKit_ID       = inactive_ids,
  CID            = inactive_list$CID,
  PreferredName  = inactive_list$PreferredName,
  IUPACName      = inactive_list$IUPACName,
  PrimarySynonym = inactive_list$PrimarySynonym,
  Synonyms       = inactive_list$Synonyms,
  SMILES         = inactive_list$SMILES,
  source         = "Non_anthelmintic",
  stringsAsFactors = FALSE
)

S1_rdkit <- bind_rows(active_s1, inactive_s1) %>%
  arrange(desc(set_label), PreferredName)

write.csv(
  S1_rdkit,
  "Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv",
  row.names = FALSE
)

cat("  -> wrote Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv\n\n")

# ==========================================================
# S2: active RDKit descriptors + metadata
# ==========================================================
cat("Building RDKit S2 (active descriptors)...\n")

# RDKit descriptor table is all descriptors for actives
# (no CID/SMILES columns, based on your header)
S2_rdkit <- cbind(
  data.frame(
    set_label      = "Active",
    RDKit_ID       = active_ids,
    CID            = active_list$CID,
    PreferredName  = active_list$PreferredName,
    IUPACName      = active_list$IUPACName,
    PrimarySynonym = active_list$PrimarySynonym,
    Synonyms       = active_list$Synonyms,
    SMILES         = active_list$SMILES,
    stringsAsFactors = FALSE
  ),
  active_rdkit
)

write.csv(
  S2_rdkit,
  "Supplementary_Table_S2_RDKit_active_descriptors.csv",
  row.names = FALSE
)

cat("  -> wrote Supplementary_Table_S2_RDKit_active_descriptors.csv\n\n")

# ==========================================================
# S3: inactive RDKit descriptors + metadata
# ==========================================================
cat("Building RDKit S3 (inactive descriptors)...\n")

S3_rdkit <- cbind(
  data.frame(
    set_label      = "Inactive",
    RDKit_ID       = inactive_ids,
    CID            = inactive_list$CID,
    PreferredName  = inactive_list$PreferredName,
    IUPACName      = inactive_list$IUPACName,
    PrimarySynonym = inactive_list$PrimarySynonym,
    Synonyms       = inactive_list$Synonyms,
    SMILES         = inactive_list$SMILES,
    stringsAsFactors = FALSE
  ),
  inactive_rdkit
)

write.csv(
  S3_rdkit,
  "Supplementary_Table_S3_RDKit_inactive_descriptors.csv",
  row.names = FALSE
)

cat("  -> wrote Supplementary_Table_S3_RDKit_inactive_descriptors.csv\n\n")
cat("=== DONE: RDKit-based supplementary tables generated ===\n")
