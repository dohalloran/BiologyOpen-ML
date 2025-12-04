############################################################
# check_supplementaries.R  (RDKit-version, no stringr/readr)
############################################################

library(dplyr)

cat("=== Loading files ===\n")

# Active / inactive lists with metadata
active_list   <- read.csv("active/active_synonyms.csv",
                          stringsAsFactors = FALSE, check.names = FALSE)
inactive_list <- read.csv("inactive/inactive_synonyms.csv",
                          stringsAsFactors = FALSE, check.names = FALSE)

# RDKit descriptor-only files (original RDKit runs)
active_rdkit   <- read.csv("active/active_rdkit.csv",
                           stringsAsFactors = FALSE, check.names = FALSE)
inactive_rdkit <- read.csv("inactive/inactive_rdkit.csv",
                           stringsAsFactors = FALSE, check.names = FALSE)

# Final supplementary tables
S1 <- read.csv("Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv",
               stringsAsFactors = FALSE, check.names = FALSE)
S2 <- read.csv("Supplementary_Table_S2_RDKit_active_descriptors.csv",
               stringsAsFactors = FALSE, check.names = FALSE)
S3 <- read.csv("Supplementary_Table_S3_RDKit_inactive_descriptors.csv",
               stringsAsFactors = FALSE, check.names = FALSE)

cat("Files loaded.\n\n")

check_equal <- function(label, a, b) {
  ok <- identical(as.integer(a), as.integer(b))
  cat(sprintf("%-50s : %s (%d vs %d)\n",
              label, if (ok) "OK" else "MISMATCH", a, b))
  invisible(ok)
}

############################################################
# Basic row counts
############################################################

cat("=== Basic row counts ===\n")

n_active_list    <- nrow(active_list)
n_inactive_list  <- nrow(inactive_list)
n_active_rdkit   <- nrow(active_rdkit)
n_inactive_rdkit <- nrow(inactive_rdkit)

n_S1_active      <- sum(S1$set_label == "Active", na.rm = TRUE)
n_S1_inactive    <- sum(S1$set_label == "Inactive", na.rm = TRUE)
n_S2             <- nrow(S2)
n_S3             <- nrow(S3)

check_equal("Actives: list vs S1 (Active rows)",       n_active_list,   n_S1_active)
check_equal("Actives: list vs active_rdkit rows",      n_active_list,   n_active_rdkit)
check_equal("Actives: S2 vs active_rdkit rows",        n_S2,            n_active_rdkit)

cat("\n")
check_equal("Inactives: list vs S1 (Inactive rows)",   n_inactive_list, n_S1_inactive)
check_equal("Inactives: list vs inactive_rdkit rows",  n_inactive_list, n_inactive_rdkit)
check_equal("Inactives: S3 vs inactive_rdkit rows",    n_S3,            n_inactive_rdkit)

############################################################
# RDKit_ID patterns and mapping
############################################################

cat("\n=== RDKit_ID patterns and mapping ===\n")

active_prefix   <- "RDKit_active_"
inactive_prefix <- "RDKit_inactive_"

S1_active   <- S1 %>% filter(set_label == "Active")
S1_inactive <- S1 %>% filter(set_label == "Inactive")

get_suffix <- function(x, prefix) {
  as.integer(sub(prefix, "", x, fixed = TRUE))
}

active_suffixes   <- get_suffix(S1_active$RDKit_ID,   active_prefix)
inactive_suffixes <- get_suffix(S1_inactive$RDKit_ID, inactive_prefix)

cat("Active RDKit_ID prefix OK?  ",
    all(startsWith(as.character(S1_active$RDKit_ID), active_prefix)),   "\n")
cat("Inactive RDKit_ID prefix OK?",
    all(startsWith(as.character(S1_inactive$RDKit_ID), inactive_prefix)),"\n")

cat("Active numeric IDs sequential from 1..N?  ",
    setequal(active_suffixes,   seq_len(length(active_suffixes))),   "\n")
cat("Inactive numeric IDs sequential from 1..N? ",
    setequal(inactive_suffixes, seq_len(length(inactive_suffixes))), "\n\n")

cat("=== Consistency of RDKit_ID between S1 and S2/S3 ===\n")

S2_ids <- unique(S2$RDKit_ID)
S3_ids <- unique(S3$RDKit_ID)

S1_active_ids   <- unique(S1_active$RDKit_ID)
S1_inactive_ids <- unique(S1_inactive$RDKit_ID)

cat("All RDKit_ID in S2 match Active entries in S1?   ",
    setequal(S2_ids, S1_active_ids),   "\n")
cat("All RDKit_ID in S3 match Inactive entries in S1? ",
    setequal(S3_ids, S1_inactive_ids), "\n\n")

############################################################
# Metadata completeness
############################################################

cat("=== Metadata completeness in S2/S3 ===\n")

meta_cols <- c("CID", "PreferredName", "IUPACName",
               "PrimarySynonym", "Synonyms", "SMILES")

meta_summary <- function(df, label) {
  cat(">>", label, "\n")
  for (col in meta_cols) {
    if (col %in% names(df)) {
      n_na   <- sum(is.na(df[[col]]) | df[[col]] == "" )
      n_tot  <- nrow(df)
      pct    <- round(100 * n_na / max(1, n_tot), 1)
      cat(sprintf("   %-15s : %4d NA/empty out of %4d (%4.1f%%)\n",
                  col, n_na, n_tot, pct))
    } else {
      cat(sprintf("   %-15s : (column not present)\n", col))
    }
  }
  cat("\n")
}

meta_summary(S2, "S2 (actives + RDKit descriptors)")
meta_summary(S3, "S3 (inactives + RDKit descriptors)")

############################################################
# Sample rows for manual inspection
############################################################

cat("=== Sample rows for manual inspection ===\n")

set.seed(123)

cat("\n--- Random 3 S1 Active entries ---\n")
idx_act <- sample(seq_len(nrow(S1_active)), size = min(3, nrow(S1_active)))
print(S1_active[idx_act,
                c("set_label", "RDKit_ID", "CID", "PreferredName", "SMILES")])

cat("\n--- Corresponding rows in S2 (if present) ---\n")
ids_act <- S1_active$RDKit_ID[idx_act]
print(
  S2[S2$RDKit_ID %in% ids_act,
     c("set_label", "RDKit_ID", "CID", "PreferredName", "SMILES",
       "MolWt", "MolLogP")]
)

cat("\n--- Random 3 S1 Inactive entries ---\n")
idx_inact <- sample(seq_len(nrow(S1_inactive)), size = min(3, nrow(S1_inactive)))
print(S1_inactive[idx_inact,
                  c("set_label", "RDKit_ID", "CID", "PreferredName", "SMILES")])

cat("\n--- Corresponding rows in S3 (if present) ---\n")
ids_inact <- S1_inactive$RDKit_ID[idx_inact]
print(
  S3[S3$RDKit_ID %in% ids_inact,
     c("set_label", "RDKit_ID", "CID", "PreferredName", "SMILES",
       "MolWt", "MolLogP")]
)

cat("\n=== DONE ===\n")
