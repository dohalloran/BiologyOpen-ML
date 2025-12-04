````markdown
# Anthelmintic Activity Pipeline

This repository contains a complete cheminformatics workflow that starts from two labeled datasets of compounds (active and inactive) and ends with predicted activity labels for DrugBank-approved drugs.

The final output of the pipeline is:

- `Training/DrugBank_approved_predictions_all_models.csv`

which contains model-based activity predictions (active = 1, inactive = 0) for each DrugBank compound across several machine-learning models.

---

## 1. Repository layout

From the project root (`Revision/`):

```text
Revision/
├── Code
│   ├── check_supplementaries.R
│   ├── make_rdkit_supplementaries.R
│   ├── names.py
│   ├── plot_tanimoto_distributions.py
│   ├── rebuild_rdkit_descriptors.py
│   ├── tanimoto_active_inactive.py
│   └── tanimoto_results/
│       ├── tanimoto_active_vs_active.csv
│       ├── tanimoto_active_vs_inactive.csv
│       ├── tanimoto_inactive_vs_inactive.csv
│       ├── tanimoto_hist_overlaid.{png,svg}
│       ├── tanimoto_hist_three_panel.{png,svg}
│       └── tanimoto_violin_jitter.{png,svg}
├── Drugbank
│   ├── DrugBank_approved.csv
│   ├── DrugBank_approved_rdkit.csv
│   └── rebuild_rdkit_descriptors.py
├── Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv
├── Supplementary_Table_S2_RDKit_active_descriptors.csv
├── Supplementary_Table_S3_RDKit_inactive_descriptors.csv
├── Training
│   ├── DrugBank_approved_predictions_all_models.csv
│   ├── DrugBank_approved_rdkit.csv
│   ├── combined_training.csv
│   ├── figures/
│   │   ├── activity_models_ROC_all.svg
│   │   ├── activity_models_confusion_matrices_all.svg
│   │   └── activity_models_metrics_bars_subplots.svg
│   ├── predict_drugbank_activity.py
│   └── train_activity_models.py
├── active
│   ├── active.smi
│   ├── active_rdkit.csv
│   ├── active_synonyms.csv
│   └── synonyms.py
└── inactive
    ├── inactive.smi
    ├── inactive_rdkit.csv
    ├── inactive_synonyms.csv
    └── synonyms.py
````

---

## 2. Software requirements

### Python

* Python 3.7+ (tested with Anaconda)
* RDKit (Python build)
* NumPy
* pandas
* scikit-learn
* matplotlib
* seaborn (for some plots in `plot_tanimoto_distributions.py`)

Example conda environment:

```bash
conda create -n anthelmintic python=3.9
conda activate anthelmintic

# RDKit from conda-forge
conda install -c conda-forge rdkit

# ML / plotting stack
pip install numpy pandas scikit-learn matplotlib seaborn
```

### R (for supplementary tables)

To rebuild the supplementary RDKit descriptor tables, you will also need:

* R (≥ 4.0)
* R packages: `tidyverse`, `data.table`, and RDKit bindings if used by the R scripts.

---

## 3. Starting data

The workflow begins with two labeled datasets of anthelmintic compounds:

* `active/active.smi` – SMILES for active compounds
* `inactive/inactive.smi` – SMILES for inactive compounds

Each line corresponds to a compound. Synonyms and metadata are kept in:

* `active/active_synonyms.csv`
* `inactive/inactive_synonyms.csv`

RDKit descriptors for these sets are stored as:

* `active/active_rdkit.csv`
* `inactive/inactive_rdkit.csv`

These descriptor tables have one compound per row and a large set of RDKit descriptor columns.

---

## 4. Generating RDKit descriptors

### 4.1. Supplementary tables (optional but reproducible)

The supplementary tables:

* `Supplementary_Table_S1_RDKit_labeled_anthelmintic_set.csv`
* `Supplementary_Table_S2_RDKit_active_descriptors.csv`
* `Supplementary_Table_S3_RDKit_inactive_descriptors.csv`

can be rebuilt from the RDKit descriptor data using the R scripts in `Code/`.

From the project root:

```bash
cd Code

# (Optional) check that supplementary tables match expectations
Rscript check_supplementaries.R

# Rebuild supplementary RDKit descriptor tables
Rscript make_rdkit_supplementaries.R
```

These scripts read the active/inactive descriptor files and produce the exact supplementary tables used in the manuscript.

### 4.2. Generic RDKit descriptor generation script

`Code/rebuild_rdkit_descriptors.py` shows a general Python pattern for creating RDKit descriptor tables from an input CSV.

It expects at minimum:

* a column named `SMILES`
* optionally an ID column (e.g., `DrugBankID`)

For DrugBank, a specialized version lives in `Drugbank/rebuild_rdkit_descriptors.py` (see Section 8).

---

## 5. Tanimoto similarity analysis

The next step compares the structural similarity of active and inactive compounds using Morgan fingerprints and Tanimoto coefficients.

### 5.1. Compute Tanimoto matrices

From the project root:

```bash
cd Code
python tanimoto_active_inactive.py
```

This script:

1. Loads the active and inactive datasets (SMILES + metadata).
2. Uses RDKit to compute Morgan fingerprints (typically radius = 2, nBits = 2048).
3. Computes:

   * active vs active Tanimoto matrix
   * inactive vs inactive Tanimoto matrix
   * active vs inactive Tanimoto matrix
4. Saves the matrices to CSV:

   * `Code/tanimoto_results/tanimoto_active_vs_active.csv`
   * `Code/tanimoto_results/tanimoto_inactive_vs_inactive.csv`
   * `Code/tanimoto_results/tanimoto_active_vs_inactive.csv`
5. Prints summary statistics:

   * matrix shapes
   * overall mean/median/min/max
   * mean and median of per-compound maximum similarity
   * top 10 most similar pairs in each matrix

### 5.2. Plot Tanimoto distributions and violin/jitter plots

To visualize Tanimoto distributions:

```bash
cd Code
python plot_tanimoto_distributions.py
```

This script reads the matrices in `Code/tanimoto_results/` and generates several SVG and PNG figures:

* `tanimoto_hist_three_panel.{png,svg}`
  Side-by-side histograms for:

  * active vs active
  * inactive vs inactive
  * active vs inactive

* `tanimoto_hist_overlaid.{png,svg}`
  Overlaid histograms on a single panel with different colors per group.

* `tanimoto_violin_jitter.{png,svg}`
  Violin + jitter plot comparing Tanimoto distribution of:

  * active–active pairs
  * inactive–inactive pairs
  * active–inactive cross-pairs

These plots summarize the similarity structure within and between activity classes.

---

## 6. Building the ML training dataset

To train activity prediction models, active and inactive RDKit descriptor tables are merged and labeled.

The merged file is:

* `Training/combined_training.csv`

This file has:

* One compound per row
* All RDKit descriptor columns
* A final column named `Class`:

  * `1` for active compounds
  * `0` for inactive compounds

A typical workflow to re-create `combined_training.csv` is:

1. Make sure `active_rdkit.csv` and `inactive_rdkit.csv` have the same descriptor columns.
2. Add a `Class` column with value 1 to every row in `active_rdkit.csv`.
3. Add a `Class` column with value 0 to every row in `inactive_rdkit.csv`.
4. Concatenate the two tables row-wise.
5. Save as `Training/combined_training.csv`.

This merged dataset is the input for all classification models.

---

## 7. Training activity prediction models

Model training and figure generation are handled by:

* `Training/train_activity_models.py`

### 7.1. What the script does

From the project root:

```bash
cd Training
python train_activity_models.py
```

The script performs the following:

1. **Load training data**

   * Reads `combined_training.csv`.
   * Splits into:

     * `X`: all descriptor columns
     * `y`: the `Class` column (0 = inactive, 1 = active)

2. **Handle missing/invalid values**

   * Uses `sklearn.impute.SimpleImputer` with `strategy='mean'` to replace `NaN` values in feature columns.

3. **Train/test split**

   * Uses `train_test_split` with:

     * `test_size = 0.2` (80/20 split)
     * `random_state = 0`
   * Gives:

     * `X_train`, `X_test`, `y_train`, `y_test`

4. **Feature scaling**

   * Applies `sklearn.preprocessing.StandardScaler`:

     * `sc.fit_transform(X_train)` for training data
     * `sc.transform(X_test)` for test data

5. **Models trained**

   The script trains several classification models:

   * Linear Support Vector Machine:

     * `sklearn.svm.SVC(kernel='linear', probability=True)`
   * k-Nearest Neighbors:

     * `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)`
   * RBF-kernel SVM:

     * `sklearn.svm.SVC(kernel='rbf', probability=True)`
   * Gaussian Naive Bayes:

     * `sklearn.naive_bayes.GaussianNB()`
   * Random Forest:

     * `sklearn.ensemble.RandomForestClassifier(...)`
   * Decision Tree:

     * `sklearn.tree.DecisionTreeClassifier(...)`

   (Exact hyperparameters are set in the script.)

6. **Evaluation metrics**

   For each model, the script computes:

   * Confusion matrix
   * Accuracy
   * Misclassification rate
   * Sensitivity (recall for the active class)
   * Specificity
   * Precision
   * F1 score
   * Classification report (`precision`, `recall`, `f1-score`, `support`)

7. **ROC curves**

   * Uses predicted probabilities or decision scores to compute ROC curves and AUC for each model.
   * Plots all model ROC curves on a single figure:

     * Saved as `Training/figures/activity_models_ROC_all.svg`.

8. **Combined confusion matrices figure**

   * Plots confusion matrices for all models as subplots in a single figure.
   * Uses a shared color scale and places the legend to the right to avoid overlapping the matrices.
   * Saves to:

     * `Training/figures/activity_models_confusion_matrices_all.svg`.

9. **Bar charts of metrics**

   * For each model, collects metrics such as Accuracy, Sensitivity, Specificity, Precision, F1.
   * Plots bar charts on subplots so that all models can be visually compared side-by-side:

     * Saved as `Training/figures/activity_models_metrics_bars_subplots.svg`.

These figures provide an at-a-glance comparison of how well each model discriminates active from inactive compounds.

---

## 8. Computing RDKit descriptors for DrugBank

DrugBank data are stored under:

* `Drugbank/DrugBank_approved.csv`

This input file should contain at least:

* `DrugBankID` – unique identifier
* `SMILES` – chemical structure

To compute RDKit descriptors and preserve DrugBank IDs in the first column, use:

* `Drugbank/rebuild_rdkit_descriptors.py`

From the project root:

```bash
cd Drugbank
python rebuild_rdkit_descriptors.py
```

This script:

1. Reads `DrugBank_approved.csv`.
2. Parses the `SMILES` for each row with RDKit.
3. Computes all RDKit descriptors listed in `Descriptors._descList`.
4. Writes them to:

   * `DrugBank_approved_rdkit.csv`

The output file has:

* First column: `DrugBankID`
* Remaining columns: RDKit descriptors (one row per compound)

A copy of this file is also present in:

* `Training/DrugBank_approved_rdkit.csv`

for convenience when running the prediction scripts.

---

## 9. Predicting activity for DrugBank compounds

The final step uses the trained models to predict whether each DrugBank compound is more similar to the actives (1) or inactives (0).

This is done with:

* `Training/predict_drugbank_activity.py`

### 9.1. What the prediction script does

From the project root:

```bash
cd Training
python predict_drugbank_activity.py
```

The script:

1. **Loads training data**

   * Reads `combined_training.csv` (same as in `train_activity_models.py`).
   * Splits into `X` (descriptors) and `y` (Class).
   * Applies mean imputation for missing values.
   * Scales features with `StandardScaler`.

2. **Defines the same suite of models**

   Typically includes:

   * Linear SVM
   * RBF SVM
   * kNN
   * Gaussian Naive Bayes
   * Random Forest
   * Decision Tree

3. **Cross-validation (on training set)**

   * For each model, performs k-fold cross-validation (e.g. 5-fold) on the training data.
   * Reports mean ROC AUC and standard deviation for each model.

4. **Fit models on full training set**

   * After evaluation, each model is refit on the entire `combined_training.csv` dataset (all rows) using the same preprocessing pipeline.

5. **Load DrugBank descriptor table**

   * Reads `DrugBank_approved_rdkit.csv` from `Training/` (linked copy of `Drugbank/DrugBank_approved_rdkit.csv`).
   * Extracts the descriptor columns aligned to those used for training.
   * Applies the same imputation and scaling used for the training data.

6. **Predict activity**

   For each model:

   * Predicts class labels (`0` or `1`) for all DrugBank compounds.
   * When supported, also predicts class probabilities (e.g. `P(active)`).

7. **Save results**

   All predictions are written to:

   * `Training/DrugBank_approved_predictions_all_models.csv`

   This file contains:

   * `DrugBankID`
   * For each model:

     * Predicted class label (0/1)
     * Predicted probability for the active class (if available)
   * Optionally: columns summarizing ensemble or consensus predictions (depending on script version).

This is the final product of the pipeline: a model-based prioritization of DrugBank-approved compounds for potential anthelmintic activity.

---

## 10. End-to-end step-by-step summary

From a clean clone of this repository, the typical workflow is:

1. **Set up environment**

   ```bash
   conda create -n anthelmintic python=3.9
   conda activate anthelmintic
   conda install -c conda-forge rdkit
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **(Optional) Rebuild supplementary tables**

   ```bash
   cd Code
   Rscript make_rdkit_supplementaries.R
   cd ..
   ```

3. **(Optional) Recompute Tanimoto similarity matrices and plots**

   ```bash
   cd Code
   python tanimoto_active_inactive.py
   python plot_tanimoto_distributions.py
   cd ..
   ```

4. **Ensure training dataset is ready**

   * Confirm that `Training/combined_training.csv` exists.
   * If needed, re-create it by merging `active_rdkit.csv` and `inactive_rdkit.csv` with a `Class` column (1 for active, 0 for inactive).

5. **Train activity models and generate figures**

   ```bash
   cd Training
   python train_activity_models.py
   # Outputs figures in Training/figures/
   cd ..
   ```

6. **Compute RDKit descriptors for DrugBank**

   ```bash
   cd Drugbank
   python rebuild_rdkit_descriptors.py
   cd ..
   ```

   Ensure that `DrugBank_approved_rdkit.csv` is available in the `Training` directory (copy/symlink if necessary).

7. **Predict DrugBank compound activity with all models**

   ```bash
   cd Training
   python predict_drugbank_activity.py
   cd ..
   ```

8. **Inspect final predictions**

   * Open `Training/DrugBank_approved_predictions_all_models.csv` in your preferred tool (Excel, R, pandas, etc.).
   * Investigate compounds predicted active (Class = 1) with high probability across one or more models.

---

## 11. Reproducibility notes

* Random seeds (`random_state=0`) are used where appropriate to make train/test splits and model training reproducible.
* Feature preprocessing (imputation + scaling) is fit on training data and reused consistently on test data and DrugBank data.
* Tanimoto similarity computations use the same Morgan fingerprint parameters across active and inactive sets.


```


```
