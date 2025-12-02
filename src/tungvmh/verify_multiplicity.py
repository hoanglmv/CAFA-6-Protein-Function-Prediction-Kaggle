import pandas as pd
import os

# Define file paths
DATA_DIR = (
    "/home/hoangtungvum/Data/ML_UET/CAFA-6-Protein-Function-Prediction-Kaggle/data"
)
TRAIN_TAXONOMY_PATH = os.path.join(DATA_DIR, "Train", "train_taxonomy.tsv")

print(f"Train Taxonomy Path: {TRAIN_TAXONOMY_PATH}")

# 1. Analyze Training Taxonomy
train_tax_df = pd.read_csv(
    TRAIN_TAXONOMY_PATH, sep="\t", header=None, names=["ProteinID", "TaxonomyID"]
)
print(f"Loaded {len(train_tax_df)} rows from train_taxonomy.tsv")

# 4. Check Protein-Taxonomy Multiplicity in Train
# Count unique taxonomies per protein
protein_tax_counts = train_tax_df.groupby("ProteinID")["TaxonomyID"].nunique()

# Check if any protein has > 1 taxonomy
multi_tax_proteins = protein_tax_counts[protein_tax_counts > 1]

print(f"Number of proteins with multiple taxonomies: {len(multi_tax_proteins)}")

if len(multi_tax_proteins) > 0:
    print("Examples:")
    print(multi_tax_proteins.head())
