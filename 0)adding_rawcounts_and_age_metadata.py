#load modules
import os
import random 
import senepy as sp
import scanpy as sc
import pandas as pd

# Set working directory
os.chdir('/fs/scratch/PAS2598/Morales/CSF_workspace/csf')

# Set seed
random.seed(42)

# Load data and check the structure
adata = sc.read_h5ad('atlas.h5ad')
adata.obs.head()

# Load raw counts
raw_counts = sc.read_h5ad('atlas_raw_counts.h5ad')

# check raw counts structure 
raw_counts.obs.head()
raw_counts

# Check cell ID formats
print("=== CELL ID FORMAT INVESTIGATION ===")

# Look at cell names in both datasets
print("Processed data cell names (first 5):")
print(list(adata.obs_names[:5]))

print("Raw counts cell names (first 5):")
print(list(raw_counts.obs_names[:5]))

# Check if they have different suffixes/prefixes
processed_sample = adata.obs_names[0]
raw_sample = raw_counts.obs_names[0]

print(f"\nExample processed cell: '{processed_sample}'")
print(f"Example raw cell: '{raw_sample}'")

# Check if there's a pattern (e.g., sample suffix differences)
print(f"\nProcessed cell length: {len(processed_sample)}")
print(f"Raw cell length: {len(raw_sample)}")

# Check sample information
print(f"\nSamples in processed data:")
print(adata.obs['sample'].unique()[:5])

if 'sample' in raw_counts.obs.columns:
    print(f"Samples in raw data:")
    print(raw_counts.obs['sample'].unique()[:5])
else:
    print("No 'sample' column in raw data")
    print("Raw data obs columns:", raw_counts.obs.columns.tolist())

# Check alignment
print(f"\n=== ALIGNMENT CHECK ===")
print(f"Processed data (adata): {adata.shape}")
print(f"Raw counts data: {raw_counts.shape}")

# Check cell overlap
processed_cells = set(adata.obs_names)
raw_cells = set(raw_counts.obs_names)
cell_overlap = processed_cells.intersection(raw_cells)

print(f"Processed cells: {len(processed_cells)}")
print(f"Raw count cells: {len(raw_cells)}")
print(f"Overlapping cells: {len(cell_overlap)}")

# Check sample overlap if possible
if 'sample' in raw_counts.obs.columns:
    processed_samples = set(adata.obs['sample'])
    raw_samples = set(raw_counts.obs['sample'])
    sample_overlap = processed_samples.intersection(raw_samples)
    
    print(f"\nProcessed samples: {len(processed_samples)}")
    print(f"Raw samples: {len(raw_samples)}")
    print(f"Overlapping samples: {len(sample_overlap)}")
    
    if len(sample_overlap) > 0:
        print("Can proceed with sample-level pseudobulk!")
        print(f"Common samples: {list(sample_overlap)[:5]}")

# Add raw counts to your processed adata object
print("=== ADDING RAW COUNTS TO PROCESSED ADATA ===")

# Make sure both objects have the same cell and gene order
print("Ensuring same order...")

# Reorder raw_counts to match adata exactly
raw_counts_aligned = raw_counts[adata.obs_names, adata.var_names]

print(f"adata shape: {adata.shape}")
print(f"raw_counts_aligned shape: {raw_counts_aligned.shape}")

# Add raw counts as a layer
if hasattr(raw_counts_aligned.X, 'toarray'):
    adata.layers['raw_counts'] = raw_counts_aligned.X.toarray()
else:
    adata.layers['raw_counts'] = raw_counts_aligned.X.copy()

print("Raw counts added as layer!")

# Verify the addition
print(f"Raw counts layer shape: {adata.layers['raw_counts'].shape}")
print(f"Raw counts layer max: {adata.layers['raw_counts'].max():,.0f}")
print(f"Raw counts layer min: {adata.layers['raw_counts'].min():,.0f}")

# Check a specific gene to verify alignment
test_gene = 'CDKN1A'
if test_gene in adata.var_names:
    gene_idx = list(adata.var_names).index(test_gene)
    raw_values = adata.layers['raw_counts'][:10, gene_idx]  # First 10 cells
    print(f"\nSample raw values for {test_gene}: {raw_values}")

# check the structure again
adata.obs.head()
adata

# Read the age CSV
age_df = pd.read_csv('atlas_age.csv')
print(age_df.head())

# Merge the dataframes
adata.obs = adata.obs.merge(age_df, on='sample', how='left')

# Check that it worked
print("Age column added:", 'age' in adata.obs.columns)
print("Sample ages:", adata.obs[['sample', 'age']].head())
print(adata.obs[['sample', 'age']].head(10))

# Verify all cells from same sample have same age
sample_ages = adata.obs.groupby('sample')['age'].nunique()
print("Each sample should have only 1 age:", sample_ages.max() == 1)

# age group creation as before
adata.obs['age_comparison'] = 'exclude'  # default
adata.obs.loc[adata.obs['age'] <= 25, 'age_comparison'] = '≤25 years'
adata.obs.loc[adata.obs['age'] >= 60, 'age_comparison'] = '≥60 years'

adata.write('adata_raw_age.h5ad')
