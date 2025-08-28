import anndata as ad
import os
import senepy as sp
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random 

random.seed(42)

# Set working directory
os.chdir('/fs/scratch/PAS2598/Morales/CSF_workspace/csf')

# Load atlas data
cd8 = sc.read_h5ad('cd8_yo_scored_8-12-25.h5ad')
print(cd8)
print(f"Number of cells: {cd8.n_obs}")
print(f"Number of genes: {cd8.n_vars}")

# View the metadata
print(cd8.obs.head())

# Check the merged object
print(cd8)
print(cd8.obs['cell_type'].value_counts())
print(cd8.obs['putative_sen'].value_counts())

# Count cells per sample
print("Cells per sample:")
print(cd8.obs['sample'].value_counts().sort_index())

# as a sorted table
sample_counts = cd8.obs['sample'].value_counts().sort_index()
print(f"\nTotal samples: {len(sample_counts)}")
print(f"Cells per sample range: {sample_counts.min()} - {sample_counts.max()}")
print(f"Mean cells per sample: {sample_counts.mean():.1f}")

# create groupings
grouping_vars = ['sample', 'putative_sen']

# create the pseudobulking function
def pseudobulk_adata(adata, groupby_vars, layer='raw_counts'):
    """
    Pseudobulk AnnData object by specified grouping variables using raw_counts layer
    """
    # Check if raw_counts layer exists
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object. Available layers: {list(adata.layers.keys())}")
    
    print(f"Using layer: '{layer}' for pseudobulking")
    
    # Create grouping key
    adata.obs['pseudobulk_id'] = adata.obs[groupby_vars].astype(str).agg('_'.join, axis=1)
    
    # Get the raw count matrix specifically from raw_counts layer
    raw_counts_matrix = adata.layers['raw_counts']
    print(f"Raw counts matrix shape: {raw_counts_matrix.shape}")
    print(f"Raw counts matrix type: {type(raw_counts_matrix)}")
    
    # Convert to dense array if sparse
    if hasattr(raw_counts_matrix, 'toarray'):
        raw_counts_dense = raw_counts_matrix.toarray()
    else:
        raw_counts_dense = raw_counts_matrix
    
    # Create pseudobulk matrix using raw counts
    pseudobulk_df = pd.DataFrame(raw_counts_dense, 
                                index=adata.obs_names, 
                                columns=adata.var_names)
    pseudobulk_df['pseudobulk_id'] = adata.obs['pseudobulk_id'].values
    
    # Sum raw counts for each pseudobulk group
    print("Summing raw counts by pseudobulk groups...")
    pseudobulk_counts = pseudobulk_df.groupby('pseudobulk_id').sum()
    
    # Create metadata for pseudobulk samples
    # Keep all relevant sample-level metadata
    metadata_cols = groupby_vars + ['sex', 'disease', 'age', 'organ', 'dataset', 'disease_group', 'batch']
    available_cols = [col for col in metadata_cols if col in adata.obs.columns]
    
    pseudobulk_meta = adata.obs.groupby('pseudobulk_id')[available_cols].first()
    
    # Add cell count information
    pseudobulk_meta['n_cells'] = adata.obs.groupby('pseudobulk_id').size()
    
    # Add cell type composition for each pseudobulk sample
    print("Calculating cell type composition...")
    cell_type_counts = adata.obs.groupby('pseudobulk_id')['cell_type'].value_counts().unstack(fill_value=0)
    cell_type_props = cell_type_counts.div(cell_type_counts.sum(axis=1), axis=0)
    
    # Add cell type proportions to metadata
    for cell_type in cell_type_props.columns:
        pseudobulk_meta[f'{cell_type}_prop'] = cell_type_props[cell_type]
        pseudobulk_meta[f'{cell_type}_count'] = cell_type_counts[cell_type]
    
    print(f"Pseudobulking complete!")
    print(f"Total raw counts in original data: {raw_counts_dense.sum():,.0f}")
    print(f"Total counts in pseudobulk data: {pseudobulk_counts.sum().sum():,.0f}")
    
    return pseudobulk_counts, pseudobulk_meta

# Verify raw_counts layer exists
print("Available layers in merged_adata:")
print(list(cd8.layers.keys()))

# Perform pseudobulking using raw_counts layer
print("\nPerforming pseudobulking by sample + senescence status using raw_counts layer...")
pb_counts, pb_meta = pseudobulk_adata(cd8, 
                                     groupby_vars=['sample', 'putative_sen'],
                                     layer='raw_counts')

print(f"\nCreated {pb_counts.shape[0]} pseudobulk samples with {pb_counts.shape[1]} genes")
print(f"Pseudobulk counts range: {pb_counts.min().min()} to {pb_counts.max().max()}")
print("\nPseudobulk sample breakdown:")
print(pb_meta[['putative_sen', 'n_cells']].groupby('putative_sen').agg(['count', 'mean']))

# Show sample of the data
print(f"\nFirst few pseudobulk samples:")
print(pb_meta[['putative_sen', 'n_cells']])

# filtering function
def filter_pseudobulk_genes_only(counts, metadata, min_counts=3, min_sample_fraction=0.3):
    """
    Filter pseudobulk data for low count genes only
    - Keep genes with at least min_counts in at least min_sample_fraction of samples
    - Keep ALL samples (no sample filtering)
    """
    print(f"Before filtering: {counts.shape[0]} samples, {counts.shape[1]} genes")
    
    # Calculate minimum number of samples (30% of total samples)
    min_samples = int(np.ceil(counts.shape[0] * min_sample_fraction))
    print(f"Requiring ≥{min_counts} counts in ≥{min_samples} samples ({min_sample_fraction*100}% of {counts.shape[0]} samples)")
    
    # Filter genes ONLY - keep genes with ≥min_counts in ≥min_samples
    genes_pass_filter = (counts >= min_counts).sum(axis=0) >= min_samples
    
    counts_final = counts.loc[:, genes_pass_filter]
    meta_final = metadata  # Keep all sample metadata unchanged
    
    print(f"After gene filtering: {counts_final.shape[0]} samples, {counts_final.shape[1]} genes")
    print(f"Removed {counts.shape[1] - counts_final.shape[1]} genes with insufficient counts")
    print(f"Kept all {counts.shape[0]} samples")
    print(f"Senescent samples: {(meta_final['putative_sen'] == 1).sum()}")
    print(f"Non-senescent samples: {(meta_final['putative_sen'] == 0).sum()}")
    
    # Show filtering stats
    print(f"\nFiltering summary:")
    print(f"- Genes kept: {genes_pass_filter.sum():,} / {len(genes_pass_filter):,} ({genes_pass_filter.mean()*100:.1f}%)")
    print(f"- Samples kept: {counts_final.shape[0]} / {counts.shape[0]} (100%)")
    
    return counts_final, meta_final

# Apply gene-only filtering
pb_counts_filt, pb_meta_filt = filter_pseudobulk_genes_only(pb_counts, pb_meta, 
                                                           min_counts=3, 
                                                           min_sample_fraction=0.3)

# Show some examples of what got filtered
print(f"\nExample of count distribution for a few genes:")
sample_genes = pb_counts.columns[:5]  # first 5 genes
for gene in sample_genes:
    n_samples_with_counts = (pb_counts[gene] >= 3).sum()
    pct_samples = n_samples_with_counts / pb_counts.shape[0] * 100
    kept = gene in pb_counts_filt.columns
    print(f"{gene}: ≥3 counts in {n_samples_with_counts}/{pb_counts.shape[0]} samples ({pct_samples:.1f}%) - {'KEPT' if kept else 'FILTERED'}")

# Export raw filtered counts (genes as rows, samples as columns for DESeq2)
raw_counts_df = pd.DataFrame(pb_counts_filt.values.T,  # Transpose: genes as rows
                           index=pb_counts_filt.columns,   # gene names as row names
                           columns=pb_counts_filt.index)   # sample names as column names

raw_counts_df.to_csv('cd8_yo_pseudobulk_raw_counts_for_deseq2.csv')

# Export metadata (samples as rows)
pb_meta_filt.to_csv('cd8_yo_pseudobulk_metadata_for_deseq2.csv')
